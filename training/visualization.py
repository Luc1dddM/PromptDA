import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import cv2
import argparse, os, sys

sys.path.insert(0, os.path.abspath("."))

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from data.ARKitScenes.depth_upsampling import dataset_keys as arkit_dataset_keys
from data.ARKitScenes.depth_upsampling import image_utils as arkit_image_utils
import sys
sys.modules.setdefault("dataset_keys", arkit_dataset_keys)
sys.modules.setdefault("image_utils", arkit_image_utils)

from data.ARKitScenes.depth_upsampling import transfroms
from dataset.dataset import MyARKitScenesDataset, collate_fn
from dataset import dataset_keys
from promptda.promptda_uncertainty import PromptDAUncertainty


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def build_transforms():
    return Compose([
        transfroms.RandomFilpLR(),
        transfroms.ValidDepthMask(gt_low_limit=0.01),
        transfroms.AsContiguousArray(),
    ])


def load_one_sample(data_root, seed=42):
    ds = MyARKitScenesDataset(
        root=data_root, split="train",
        max_samples=None, seed=seed,
        transform=build_transforms(),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=True,
                        num_workers=0, collate_fn=collate_fn)
    batch = next(iter(loader))
    batch = {k: v for k, v in batch.items()
             if k not in ("bounding_box", "bounding_box_image")}
    return batch


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model, batch, device):
    model.eval()
    with torch.no_grad():
        image  = batch[dataset_keys.COLOR_IMG].to(device)
        prompt = batch[dataset_keys.LOW_RES_DEPTH_IMG].to(device)
        pred   = model(image, prompt)

    mu = pred["mu"]
    s  = pred["s"]

    gt_shape = batch[dataset_keys.HIGH_RES_DEPTH_IMG].shape[-2:]
    if mu.shape[-2:] != gt_shape:
        mu = torch.nn.functional.interpolate(
            mu, size=gt_shape, mode="bilinear", align_corners=False)
        s = torch.nn.functional.interpolate(
            s, size=gt_shape, mode="bilinear", align_corners=False)

    mu          = mu.squeeze().cpu().numpy()
    uncertainty = np.exp(s.squeeze().cpu().numpy())  # b = exp(s), Laplace scale
    return mu, uncertainty


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred, gt):
    """Compute RMSE and MAE on valid (gt > 0) pixels."""
    mask = gt > 0
    if mask.sum() == 0:
        return float("nan"), float("nan")
    err  = pred[mask] - gt[mask]
    rmse = float(np.sqrt((err ** 2).mean()))
    mae  = float(np.abs(err).mean())
    return rmse, mae


# ─────────────────────────────────────────────────────────────────────────────
# Confidence-guided refinement (sparse LiDAR fusion)
# ─────────────────────────────────────────────────────────────────────────────

def upsample_lidar(lidar, target_shape):
    """Nearest-neighbor upsample sparse LiDAR to target H×W (preserves zeros)."""
    if lidar.shape == target_shape:
        return lidar
    lidar_up = cv2.resize(
        lidar.astype(np.float32),
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return lidar_up


def confidence_guided_refine(mu, uncertainty, lidar, rgb):
    """
    Uncertainty-aware sparse-LiDAR fusion:

    1. Joint bilateral filter on mu guided by RGB → spatially-smooth refined depth.
    2. Confidence  = 1 / (b + eps),  normalised to [0, 1].
    3. At pixels where LiDAR is valid:
           output = conf * refined  +  (1 - conf) * lidar
       (high confidence → trust the smooth prediction;
        low confidence → pull toward measured LiDAR)
    4. At pixels where LiDAR is missing: output = refined.
    """
    H, W = mu.shape

    # -- upsample sparse lidar to GT resolution --------------------------
    lidar_up = upsample_lidar(lidar, (H, W))

    # -- confidence map --------------------------------------------------
    eps        = 1e-6
    conf_raw   = 1.0 / (uncertainty + eps)
    conf       = (conf_raw - conf_raw.min()) / (conf_raw.ptp() + eps)  # [0,1]

    # -- joint bilateral filter (RGB-guided smoothing of mu) -------------
    # standard bilateral filter on mu (no contrib needed)
    mu_max  = mu.max() if mu.max() > 0 else 1.0
    mu_f32  = mu.astype(np.float32)
    refined = cv2.bilateralFilter(mu_f32, d=9, sigmaColor=0.3, sigmaSpace=9)

    # -- sparse fusion ---------------------------------------------------
    output      = refined.copy()
    lidar_valid = lidar_up > 0
    output[lidar_valid] = (
        conf[lidar_valid]       * refined[lidar_valid]
        + (1 - conf[lidar_valid]) * lidar_up[lidar_valid]
    )

    return output, refined, conf


# ─────────────────────────────────────────────────────────────────────────────
# Calibration analysis
# ─────────────────────────────────────────────────────────────────────────────

def calibration_analysis(mu, uncertainty, gt, n_bins=10):
    """
    Bin pixels by predicted uncertainty, report mean abs-error per bin.
    Good calibration → mean error increases monotonically with uncertainty.
    """
    mask     = gt > 0
    unc_flat = uncertainty[mask].ravel()
    err_flat = np.abs(mu[mask] - gt[mask]).ravel()

    bin_edges = np.percentile(unc_flat, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-6  # include max

    bin_centers, mean_errs, std_errs, counts = [], [], [], []
    for i in range(n_bins):
        m = (unc_flat >= bin_edges[i]) & (unc_flat < bin_edges[i + 1])
        if m.sum() == 0:
            continue
        bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
        mean_errs.append(err_flat[m].mean())
        std_errs.append(err_flat[m].std())
        counts.append(m.sum())

    return np.array(bin_centers), np.array(mean_errs), np.array(std_errs), np.array(counts)


# ─────────────────────────────────────────────────────────────────────────────
# Visualize
# ─────────────────────────────────────────────────────────────────────────────

def visualize(batch, mu, uncertainty, save_path="uncertainty_vis.png"):
    rgb    = batch[dataset_keys.COLOR_IMG].squeeze().permute(1, 2, 0).numpy()
    rgb    = np.clip(rgb, 0, 1)
    gt     = batch[dataset_keys.HIGH_RES_DEPTH_IMG].squeeze().numpy()
    lidar  = batch[dataset_keys.LOW_RES_DEPTH_IMG].squeeze().numpy()

    # ── refinement ──────────────────────────────────────────────────────
    refined, jbf_only, conf = confidence_guided_refine(mu, uncertainty, lidar, rgb)

    # ── metrics ─────────────────────────────────────────────────────────
    rmse_mu,      mae_mu      = compute_metrics(mu,      gt)
    rmse_jbf,     mae_jbf     = compute_metrics(jbf_only, gt)
    rmse_refined, mae_refined = compute_metrics(refined,  gt)

    print("\n──────────────── Metrics (valid GT pixels only) ────────────────")
    print(f"  Raw prediction  mu   : RMSE={rmse_mu:.4f}  MAE={mae_mu:.4f}")
    print(f"  JBF only (no fusion) : RMSE={rmse_jbf:.4f}  MAE={mae_jbf:.4f}")
    print(f"  Conf-guided refined  : RMSE={rmse_refined:.4f}  MAE={mae_refined:.4f}")
    delta_rmse = rmse_mu - rmse_refined
    delta_mae  = mae_mu  - mae_refined
    print(f"  Δ RMSE (mu→refined)  : {delta_rmse:+.4f}  ({'↓ improved' if delta_rmse > 0 else '↑ worse'})")
    print(f"  Δ MAE  (mu→refined)  : {delta_mae:+.4f}  ({'↓ improved' if delta_mae  > 0 else '↑ worse'})")
    print("────────────────────────────────────────────────────────────────\n")

    # ── calibration bins ────────────────────────────────────────────────
    bin_centers, mean_errs, std_errs, counts = calibration_analysis(mu, uncertainty, gt)
    spearman = np.corrcoef(bin_centers, mean_errs)[0, 1]
    print(f"  Uncertainty–error Pearson r (bin means): {spearman:.4f}")
    print(f"  (r > 0.7 → well-calibrated, r < 0.3 → poorly calibrated)\n")

    # ── depth range for consistent colormap ─────────────────────────────
    valid = gt[gt > 0]
    vmin_d = np.percentile(valid, 2)
    vmax_d = np.percentile(valid, 98)
    err_gt = np.abs(mu - gt); err_gt[gt == 0] = np.nan

    # ════════════════════════════════════════════════════════════════════
    # Figure 1 — depth maps (7 panels)
    # ════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 7, figsize=(32, 4.5),
                             gridspec_kw={"wspace": 0.05})

    panels = [
        ("RGB input",              rgb,        None,    None,    None),
        ("LiDAR prompt (sparse)",  lidar,      "plasma", vmin_d, vmax_d),
        ("GT depth",               gt,         "plasma", vmin_d, vmax_d),
        ("Predicted μ",            mu,         "plasma", vmin_d, vmax_d),
        ("Uncertainty b=eˢ",       uncertainty,"hot",    None,   None),
        ("JBF smooth (no fusion)", jbf_only,   "plasma", vmin_d, vmax_d),
        ("Conf-guided refined",    refined,    "plasma", vmin_d, vmax_d),
    ]

    for ax, (title, img, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        if cmap is not None:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    metric_text = (
        f"RMSE  μ={rmse_mu:.4f}  JBF={rmse_jbf:.4f}  refined={rmse_refined:.4f}\n"
        f"MAE   μ={mae_mu:.4f}  JBF={mae_jbf:.4f}  refined={mae_refined:.4f}"
    )
    fig.text(0.5, -0.01, metric_text, ha="center", fontsize=9,
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", lw=0.5))

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")

    # ════════════════════════════════════════════════════════════════════
    # Figure 2 — analysis (3 subplots)
    # ════════════════════════════════════════════════════════════════════
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4.5))

    # -- 2a: scatter uncertainty vs abs-error ----------------------------
    ax = axes2[0]
    mask     = gt > 0
    unc_flat = uncertainty[mask].ravel()
    err_flat = err_gt[mask].ravel()
    idx      = np.random.choice(len(unc_flat), min(8000, len(unc_flat)), replace=False)
    ax.scatter(unc_flat[idx], err_flat[idx],
               s=4, alpha=0.25, c="#7F77DD", linewidths=0)
    ax.set_xlabel("Uncertainty b")
    ax.set_ylabel("|μ − GT|")
    ax.set_title(f"Calibration scatter\nPearson r={spearman:.3f}")

    # -- 2b: binned calibration curve ------------------------------------
    ax = axes2[1]
    ax.errorbar(bin_centers, mean_errs, yerr=std_errs,
                fmt="o-", color="#1D9E75", capsize=4, linewidth=1.5)
    ax.set_xlabel("Uncertainty bin center")
    ax.set_ylabel("Mean |μ − GT|")
    ax.set_title("Calibration curve\n(monotone ↑ = well-calibrated)")

    # -- 2c: RMSE bar comparison -----------------------------------------
    ax = axes2[2]
    labels = ["μ (raw)", "JBF only", "conf-guided\nrefined"]
    rmses  = [rmse_mu, rmse_jbf, rmse_refined]
    colors = ["#B4B2A9", "#85B7EB", "#1D9E75"]
    bars   = ax.bar(labels, rmses, color=colors, width=0.5,
                    edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE comparison")
    ax.set_ylim(0, max(rmses) * 1.2)

    fig2.tight_layout()
    scatter_path = save_path.replace(".png", "_analysis.png")
    fig2.savefig(scatter_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {scatter_path}")

    plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/ARKitScenes/data/upsampling")
    p.add_argument("--ckpt",      default=None,
                   help="Path to uncertainty checkpoint (.pth)")
    p.add_argument("--encoder",   default="vits")
    p.add_argument("--variant",   default="legacy")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--save",      default="uncertainty_vis.png")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load model (bypass from_pretrained wrapper) ──────────────────────
    model = PromptDAUncertainty(
        encoder=args.encoder,
        dpt_variant=args.variant,
    ).to(device)

    ckpt       = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[load] missing={len(missing)}  unexpected={len(unexpected)}")
    if missing:
        print("  first 3 missing:", missing[:3])

    model.eval()

    batch = load_one_sample(args.data_root, seed=args.seed)
    mu, uncertainty = run_inference(model, batch, device)

    # ── sanity check ─────────────────────────────────────────────────────
    print(f"mu  — min:{mu.min():.4f}  max:{mu.max():.4f}  std:{mu.std():.4f}")
    print(f"unc — min:{uncertainty.min():.4f}  max:{uncertainty.max():.4f}  std:{uncertainty.std():.4f}")

    visualize(batch, mu, uncertainty, save_path=args.save)