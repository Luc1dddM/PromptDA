"""
Baseline inference script for PromptDA.

Usage:
    python inference_baseline.py \
        --data_root data/ARKitScenes/data/upsampling \
        --encoder vitl \
        --pretrained_path /path/to/model.ckpt \
        --output_dir results/baseline
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dataset.dataset import MyARKitScenesDataset
from promptda.promptda_baseline import PromptDA
from promptda.utils.logger import Log
from training.metrics import compute_depth_metrics, aggregate_metrics


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="PromptDA Baseline Inference")

    p.add_argument("--data_root",       type=str, default="data/ARKitScenes/data/upsampling")
    p.add_argument("--split",           type=str, default="val", choices=["train", "val"])
    p.add_argument("--max_samples",     type=int, default=None)
    p.add_argument("--num_workers",     type=int, default=4)

    p.add_argument("--encoder",         type=str, default="vitl",
                   choices=["vits", "vitb", "vitl"])
    p.add_argument("--pretrained_path", type=str,
                   default="depth-anything/prompt-depth-anything-vitl")

    p.add_argument("--max_size",        type=int, default=1008,
                   help="Max longer-side of image (floored to multiple of 14). "
                        "Default 1008 = 72x14.")

    p.add_argument("--output_dir",      type=str, default="results/baseline")
    p.add_argument("--save_depth",      action="store_true")
    p.add_argument("--batch_size",      type=int, default=1)

    return p.parse_args()


# --------------------------------------------------------------------------- #
# Per-sample predict wrapper
# --------------------------------------------------------------------------- #

def run_batch(model, image, prompt):
    """
    image: (B, 3, H, W)
    prompt: (B, 1, h, w)
    returns: pred: (B, 1, H, W)
    """
    preds = []
    for i in range(image.shape[0]):
        depth_i = model.predict(
            image[i].unsqueeze(0),   # (1, 3, H, W)
            prompt[i].unsqueeze(0),  # (1, 1, h, w)
        )

        # Normalize về đúng (1, H, W)
        depth_i = depth_i.squeeze()          # loại bỏ TẤT CẢ dim=1 -> (H, W)
        depth_i = depth_i.unsqueeze(0)       # thêm channel dim -> (1, H, W)

        preds.append(depth_i)

    return torch.stack(preds, dim=0)         # (B, 1, H, W)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Log.info(f"Device        : {device}")
    Log.info(f"Encoder       : {args.encoder}")
    Log.info(f"Split         : {args.split}")
    Log.info(f"Batch size    : {args.batch_size}")
    Log.info(f"Max img size  : {args.max_size} (multiple of 14)")
    Log.info(f"Pretrained    : {args.pretrained_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_depth:
        depth_dir = os.path.join(args.output_dir, "depth_maps")
        os.makedirs(depth_dir, exist_ok=True)

    # ── Transform ───────────────────────────────────────────────────────── #
    # ImageTransform: resize IMAGE tensor (C,H,W) -> multiple of 14
    # Dataset goi: sample["color_img"] = self.transform(sample["color_img"])

    # ── Dataset ─────────────────────────────────────────────────────────── #
    Log.info(f"Loading '{args.split}' dataset from: {args.data_root}")

    dataset = MyARKitScenesDataset(
        root=args.data_root,
        split=args.split,
    )

    if args.max_samples is not None:
        dataset = torch.utils.data.Subset(
            dataset, range(min(args.max_samples, len(dataset)))
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    Log.info(f"Dataset size  : {len(dataset)} | Batches: {len(loader)}")

    # ── Model ────────────────────────────────────────────────────────────── #
    Log.info("Loading PromptDA model...")
    model = PromptDA.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_path
    ).to(device).eval()
    Log.info("Model ready.")

    # ── Inference ────────────────────────────────────────────────────────── #
    all_metrics = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Inference")):

            image    = batch["color_img"].to(device)           # (B, 3, H, W)
            depth_gt = batch["high_res_depth_img"].to(device)  # (B, 1, H, W)
            prompt   = batch["low_res_depth_img"].to(device)   # (B, 1, h, w)

            bouding_boxes = batch["bounding_box"]                    # list of (B, N, 4)
            # bouding_boxes = [b.to(device) for b in bouding_boxes]
            print(bouding_boxes)

            pred = run_batch(model, image, prompt)             # (B, 1, H, W)

            if pred.shape[-2:] != depth_gt.shape[-2:]:
                pred = torch.nn.functional.interpolate(
                    pred,
                    size=depth_gt.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            all_metrics.append(compute_depth_metrics(pred, depth_gt))

            if args.save_depth:
                for i in range(pred.shape[0]):
                    fname = f"batch_{batch_idx:04d}_sample_{i:02d}.npy"
                    np.save(os.path.join(depth_dir, fname), pred[i, 0].cpu().numpy())

    # ── Results ──────────────────────────────────────────────────────────── #
    agg = aggregate_metrics(all_metrics)

    Log.info("=" * 60)
    Log.info(f"Baseline Results  [{args.split}]")
    Log.info("=" * 60)
    for k, v in sorted(agg.items()):
        Log.info(f"  {k:20s}: {v:.6f}")
    Log.info("=" * 60)

    txt_path = os.path.join(args.output_dir, "baseline_metrics.txt")
    with open(txt_path, "w") as f:
        f.write(f"Baseline Inference Results ({args.split})\n")
        f.write("=" * 60 + "\n")
        for k, v in sorted(agg.items()):
            f.write(f"  {k:20s}: {v:.6f}\n")
        f.write("=" * 60 + "\n")
    Log.info(f"Saved -> {txt_path}")

    npz_path = os.path.join(args.output_dir, "baseline_metrics.npz")
    np.savez(npz_path, **agg)
    Log.info(f"Saved -> {npz_path}")

    return agg


if __name__ == "__main__":
    main()