"""
Visualization utilities for aleatoric uncertainty in depth completion.

Produces a 1×4 figure: RGB | Target Depth | Predicted μ | Predicted σ (Uncertainty)
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Detach, move to CPU, and convert to numpy (H, W) float32."""
    if tensor is None:
        return None
    t = tensor.detach().cpu()
    while t.ndim > 2:
        t = t.squeeze(0)   # remove batch dim, then channel dim
    return t.float().numpy()


def _normalize_sigma(sigma: np.ndarray, percentile: float = 99.0) -> tuple[np.ndarray, float]:
    """Clip sigma to [0, percentile] range for visualization, return (clipped, vmax)."""
    vmax = np.percentile(sigma, percentile)
    if vmax <= 0:
        vmax = 1.0
    clipped = np.clip(sigma, 0.0, vmax)
    return clipped, vmax


def plot_uncertainty(
    rgb: torch.Tensor | None = None,
    target: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    sigma: torch.Tensor | None = None,
    save_path: str | None = None,
    sigma_percentile: float = 99.0,
    figsize: tuple = (20, 5),
    dpi: int = 150,
):
    """Draw a 1×4 figure for uncertainty visualization.

    Args:
        rgb:          RGB image tensor [3, H, W] or [1, 3, H, W], values in [0, 1] or [0, 255].
        target:       GT depth tensor [1, H, W] or [H, W].
        mu:           Predicted depth tensor [1, H, W] or [H, W].
        sigma:        Predicted uncertainty (σ) tensor [1, H, W] or [H, W].
        save_path:    If provided, save figure to this path.
        sigma_percentile: Percentile for clipping σ outliers (default 99).
        figsize:      Figure size (width, height).
        dpi:          Output resolution.
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize, dpi=dpi)

    # --- Column 1: RGB ---
    ax = axes[0]
    if rgb is not None:
        rgb_np = _to_numpy(rgb)
        if rgb_np.ndim == 3 and rgb_np.shape[0] == 3:
            rgb_np = np.transpose(rgb_np, (1, 2, 0))   # CHW → HWC
        if rgb_np.max() > 1.5:
            rgb_np = rgb_np / 255.0
        rgb_np = np.clip(rgb_np, 0.0, 1.0)
        ax.imshow(rgb_np)
        ax.set_title("RGB")
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("RGB")
    ax.axis("off")

    # --- Column 2: Target Depth ---
    ax = axes[1]
    if target is not None:
        target_np = _to_numpy(target)
        valid = target_np > 0
        vmin = target_np[valid].min() if valid.any() else 0
        vmax = target_np[valid].max() if valid.any() else 1
        ax.imshow(target_np, cmap="jet", vmin=vmin, vmax=vmax)
        ax.set_title(f"Target Depth\n[{vmin:.2f}, {vmax:.2f}]")
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Target Depth")
    ax.axis("off")

    # --- Column 3: Predicted μ ---
    ax = axes[2]
    if mu is not None:
        mu_np = _to_numpy(mu)
        valid = target_np > 0 if target is not None and target_np is not None else np.ones_like(mu_np, dtype=bool)
        # Use target's vmin/vmax if available for fair comparison
        if target is not None:
            t_np = _to_numpy(target)
            tv = t_np[t_np > 0]
            if tv.size:
                vmin_t, vmax_t = tv.min(), tv.max()
        else:
            vmin_t, vmax_t = mu_np.min(), mu_np.max()
        ax.imshow(mu_np, cmap="jet", vmin=vmin_t, vmax=vmax_t)
        ax.set_title(f"Predicted μ\n[{vmin_t:.2f}, {vmax_t:.2f}]")
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Predicted μ")
    ax.axis("off")

    # --- Column 4: Predicted σ (Uncertainty Map) ---
    ax = axes[3]
    if sigma is not None:
        sigma_np = _to_numpy(sigma)
        sigma_clipped, vmax_s = _normalize_sigma(sigma_np, sigma_percentile)
        im = ax.imshow(sigma_clipped, cmap="magma", vmin=0.0, vmax=vmax_s)
        ax.set_title(f"Uncertainty σ\n[0, {vmax_s:.4f}] (p{sigma_percentile})")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Uncertainty σ")
    ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return fig