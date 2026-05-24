"""
Visualization utilities for depth prediction, SACG comparison, and uncertainty.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(tensor: torch.Tensor | np.ndarray | None) -> np.ndarray | None:
    if tensor is None:
        return None
    if isinstance(tensor, np.ndarray):
        arr = tensor
    else:
        arr = tensor.detach().cpu().float().numpy()
    while arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _rgb_to_numpy(rgb: torch.Tensor | np.ndarray | None) -> np.ndarray | None:
    rgb_np = _to_numpy(rgb)
    if rgb_np is None:
        return None
    if rgb_np.ndim == 3 and rgb_np.shape[0] == 3:
        rgb_np = np.transpose(rgb_np, (1, 2, 0))
    if rgb_np.max() > 1.5:
        rgb_np = rgb_np / 255.0
    return np.clip(rgb_np, 0.0, 1.0)


def _depth_to_numpy(depth: torch.Tensor | np.ndarray | None) -> np.ndarray | None:
    depth_np = _to_numpy(depth)
    if depth_np is None:
        return None
    if depth_np.ndim == 3 and depth_np.shape[0] == 1:
        depth_np = depth_np[0]
    if depth_np.ndim != 2:
        raise ValueError(f"Expected depth map with shape [H,W] or [1,H,W], got {depth_np.shape}")
    return depth_np.astype(np.float32, copy=False)


def _normalize_sigma(sigma: np.ndarray, percentile: float = 99.0) -> tuple[np.ndarray, float]:
    vmax = np.percentile(sigma, percentile)
    if vmax <= 0:
        vmax = 1.0
    clipped = np.clip(sigma, 0.0, vmax)
    return clipped, vmax


def _depth_limits(*depth_maps: np.ndarray | None) -> tuple[float, float]:
    for depth in depth_maps:
        if depth is None:
            continue
        valid = np.isfinite(depth) & (depth > 0)
        if valid.any():
            return float(depth[valid].min()), float(depth[valid].max())
    return 0.0, 1.0


def _error_limits(*error_maps: np.ndarray | None) -> float:
    max_value = 0.0
    for error in error_maps:
        if error is None:
            continue
        valid = np.isfinite(error)
        if valid.any():
            max_value = max(max_value, float(error[valid].max()))
    return max(max_value, 1e-6)


def _draw_depth(ax, depth: np.ndarray | None, title: str, vmin: float, vmax: float):
    if depth is None:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
    else:
        ax.imshow(depth, cmap="jet", vmin=vmin, vmax=vmax)
        ax.set_title(f"{title}\n[{vmin:.2f}, {vmax:.2f}]")
    ax.axis("off")


def _draw_map(
    ax,
    value_map: np.ndarray | None,
    title: str,
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    add_colorbar: bool = False,
):
    if value_map is None:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.axis("off")
        return

    im = ax.imshow(value_map, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    if add_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _draw_sparse_overlay(ax, rgb_np: np.ndarray | None, sparse_np: np.ndarray | None):
    if rgb_np is None and sparse_np is None:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Sparse LiDAR")
        ax.axis("off")
        return

    if rgb_np is not None:
        ax.imshow(rgb_np)
    elif sparse_np is not None:
        ax.imshow(np.zeros((*sparse_np.shape, 3), dtype=np.float32))

    if sparse_np is not None:
        valid = sparse_np > 0
        if valid.any():
            ax.imshow(np.where(valid, sparse_np, np.nan), cmap="turbo", alpha=0.85)
    ax.set_title("Sparse LiDAR")
    ax.axis("off")


def plot_uncertainty(
    rgb: torch.Tensor | np.ndarray | None = None,
    target: torch.Tensor | np.ndarray | None = None,
    mu: torch.Tensor | np.ndarray | None = None,
    sigma: torch.Tensor | np.ndarray | None = None,
    baseline: torch.Tensor | np.ndarray | None = None,
    save_path: str | None = None,
    sigma_percentile: float = 99.0,
    figsize: tuple | None = None,
    dpi: int = 150,
):
    rgb_np = _rgb_to_numpy(rgb)
    target_np = _depth_to_numpy(target)
    baseline_np = _depth_to_numpy(baseline)
    mu_np = _depth_to_numpy(mu)
    sigma_np = _depth_to_numpy(sigma)

    columns = [("rgb", "RGB"), ("target", "Target Depth")]
    if baseline_np is not None:
        columns.append(("baseline", "Baseline Depth"))
    if mu_np is not None:
        columns.append(("mu", "Predicted mu"))
    if sigma_np is not None:
        columns.append(("sigma", "Uncertainty sigma"))

    if figsize is None:
        figsize = (5 * len(columns), 5)

    fig, axes = plt.subplots(1, len(columns), figsize=figsize, dpi=dpi)
    axes = np.atleast_1d(axes)
    depth_vmin, depth_vmax = _depth_limits(target_np, baseline_np, mu_np)

    for ax, (kind, title) in zip(axes, columns):
        if kind == "rgb":
            if rgb_np is not None:
                ax.imshow(rgb_np)
                ax.set_title(title)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(title)
            ax.axis("off")
        elif kind == "target":
            _draw_depth(ax, target_np, title, depth_vmin, depth_vmax)
        elif kind == "baseline":
            _draw_depth(ax, baseline_np, title, depth_vmin, depth_vmax)
        elif kind == "mu":
            _draw_depth(ax, mu_np, title, depth_vmin, depth_vmax)
        elif kind == "sigma":
            sigma_clipped, vmax_s = _normalize_sigma(sigma_np, sigma_percentile)
            _draw_map(
                ax,
                sigma_clipped,
                f"Uncertainty sigma\n[0, {vmax_s:.4f}] (p{sigma_percentile:g})",
                cmap="magma",
                vmin=0.0,
                vmax=vmax_s,
                add_colorbar=True,
            )

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return fig


def plot_sacg_comparison(
    rgb: torch.Tensor | np.ndarray,
    sparse_depth: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    baseline_depth: torch.Tensor | np.ndarray,
    refined_depth: torch.Tensor | np.ndarray,
    gate_map: torch.Tensor | np.ndarray | None = None,
    c_grad: torch.Tensor | np.ndarray | None = None,
    f_lidar: torch.Tensor | np.ndarray | None = None,
    save_path: str | None = None,
    figsize: tuple = (24, 12),
    dpi: int = 150,
):
    rgb_np = _rgb_to_numpy(rgb)
    sparse_np = _depth_to_numpy(sparse_depth)
    target_np = _depth_to_numpy(target)
    baseline_np = _depth_to_numpy(baseline_depth)
    refined_np = _depth_to_numpy(refined_depth)
    gate_np = _depth_to_numpy(gate_map)
    c_grad_np = _depth_to_numpy(c_grad)
    f_lidar_np = _depth_to_numpy(f_lidar)

    baseline_error = None
    refined_error = None
    if target_np is not None and baseline_np is not None:
        baseline_error = np.abs(baseline_np - target_np)
    if target_np is not None and refined_np is not None:
        refined_error = np.abs(refined_np - target_np)

    depth_vmin, depth_vmax = _depth_limits(target_np, baseline_np, refined_np, sparse_np)
    err_vmax = _error_limits(baseline_error, refined_error)

    fig, axes = plt.subplots(2, 4, figsize=figsize, dpi=dpi)
    axes = axes.reshape(2, 4)

    if rgb_np is not None:
        axes[0, 0].imshow(rgb_np)
        axes[0, 0].set_title("RGB")
    else:
        axes[0, 0].text(0.5, 0.5, "N/A", ha="center", va="center", transform=axes[0, 0].transAxes)
        axes[0, 0].set_title("RGB")
    axes[0, 0].axis("off")

    _draw_sparse_overlay(axes[0, 1], rgb_np, sparse_np)
    _draw_depth(axes[0, 2], baseline_np, "Baseline Coarse", depth_vmin, depth_vmax)
    _draw_depth(axes[0, 3], refined_np, "SACG Refined", depth_vmin, depth_vmax)

    _draw_map(
        axes[1, 0],
        baseline_error,
        "Baseline Error",
        cmap="inferno",
        vmin=0.0,
        vmax=err_vmax,
        add_colorbar=True,
    )
    _draw_map(
        axes[1, 1],
        refined_error,
        "SACG Error",
        cmap="inferno",
        vmin=0.0,
        vmax=err_vmax,
        add_colorbar=True,
    )
    _draw_map(axes[1, 2], gate_np, "Gate Map", cmap="magma", vmin=0.0, vmax=1.0, add_colorbar=True)

    component_map = c_grad_np if c_grad_np is not None else f_lidar_np
    component_title = "C_grad" if c_grad_np is not None else "F_lidar"
    if c_grad_np is not None and f_lidar_np is not None:
        component_map = 0.5 * (c_grad_np + f_lidar_np)
        component_title = "0.5*(C_grad + F_lidar)"
    _draw_map(
        axes[1, 3],
        component_map,
        component_title,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        add_colorbar=True,
    )

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return fig
