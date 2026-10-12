import math

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as compute_ssim

EPS = 1e-6


def _sobel_kernel(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    return kernel_x, kernel_y


def compute_edge_strength(rgb: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    gray = rgb.mean(dim=1, keepdim=True)
    kernel_x, kernel_y = _sobel_kernel(gray.device, gray.dtype)
    grad_x = F.conv2d(gray, kernel_x, padding=1)
    grad_y = F.conv2d(gray, kernel_y, padding=1)
    magnitude = torch.sqrt(grad_x.square() + grad_y.square() + eps)
    scale = magnitude.flatten(2).amax(dim=2, keepdim=True).view(-1, 1, 1, 1).clamp_min(eps)
    return magnitude / scale


def compute_edge_mask(
    rgb: torch.Tensor,
    threshold: float = 0.1,
    dilate_kernel: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_strength = compute_edge_strength(rgb)
    edge_mask = edge_strength > threshold
    if dilate_kernel > 1:
        edge_mask = F.max_pool2d(
            edge_mask.float(),
            kernel_size=dilate_kernel,
            stride=1,
            padding=dilate_kernel // 2,
        ) > 0
    return edge_mask, edge_strength


@torch.no_grad()
def compute_boundary_absrel(
    pred: torch.Tensor,
    gt: torch.Tensor,
    rgb: torch.Tensor,
    edge_threshold: float = 0.1,
    edge_dilate_kernel: int = 5,
    eps: float = EPS,
) -> float:
    valid_mask = (gt > 0) & torch.isfinite(gt) & torch.isfinite(pred)
    edge_mask, _ = compute_edge_mask(
        rgb,
        threshold=edge_threshold,
        dilate_kernel=edge_dilate_kernel,
    )
    if edge_mask.shape[-2:] != gt.shape[-2:]:
        edge_mask = F.interpolate(
            edge_mask.float(),
            size=gt.shape[-2:],
            mode="nearest",
        ) > 0
    boundary_mask = valid_mask & edge_mask
    if not boundary_mask.any():
        return float("nan")

    pred_valid = pred[boundary_mask].clamp_min(eps)
    gt_valid = gt[boundary_mask].clamp_min(eps)
    return torch.mean(torch.abs(pred_valid - gt_valid) / gt_valid).item()


@torch.no_grad()
def compute_depth_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    rgb: torch.Tensor | None = None,
    edge_threshold: float = 0.1,
    edge_dilate_kernel: int = 5,
    eps: float = EPS,
    valid_mask: torch.Tensor | None = None,
) -> dict:
    if pred.shape != gt.shape:
        raise ValueError("Prediction and GT shapes must match")
    if pred.shape[0] != 1:
        return aggregate_metrics(compute_depth_metrics_per_image(
            pred, gt, rgb, valid_mask=valid_mask, edge_threshold=edge_threshold,
            edge_dilate_kernel=edge_dilate_kernel, eps=eps))
    valid_mask = ((gt > 0.01) & torch.isfinite(gt)) if valid_mask is None else (
        valid_mask.bool() & (gt > 0.01) & torch.isfinite(gt))
    if not torch.isfinite(pred[valid_mask]).all():
        raise ValueError("Non-finite prediction at a valid GT pixel")
    error = pred[valid_mask] - gt[valid_mask]
    l1_value = error.abs().mean().item()
    rmse_value = error.square().mean().sqrt().item()

    if not valid_mask.any():
        metrics = {
            "L1": l1_value,
            "RMSE": rmse_value,
            "AbsRel": float("nan"),
            "delta0.5": float("nan"),
            "delta1": float("nan"),
            "delta2": float("nan"),
            "delta3": float("nan"),
            "SSIM": float("nan"),
        }
        if rgb is not None:
            metrics["BoundaryAbsRel"] = float("nan")
        return metrics

    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask].clamp_min(eps)
    ratio = torch.maximum(pred_valid / gt_valid, gt_valid / pred_valid.clamp_min(eps))

    metrics = {
        "L1": l1_value,
        "RMSE": rmse_value,
        "AbsRel": torch.mean(torch.abs(pred_valid - gt_valid) / gt_valid).item(),
        "delta0.5": torch.mean((ratio < 1.25 ** 0.5).float()).item(),
        "delta1": torch.mean((ratio < 1.25).float()).item(),
        "delta2": torch.mean((ratio < 1.25 ** 2).float()).item(),
        "delta3": torch.mean((ratio < 1.25 ** 3).float()).item(),
        "SSIM": compute_ssim_metric(pred, torch.where(valid_mask, gt, 0)),
    }

    if rgb is not None:
        metrics["BoundaryAbsRel"] = compute_boundary_absrel(
            pred,
            torch.where(valid_mask, gt, 0),
            rgb,
            edge_threshold=edge_threshold,
            edge_dilate_kernel=edge_dilate_kernel,
            eps=eps,
        )

    return metrics


def compute_depth_metrics_per_image(pred, gt, rgb=None, valid_mask=None, **kwargs):
    """One record per frame; aggregate these records, never batch averages."""
    return [compute_depth_metrics(
        pred[i:i+1], gt[i:i+1], None if rgb is None else rgb[i:i+1],
        valid_mask=None if valid_mask is None else valid_mask[i:i+1], **kwargs)
        for i in range(pred.shape[0])]


def aggregate_metrics(metrics_list: list[dict]) -> dict:
    if not metrics_list:
        return {}

    keys = sorted({key for metrics in metrics_list for key in metrics.keys()})
    aggregated = {}
    for key in keys:
        values = []
        for metrics in metrics_list:
            if key not in metrics:
                continue
            value = float(metrics[key])
            if math.isnan(value):
                continue
            values.append(value)
        aggregated[key] = sum(values) / len(values) if values else float("nan")
    return aggregated


@torch.no_grad()
def compute_ssim_metric(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Image-level SSIM averaged over batch (skimage, returns value in [-1, 1])."""
    valid_mask = (gt > 0) & torch.isfinite(gt) & torch.isfinite(pred)
    ssim_vals = []
    for b in range(pred.shape[0]):
        p = pred[b, 0].cpu().numpy().copy()
        t = gt[b, 0].cpu().numpy().copy()
        v = valid_mask[b, 0].cpu().numpy().copy()
        if not v.any() or p.shape[0] < 7 or p.shape[1] < 7:
            ssim_vals.append(float("nan"))
            continue
        dr = max(float(t[v].max() - t[v].min()), 1e-6)
        if np.isnan(dr) or dr <= 0:
            ssim_vals.append(float("nan"))
            continue
        p[~v] = 0
        t[~v] = 0
        val = compute_ssim(p, t, data_range=dr, win_size=7)
        ssim_vals.append(float(val))
    values = [x for x in ssim_vals if not math.isnan(x)]
    return sum(values) / len(values) if values else float("nan")
