import math
import sys

import torch
import torch.nn.functional as F

from data.ARKitScenes.depth_upsampling import dataset_keys as arkit_dataset_keys

sys.modules.setdefault("dataset_keys", arkit_dataset_keys)

from data.ARKitScenes.depth_upsampling.losses.l1_loss import l1_loss
from data.ARKitScenes.depth_upsampling.losses.rmse import rmse_loss


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
) -> dict:
    valid_mask = (gt > 0) & torch.isfinite(gt) & torch.isfinite(pred)

    inputs = {
        arkit_dataset_keys.HIGH_RES_DEPTH_IMG: gt,
        arkit_dataset_keys.VALID_MASK_IMG: valid_mask,
    }
    outputs = {
        arkit_dataset_keys.PREDICTION_DEPTH_IMG: pred,
    }

    l1_value = l1_loss(outputs, inputs).item()
    rmse_value = rmse_loss(outputs, inputs).item()

    if not valid_mask.any():
        metrics = {
            "L1": l1_value,
            "RMSE": rmse_value,
            "AbsRel": float("nan"),
            "delta1": float("nan"),
            "delta2": float("nan"),
            "delta3": float("nan"),
        }
        if rgb is not None:
            metrics["BoundaryAbsRel"] = float("nan")
        return metrics

    pred_valid = pred[valid_mask].clamp_min(eps)
    gt_valid = gt[valid_mask].clamp_min(eps)
    ratio = torch.maximum(pred_valid / gt_valid, gt_valid / pred_valid)

    metrics = {
        "L1": l1_value,
        "RMSE": rmse_value,
        "AbsRel": torch.mean(torch.abs(pred_valid - gt_valid) / gt_valid).item(),
        "delta1": torch.mean((ratio < 1.25).float()).item(),
        "delta2": torch.mean((ratio < 1.25 ** 2).float()).item(),
        "delta3": torch.mean((ratio < 1.25 ** 3).float()).item(),
    }

    if rgb is not None:
        metrics["BoundaryAbsRel"] = compute_boundary_absrel(
            pred,
            gt,
            rgb,
            edge_threshold=edge_threshold,
            edge_dilate_kernel=edge_dilate_kernel,
            eps=eps,
        )

    return metrics


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
