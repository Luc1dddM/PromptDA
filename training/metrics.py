"""
promptda/model/metrics.py

Evaluation metrics for depth estimation.
Used to compare baseline and MLF runs.
"""

import torch


@torch.no_grad()
def compute_depth_metrics(
    pred: torch.Tensor,   # [B, 1, H, W]
    gt: torch.Tensor,     # [B, 1, H, W]
) -> dict:
    """
    Compute PromptDA-style depth metrics on one batch.

    Returns:
        `AbsRel`: Mean absolute relative error (lower is better).
        `delta1`: Ratio of valid pixels with threshold < 1.25 (higher is better).
        `delta2`: Ratio with threshold < $1.25^2$.
        `delta3`: Ratio with threshold < $1.25^3$.
    """
    mask   = (gt > 0).squeeze(1)          # [B, H, W]
    pred_m = pred.squeeze(1)[mask]
    gt_m   = gt.squeeze(1)[mask]

    if pred_m.numel() == 0:
        return {"AbsRel": 0.0, "delta1": 0.0, "delta2": 0.0, "delta3": 0.0}

    # Scale-align prediction → GT (median scaling)
    scale  = gt_m.median() / (pred_m.median() + 1e-8)
    pred_m = (pred_m * scale).clamp(min=1e-8)

    abs_rel = ((pred_m - gt_m).abs() / (gt_m + 1e-8)).mean().item()

    ratio  = torch.max(pred_m / (gt_m + 1e-8), gt_m / (pred_m + 1e-8))
    delta1 = (ratio < 1.25     ).float().mean().item()
    delta2 = (ratio < 1.25 ** 2).float().mean().item()
    delta3 = (ratio < 1.25 ** 3).float().mean().item()

    return {
        "AbsRel": abs_rel,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
    }


def aggregate_metrics(metrics_list: list[dict]) -> dict:
    """
    Average a list of metric dictionaries into one dictionary.
    """
    keys = metrics_list[0].keys()
    return {
        k: sum(m[k] for m in metrics_list) / len(metrics_list)
        for k in keys
    }