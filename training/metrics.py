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
    Compute comprehensive depth estimation metrics on one batch.

    Returns dict with:
        `AbsRel`: Mean absolute relative error (lower is better).
        `MAE`: Mean absolute error in meters (lower is better).
        `RMSE`: Root mean squared error (lower is better).
        `Log10`: Mean log10 error (lower is better).
        `delta1`: Ratio of valid pixels with threshold < 1.25 (higher is better).
        `delta2`: Ratio with threshold < $1.25^2$ (higher is better).
        `delta3`: Ratio with threshold < $1.25^3$ (higher is better).
        `SILog`: Scale-invariant logarithmic error (lower is better).
    """
    # Ensure pred and gt are [B, H, W] by squeezing the channel dimension if it exists
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if gt.dim() == 4:
        gt = gt.squeeze(1)

    mask = gt > 0
    pred_m = pred[mask]
    gt_m = gt[mask]

    if pred_m.numel() == 0:
        return {
            "AbsRel": 0.0,
            "MAE": 0.0,
            "RMSE": 0.0,
            "Log10": 0.0,
            "delta1": 0.0,
            "delta2": 0.0,
            "delta3": 0.0,
            "SILog": 0.0,
        }

    # === Error metrics without scaling ===
    mae = torch.abs(pred_m - gt_m).mean().item()
    mse = ((pred_m - gt_m) ** 2).mean().item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    
    # === Log metrics ===
    log10_error = torch.abs(torch.log10(pred_m + 1e-8) - torch.log10(gt_m + 1e-8)).mean().item()
    
    # Scale-invariant logarithmic error (SILog)
    log_pred = torch.log(pred_m + 1e-8)
    log_gt = torch.log(gt_m + 1e-8)
    si_log = torch.sqrt(((log_pred - log_gt) ** 2).mean() - ((log_pred - log_gt).mean() ** 2)).item()

    # === Relative error (with median scaling) ===
    scale  = gt_m.median() / (pred_m.median() + 1e-8)
    pred_scaled = (pred_m * scale).clamp(min=1e-8)

    abs_rel = ((pred_scaled - gt_m).abs() / (gt_m + 1e-8)).mean().item()

    # === Threshold metrics (accuracy) ===
    ratio  = torch.max(pred_scaled / (gt_m + 1e-8), gt_m / (pred_scaled + 1e-8))
    delta1 = (ratio < 1.25     ).float().mean().item()
    delta2 = (ratio < 1.25 ** 2).float().mean().item()
    delta3 = (ratio < 1.25 ** 3).float().mean().item()

    return {
        "AbsRel": abs_rel,
        "MAE": mae,
        "RMSE": rmse,
        "Log10": log10_error,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
        "SILog": si_log,
    }


def aggregate_metrics(metrics_list: list[dict]) -> dict:
    """
    Average a list of metric dictionaries into one dictionary.
    """
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {
        k: sum(m[k] for m in metrics_list) / len(metrics_list)
        for k in keys
    }