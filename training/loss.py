"""
promptda/model/losses.py

Loss functions for depth estimation training.
"""

import torch
import torch.nn as nn


class ScaleAndShiftInvariantLoss(nn.Module):
    """
    Scale-and-Shift Invariant Loss used by PromptDA.

    The loss solves for the optimal scale and shift (least squares) to align
    prediction to ground truth before computing L1 loss.
    Only valid depth pixels (`gt > 0`) are used.
    """

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, 1, H, W] predicted depth (arbitrary scale).
            gt: [B, 1, H, W] ground truth depth in meters (`0` means invalid).
        Returns:
            Scalar loss tensor.
        """
        mask = (gt > 0).float()
        n    = mask.sum(dim=[1, 2, 3]).clamp(min=1)

        pred_m = pred * mask
        gt_m   = gt   * mask

        sum_pred  = pred_m.sum(dim=[1, 2, 3])
        sum_gt    = gt_m.sum(dim=[1, 2, 3])
        sum_pred2 = (pred_m ** 2).sum(dim=[1, 2, 3])
        sum_pg    = (pred_m * gt_m).sum(dim=[1, 2, 3])

        denom = sum_pred2 - sum_pred ** 2 / n + 1e-8
        scale = (sum_pg - sum_pred * sum_gt / n) / denom
        shift = (sum_gt - scale * sum_pred) / n

        scale = scale.view(-1, 1, 1, 1)
        shift = shift.view(-1, 1, 1, 1)

        pred_aligned = scale * pred + shift
        loss = (mask * (pred_aligned - gt).abs()).sum(dim=[1, 2, 3]) / n
        return loss.mean()