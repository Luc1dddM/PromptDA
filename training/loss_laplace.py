"""
Robust Laplace NLL Loss for Aleatoric Uncertainty in Depth Completion.

Formula (per valid pixel):
    L1_term   = sqrt(2) * |mu - target| / sigma
    reg_term  = log(sigma)
    loss      = mean(L1_term + reg_term)

where sigma = exp(s) + eps, and s = log(sigma) is the raw network output.
Reference: Kendall & Gal, NeurIPS 2017.
"""

import torch
import torch.nn as nn


class RobustLaplaceNLLLoss(nn.Module):
    """Laplace negative log-likelihood loss with numerical stability guards.

    Expects:
        pred:   dict with keys "mu" [B, 1, H, W] and "s" [B, 1, H, W]
                OR a tensor [B, 2, H, W] where channel 0 = mu, channel 1 = s.
        target: [B, 1, H, W] ground-truth depth (0 = invalid/missing).

    Returns:
        loss: scalar tensor (0.0 if no valid pixels in batch).
        info: dict with "loss_laplace_nll", "loss_total", "mean_sigma" for logging.
    """

    def __init__(self, s_min: float = -10.0, s_max: float = 10.0, eps: float = 1e-6):
        super().__init__()
        self.s_min = s_min
        self.s_max = s_max
        self.eps = eps
        self.sqrt2 = 2.0 ** 0.5

    def forward(self, pred, target):
        # --- unpack prediction ---
        if isinstance(pred, dict):
            mu = pred["mu"]       # [B, 1, H, W]
            s = pred["s"]         # [B, 1, H, W]
        else:
            # tensor [B, 2, H, W]
            mu = pred[:, 0:1, :, :]   # channel 0 = depth
            s  = pred[:, 1:2, :, :]   # channel 1 = log(sigma)

        # --- numerical safety: clamp log-sigma, then exponentiate ---
        s = torch.clamp(s, min=self.s_min, max=self.s_max)
        sigma = torch.exp(s) + self.eps   # sigma > 0 guaranteed

        # --- valid mask: only compute loss where GT depth > 0 ---
        valid_mask = target > 0

        # If no valid pixels in the entire batch, return zero loss
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device if isinstance(pred, torch.Tensor) else mu.device), {
                "loss_laplace_nll": 0.0,
                "loss_total": 0.0,
                "mean_sigma": 0.0,
                "valid_pixels": 0,
            }

        # --- Laplace NLL ---
        # L1 term: sqrt2 * |mu - target| / sigma
        l1_term = self.sqrt2 * torch.abs(mu - target) / sigma

        # Regularization term: log(sigma)
        reg_term = torch.log(sigma)

        # Per-pixel loss, masked
        per_pixel_loss = l1_term + reg_term                        # [B, 1, H, W]
        per_pixel_loss = per_pixel_loss[valid_mask]                # [N_valid]

        loss = per_pixel_loss.mean()

        # Logging info
        with torch.no_grad():
            mean_sigma = sigma[valid_mask].mean().item()
            loss_val = loss.item()
            n_valid = valid_mask.sum().item()

        return loss, {
            "loss_laplace_nll": loss_val,
            "loss_total": loss_val,
            "mean_sigma": mean_sigma,
            "valid_pixels": n_valid,
        }