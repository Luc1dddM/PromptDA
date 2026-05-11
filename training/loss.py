import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.ARKitScenes.depth_upsampling import dataset_keys as arkit_dataset_keys

sys.modules.setdefault("dataset_keys", arkit_dataset_keys)

from data.ARKitScenes.depth_upsampling.losses.l1_loss import l1_loss
from data.ARKitScenes.depth_upsampling.losses.gradient_loss import gradient_loss


def edge_aware_smooth_loss(pred_depth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """Edge-aware smoothness loss.

    L_smooth = |∂x D| · exp(-|∂x I|) + |∂y D| · exp(-|∂y I|)

    Args:
        pred_depth: (B, 1, H, W) predicted depth map.
        image:      (B, C, H, W) RGB image used as edge guide.

    Returns:
        Scalar mean loss.
    """
    # Depth gradients
    d_dx = torch.abs(pred_depth[:, :, :, :-1] - pred_depth[:, :, :, 1:])   # (B,1,H,W-1)
    d_dy = torch.abs(pred_depth[:, :, :-1, :] - pred_depth[:, :, 1:, :])   # (B,1,H-1,W)

    # Image gradients (mean over channels → single-channel edge map)
    i_dx = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]).mean(dim=1, keepdim=True)  # (B,1,H,W-1)
    i_dy = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]).mean(dim=1, keepdim=True)  # (B,1,H-1,W)

    loss = (d_dx * torch.exp(-i_dx)).mean() + (d_dy * torch.exp(-i_dy)).mean()
    return loss


class CombinedLoss(nn.Module):
    """Combined depth loss.

    Args:
        use_smooth: If True, adds edge-aware smoothness term to the base
                    (L1 + 2·gradient) loss.
        smooth_weight: Weight λ for the smoothness term (default 0.1).
    """

    def __init__(self, use_smooth: bool = False, smooth_weight: float = 0.1):
        super().__init__()
        self.use_smooth = use_smooth
        self.smooth_weight = smooth_weight

    def forward(self, pred, target, image: torch.Tensor | None = None):
        """
        Args:
            pred:   Predicted depth tensor (B, 1, H, W) or dict with key 'mu'.
            target: Ground-truth depth tensor (B, 1, H, W).
            image:  RGB image tensor (B, 3, H, W).  Required when use_smooth=True.
        """
        pred_depth = pred["mu"] if isinstance(pred, dict) else pred

        inputs = {
            arkit_dataset_keys.HIGH_RES_DEPTH_IMG: target,
            arkit_dataset_keys.VALID_MASK_IMG: (target > 0),
        }
        outputs = {
            arkit_dataset_keys.PREDICTION_DEPTH_IMG: pred_depth,
        }

        loss_l1 = l1_loss(outputs, inputs)
        loss_grad = gradient_loss(outputs, inputs)
        loss_base = loss_l1 + 2.0 * loss_grad

        loss_info = {
            "loss_l1": loss_l1.item(),
            "loss_grad": loss_grad.item(),
        }

        if self.use_smooth:
            if image is None:
                raise ValueError("CombinedLoss: `image` must be provided when use_smooth=True.")
            # Resize image to match pred_depth spatial size if needed
            if image.shape[-2:] != pred_depth.shape[-2:]:
                image = F.interpolate(image, size=pred_depth.shape[-2:], mode="bilinear", align_corners=False)
            loss_smooth = edge_aware_smooth_loss(pred_depth, image)
            loss_total = loss_base + self.smooth_weight * loss_smooth
            loss_info["loss_smooth"] = loss_smooth.item()
        else:
            loss_total = loss_base
            loss_info["loss_smooth"] = 0.0

        loss_total = torch.nan_to_num(loss_total, nan=0.0, posinf=1e4, neginf=0.0)
        loss_info["loss_total"] = loss_total.item()

        return loss_total, loss_info
