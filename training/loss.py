import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.ARKitScenes.depth_upsampling import dataset_keys as arkit_dataset_keys

sys.modules.setdefault("dataset_keys", arkit_dataset_keys)

from data.ARKitScenes.depth_upsampling.losses.l1_loss import l1_loss
from data.ARKitScenes.depth_upsampling.losses.gradient_loss import gradient_loss


def edge_aware_smooth_loss(pred_depth: torch.Tensor, image: torch.Tensor, alpha: float = 10.0) -> torch.Tensor:
    """Edge-aware smoothness loss.

    L_smooth = |∂x D| · exp(-alpha · |∂x I|) + |∂y D| · exp(-alpha · |∂y I|)

    Args:
        pred_depth: (B, 1, H, W) predicted depth map.
        image:      (B, C, H, W) RGB image used as edge guide (nên ở dải giá trị [0, 1]).
        alpha:      Hệ số khuếch đại gradient ảnh (giúp nhạy bén hơn với cạnh nhỏ).

    Returns:
        Scalar mean loss.
    """
    # Chuẩn hóa pred_depth theo giá trị trung bình để loss scale-invariant
    # Giúp hàm loss không bị áp đảo bởi các cảnh có độ sâu tuyệt đối quá lớn
    mean_depth = pred_depth.mean(dim=[2, 3], keepdim=True)
    norm_depth = pred_depth / (mean_depth + 1e-8)

    # Depth gradients (tính trên depth đã chuẩn hóa)
    d_dx = torch.abs(norm_depth[:, :, :, :-1] - norm_depth[:, :, :, 1:])   # (B,1,H,W-1)
    d_dy = torch.abs(norm_depth[:, :, :-1, :] - norm_depth[:, :, 1:, :])   # (B,1,H-1,W)

    # Image gradients (mean over channels → single-channel edge map)
    i_dx = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]).mean(dim=1, keepdim=True)  # (B,1,H,W-1)
    i_dy = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]).mean(dim=1, keepdim=True)  # (B,1,H-1,W)

    # Tính loss với trọng số là e^(-alpha * grad_img)
    loss = (d_dx * torch.exp(-alpha * i_dx)).mean() + (d_dy * torch.exp(-alpha * i_dy)).mean()
    return loss


class CombinedLoss(nn.Module):
    """Combined depth loss.

    Args:
        use_smooth:    If True, adds edge-aware smoothness term to the base
                       (L1 + 2·gradient) loss.
        smooth_weight: Weight λ for the smoothness term (default 0.1).
        smooth_alpha:  Alpha parameter for edge sensitivity in smoothness loss (default 10.0).
    """

    def __init__(self, use_smooth: bool = False, smooth_weight: float = 0.1, smooth_alpha: float = 10.0):
        super().__init__()
        self.use_smooth = use_smooth
        self.smooth_weight = smooth_weight
        self.smooth_alpha = smooth_alpha

    def forward(self, pred, target, image: torch.Tensor | None = None):
        """
        Args:
            pred:   Predicted depth tensor (B, 1, H, W) or dict with key 'mu'.
            target: Ground-truth depth tensor (B, 1, H, W).
            image:  RGB image tensor (B, 3, H, W). Required when use_smooth=True. 
                    Ensure image tensor values are in [0, 1].
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
            
            # Đảm bảo ảnh nằm trong dải [0, 1] nếu nó đang ở dạng [0, 255]
            if image.max() > 1.0:
                image = image / 255.0

            # Resize image to match pred_depth spatial size if needed
            if image.shape[-2:] != pred_depth.shape[-2:]:
                image = F.interpolate(image, size=pred_depth.shape[-2:], mode="bilinear", align_corners=False)
            
            # Tính smooth loss với tham số alpha
            loss_smooth = edge_aware_smooth_loss(pred_depth, image, alpha=self.smooth_alpha)
            
            loss_total = loss_base + self.smooth_weight * loss_smooth
            loss_info["loss_smooth"] = loss_smooth.item()
        else:
            loss_total = loss_base
            loss_info["loss_smooth"] = 0.0

        loss_total = torch.nan_to_num(loss_total, nan=0.0, posinf=1e4, neginf=0.0)
        loss_info["loss_total"] = loss_total.item()

        return loss_total, loss_info