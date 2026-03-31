import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

class ScaleAndShiftInvariantLoss(nn.Module):
    def _compute_scale_shift(self, pred, target):
        """Compute scale and shift for aligning pred to target in least-squares sense."""
        mean_pred = pred.mean()
        mean_target = target.mean()
        
        var_pred = ((pred - mean_pred) ** 2).mean()
        cov = ((pred - mean_pred) * (target - mean_target)).mean()
        
        if var_pred < 1e-8:
            return torch.tensor(1.0, device=pred.device), mean_target - mean_pred
        
        scale = cov / var_pred
        shift = mean_target - scale * mean_pred
        
        return scale, shift
    
    def forward(self, pred, target, mask=None):
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        if isinstance(target, (list, tuple)):
            target = target[0]

        if mask is None:
            mask = (target > 0)

        if isinstance(mask, torch.Tensor):
            mask = mask.to(device=target.device).bool()
        else:
            mask = torch.as_tensor(mask, dtype=torch.bool, device=target.device)

        pred_flat   = pred[mask].flatten()
        target_flat = target[mask].flatten()

        # Skip tiny or invalid masks to avoid degenerate scale/shift solves.
        if pred_flat.numel() < 50:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Log-domain stabilization and NaN/Inf cleanup.
        pred_flat   = torch.log1p(pred_flat.clamp(min=0))
        target_flat = torch.log1p(target_flat.clamp(min=0))
        pred_flat   = torch.nan_to_num(pred_flat, nan=0.0, posinf=0.0, neginf=0.0)
        target_flat = torch.nan_to_num(target_flat, nan=0.0, posinf=0.0, neginf=0.0)

        scale, shift = self._compute_scale_shift(pred_flat, target_flat)

        # --- NEW: clamp scale/shift để tránh degenerate solution ---
        scale = scale.clamp(-10, 10)
        shift = shift.clamp(-10, 10)

        pred_aligned = scale * pred_flat + shift
        loss = torch.mean((pred_aligned - target_flat) ** 2)
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0)
        return loss


class LocalROILoss(nn.Module):
    """SSI loss computed within each bounding box ROI.

    Args:
        boxes: list of per-image tensors, each shaped (N_i, 4) in xyxy pixel coords.
    """
    def __init__(self, roi_output_size=7):
        super().__init__()
        self.roi_output_size = roi_output_size
        self._ssi = ScaleAndShiftInvariantLoss()

    def forward(self, pred, target, boxes=None):
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        if isinstance(target, (list, tuple)):
            target = target[0]

        if boxes is None or not any(b is not None and len(b) > 0 for b in boxes):
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        _, _, H, W = pred.shape
        losses = []

        for i, b in enumerate(boxes):
            if b is None or len(b) == 0:
                continue
            for box in b:
                x1 = box[0].long().clamp(0, W - 1)
                y1 = box[1].long().clamp(0, H - 1)
                x2 = box[2].long().clamp(x1 + 1, W)
                y2 = box[3].long().clamp(y1 + 1, H)

                # Skip degenerate ROIs.
                if (x2 - x1) < 4 or (y2 - y1) < 4:
                    continue

                pred_roi   = pred[i:i+1, :, y1:y2, x1:x2]
                target_roi = target[i:i+1, :, y1:y2, x1:x2]

                valid_mask = target_roi > 0
                if valid_mask.sum() < 50:
                    continue

                # Skip near-constant depth slices where SSI is unstable.
                if target_roi[valid_mask].std() < 1e-4:
                    continue

                losses.append(self._ssi.forward(pred_roi, target_roi))

        if not losses:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        loss = torch.stack(losses).mean()
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0)
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, lambda_local=0.05, roi_size=7):
        super().__init__()
        self.ssi_loss = ScaleAndShiftInvariantLoss()
        self.local_loss = LocalROILoss(roi_output_size=roi_size)
        self.lambda_local = lambda_local

    def forward(self, pred, target, boxes=None):
        # SSI loss calculation
        loss_ssi = self.ssi_loss(pred, target)
        
        has_boxes = (
            boxes is not None
            and len(boxes) > 0
            and any(b is not None and len(b) > 0 for b in boxes)
        )

        if has_boxes:
            loss_local = self.local_loss(pred, target, boxes)
            total = loss_ssi + self.lambda_local * loss_local
        else:
            # Detect device from tensor or list of tensors
            dev = pred[0].device if isinstance(pred, (list, tuple)) else pred.device
            loss_local = torch.tensor(0.0, device=dev)
            total = loss_ssi

        total = torch.nan_to_num(total, nan=0.0, posinf=1e4, neginf=0.0)

        return total, {
            "loss_ssi":   loss_ssi.item(),
            "loss_local": loss_local.item(),
            "loss_total": total.item(),
            "has_boxes":  has_boxes,
        }