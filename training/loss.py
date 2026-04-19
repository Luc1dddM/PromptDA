import sys
import torch
import torch.nn as nn

from data.ARKitScenes.depth_upsampling import dataset_keys as arkit_dataset_keys

sys.modules.setdefault("dataset_keys", arkit_dataset_keys)

from data.ARKitScenes.depth_upsampling.losses.l1_loss import l1_loss


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        inputs = {
            arkit_dataset_keys.HIGH_RES_DEPTH_IMG: target,
            arkit_dataset_keys.VALID_MASK_IMG: (target > 0).to(dtype=target.dtype),
        }
        outputs = {
            arkit_dataset_keys.PREDICTION_DEPTH_IMG: pred,
        }

        loss_mae = l1_loss(outputs, inputs)
        loss_mae = torch.nan_to_num(loss_mae, nan=0.0, posinf=1e4, neginf=0.0)

        return loss_mae, {
            "loss_mae": loss_mae.item(),
            "loss_total": loss_mae.item(),
        }
