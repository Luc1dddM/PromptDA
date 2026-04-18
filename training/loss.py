import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae_loss = nn.L1Loss()

    def forward(self, pred, target):
        loss_mae = self.mae_loss(pred, target)
        loss_mae = torch.nan_to_num(loss_mae, nan=0.0, posinf=1e4, neginf=0.0)

        return loss_mae, {
            "loss_mae": loss_mae.item(),
            "loss_total": loss_mae.item(),
        }
