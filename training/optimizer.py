from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn


def build_optimizer(model: nn.Module, lr_dpt: float) -> AdamW:
    if hasattr(model, "trainable_parameters"):
        params = list(model.trainable_parameters())
    else:
        params = [p for p in model.parameters() if p.requires_grad]

    if not params:
        raise ValueError("No trainable parameters found for optimizer.")

    return AdamW(
        params,
        lr=lr_dpt,
        weight_decay=1e-4,
    )


def build_scheduler(
    optimizer: AdamW,
    steps_per_epoch: int,
    epochs: int,
    lr_dpt: float,
) -> OneCycleLR:
    return OneCycleLR(
        optimizer,
        max_lr=lr_dpt,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100,
        anneal_strategy='cos',
    )