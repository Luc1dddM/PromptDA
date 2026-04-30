"""Optimizer and scheduler setup for PromptDA training modes."""

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn


def build_optimizer(
    model: nn.Module,
    use_mlf: bool,
    from_scratch: bool,
    lr_backbone: float,
    lr_head: float,
    lr_mlf: float,
) -> AdamW:
    """Build AdamW optimizer for either MLF-only or full scratch training."""
    if from_scratch:
        backbone_params = [p for p in model.pretrained.parameters() if p.requires_grad]
        depth_head_params = [p for p in model.depth_head.parameters() if p.requires_grad]

        if not backbone_params and not depth_head_params:
            raise ValueError("No trainable parameters found for from_scratch mode.")

        return AdamW(
            [
                {"params": backbone_params, "lr": lr_backbone, "weight_decay": 1e-4},
                {"params": depth_head_params, "lr": lr_head, "weight_decay": 1e-4},
            ]
        )

    if use_mlf:
        mlf_params = [p for p in model.mlf.parameters() if p.requires_grad]
        depth_head_params = [p for p in model.depth_head.parameters() if p.requires_grad]

        if not mlf_params:
            raise ValueError("No trainable parameters found in MLF. Check `use_mlf=True`.")

        return AdamW(
            [
                {"params": depth_head_params, "lr": 0.0, "weight_decay": 0.0},
                {"params": mlf_params, "lr": lr_mlf, "weight_decay": 1e-4},
            ]
        )

    raise ValueError("Invalid optimizer mode: baseline mode does not require optimizer.")


def build_scheduler(
    optimizer: AdamW,
    use_mlf: bool,
    from_scratch: bool,
    steps_per_epoch: int,
    epochs: int,
    lr_backbone: float,
    lr_head: float,
    lr_mlf: float,
) -> OneCycleLR:
    """Build OneCycleLR scheduler matching optimizer parameter groups."""
    if from_scratch:
        max_lr = [lr_backbone, lr_head]
    elif use_mlf:
        max_lr = [0.0, lr_mlf]
    else:
        raise ValueError("Invalid scheduler mode: baseline mode does not require scheduler.")

    return OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100,
    )
