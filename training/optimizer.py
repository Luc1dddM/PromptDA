from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn


def build_optimizer(model: nn.Module, lr_mlf: float) -> AdamW:
    """
    Chỉ optimize MLF projector.
    DPT head và DINOv2 đã được freeze trong PromptDA._apply_freeze_strategy()
    → requires_grad=False → không cần đưa vào optimizer.
    
    Gradient vẫn chảy qua DPT head vì các activation tensor vẫn tracked,
    chỉ có weight của DPT head là không update.
    """
    mlf_params = [p for p in model.mlf.parameters() if p.requires_grad]

    if not mlf_params:
        raise ValueError(
            "No trainable parameters in MLF. "
            "Check that use_mlf=True and MaskedLocalFusion is initialized."
        )

    return AdamW(
        mlf_params,
        lr=lr_mlf,
        weight_decay=1e-4,
    )


def build_scheduler(
    optimizer: AdamW,
    steps_per_epoch: int,
    epochs: int,
    lr_mlf: float,
) -> OneCycleLR:
    """
    OneCycleLR chỉ cho MLF — warmup 10% → peak → cooldown.
    """
    return OneCycleLR(
        optimizer,
        max_lr=lr_mlf,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.1,          # 10% warmup
        div_factor=10,          # initial_lr = lr_mlf / 10
        final_div_factor=100,   # min_lr = lr_mlf / 100
        anneal_strategy='cos',
    )