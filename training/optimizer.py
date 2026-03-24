"""
training/optimizer.py

Optimizer setup for MLF training.

Why use `lr=0` for the DPT head instead of hard-freezing it?
    - The gradient must flow: loss → depth_head → MLF projector.
    - If the DPT head is frozen with `requires_grad=False`, the graph can break
        depending on upstream tensors, and `loss.backward()` may fail.
    - Keeping DPT parameters trainable but assigning `lr=0` preserves gradient
        propagation while preventing weight updates.
"""

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn


def build_optimizer(model: nn.Module, lr_mlf: float) -> AdamW:
    """
        Two optimizer parameter groups:
            - DPT head      : `lr=0` (no updates, but graph remains connected)
            - MLF projector : `lr=lr_mlf` (actively trained)

    Args:
                model: PromptDA instance.
                lr_mlf: Learning rate for MLF projector (recommended: 1e-4).
    """
    mlf_params        = [p for p in model.mlf.parameters() if p.requires_grad]
    depth_head_params = [p for p in model.depth_head.parameters() if p.requires_grad]

    if not mlf_params:
                raise ValueError("No trainable parameters found in MLF. Check `use_mlf=True`.")

    return AdamW(
        [
            # DPT head participates in autograd but is not updated.
            {"params": depth_head_params, "lr": 0.0,    "weight_decay": 0.0},
            # MLF projector is actively optimized.
            {"params": mlf_params,        "lr": lr_mlf, "weight_decay": 1e-4},
        ],
    )


def build_scheduler(
    optimizer: AdamW,
    steps_per_epoch: int,
    epochs: int,
    lr_mlf: float,
) -> OneCycleLR:
    """
    OneCycleLR schedule where only MLF uses a non-zero max LR.
    DPT head LR remains zero for the whole run.
    """
    return OneCycleLR(
        optimizer,
        max_lr=[0.0, lr_mlf],        # [DPT Head=0, MLF=lr_mlf]
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100,
    )