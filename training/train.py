"""
training/train.py

Entry point for PromptDA baseline evaluation and MLF training.

Dataset split convention:
  data/ARKitScenes/Training/   → train
  data/ARKitScenes/Validation/ → evaluate

Reference commands:

    # Baseline: zero-shot evaluation (no training)
  python training/train.py --use_mlf false --run_name baseline

    # Experiment: train only the MLF projector
  python training/train.py --use_mlf true --run_name mlf
"""

import argparse
import os
import random
import sys

import numpy as np

from torchvision.transforms import Compose

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

from dataset.dataset import MyARKitScenesDataset, collate_fn
from promptda.promptda import PromptDA
from promptda.utils.logger import Log
from training.optimizer import build_optimizer, build_scheduler
from training.trainer import Trainer
from data.ARKitScenes.depth_upsampling import transfroms


def str2bool(value: str) -> bool:
    return value.lower() in {"1", "true", "t", "yes", "y"}

def parse_args():
    p = argparse.ArgumentParser(description="PromptDA baseline vs MLF training")

    # Data
    p.add_argument("--data_root",   type=str, default="data/ARKitScenes/data/upsampling")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)

    # Model
    p.add_argument("--encoder", type=str, default="vitl",
                   choices=["vits", "vitb", "vitl"])
    p.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Local .ckpt path or Hugging Face repo id. None uses encoder default.",
    )
    p.add_argument(
        "--use_mlf",
        type=str2bool,
        default=True,
        help="false = baseline (zero-shot) | true = train MLF projector",
    )

    # Training (used only when --use_mlf=true)
    p.add_argument("--run_name",       type=str,   default="experiment")
    p.add_argument("--epochs",         type=int,   default=20)
    p.add_argument("--batch_size",     type=int,   default=4)
    p.add_argument("--lr_mlf",         type=float, default=1e-4,
                   help="Learning rate for the MLF projector")

    # Checkpoint
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--resume",         type=str, default=None,
                   help="Resume training from a trainer checkpoint")

    # Seed
    p.add_argument("--seed",           type=int, default=42,
                   help="Random seed for reproducibility")

    # Weights & Biases
    p.add_argument("--use_wandb", type=str2bool, default=True,
                   help="Enable logging to Weights & Biases")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Enable deterministic CuDNN behavior (may reduce performance slightly).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_loader(data_root, split, batch_size, num_workers, shuffle):
    ds = MyARKitScenesDataset(
        root=data_root,
        split=split,
    )
    if split == "train":
        transform = Compose([transfroms.RandomCrop(height=patch_size, width=patch_size, upsample_factor=upsample_factor),
                         transfroms.RandomFilpLR(),
                         transfroms.ValidDepthMask(gt_low_limit=0.01),
                         transfroms.AsContiguousArray()])
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    Log.info(f"{split}: {len(ds)} samples")
    return loader


def main():
    args   = parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Log.info(f"Device  : {device}")
    Log.info(f"Seed    : {args.seed}")
    Log.info(f"Run     : {args.run_name}")
    Log.info(f"Mode    : {'EXPERIMENT (train MLF)' if args.use_mlf else 'BASELINE (zero-shot)'}")
    Log.info(f"Encoder : {args.encoder}")

    # Validation loader is required for both baseline and training modes.
    val_loader = build_loader(
        args.data_root, "val", args.batch_size, args.num_workers, shuffle=False,
    )

    # Model
    model = PromptDA.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_path,
        encoder=args.encoder,
        use_mlf=args.use_mlf,
    ).to(device)

    ckpt_dir = f"{args.checkpoint_dir}/{args.run_name}_{args.encoder}_{args.seed}"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Optional W&B run
    wandb_run = None
    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Install it with: pip install wandb")
        wandb_run = wandb.init(
            project="ObjectPromptDA",
            entity="ObjectPromptDA",
            name=f"{args.run_name}_{args.encoder}_seed{args.seed}",
            dir=ckpt_dir,
            mode=args.wandb_mode,
            config=vars(args),
            tags=["mlf" if args.use_mlf else "baseline", args.encoder],
        )

    # Trainer
    trainer  = Trainer(
        model=model,
        optimizer=None,
        scheduler=None,
        device=device,
        ckpt_dir=ckpt_dir,
        wandb_run=wandb_run,
    )

    # Baseline: zero-shot evaluation on validation split.
    if not args.use_mlf:
        Log.info("Baseline: running zero-shot evaluation on validation split...")
        val_loss, metrics = trainer.eval_epoch(val_loader, epoch=0)
        Log.info(
            f"[BASELINE] AbsRel={metrics['AbsRel']:.4f} | "
            f"δ<1.25={metrics['delta1']:.4f} | "
            f"δ<1.25²={metrics['delta2']:.4f} | "
            f"δ<1.25³={metrics['delta3']:.4f}"
        )
        trainer.history["train_loss"].append(val_loss)
        trainer.history["val_loss"].append(val_loss)
        trainer.history["AbsRel"].append(metrics["AbsRel"])
        trainer.history["delta1"].append(metrics["delta1"])
        trainer.history["delta2"].append(metrics["delta2"])
        trainer.history["delta3"].append(metrics["delta3"])

        trainer.plot_history()
        trainer.save_checkpoint(epoch=0, metrics=metrics, tag="baseline_zeroshot")
        trainer.log_wandb_metrics(
            epoch=0,
            train_loss=None,
            val_loss=val_loss,
            metrics=metrics,
            stage="baseline",
        )
        if wandb_run is not None:
            wandb_run.finish()
        return

    # Experiment: train MLF on training split and evaluate on validation split.
    train_loader = build_loader(
        args.data_root, "train", args.batch_size, args.num_workers, shuffle=True,
    )

    optimizer = build_optimizer(model, lr_mlf=args.lr_mlf)
    scheduler = build_scheduler(
        optimizer,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        lr_mlf=args.lr_mlf,
    )
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler

    if args.resume:
        trainer.load_checkpoint(args.resume)
        Log.info(f"Resumed from: {args.resume}")

    trainer.fit(train_loader, val_loader, epochs=args.epochs)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()