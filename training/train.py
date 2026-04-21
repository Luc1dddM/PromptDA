import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

from dataset.dataset import MyARKitScenesDataset, collate_fn
from promptda.promptda_baseline import PromptDA
from promptda.utils.logger import Log
from training.optimizer import build_optimizer, build_scheduler
from training.trainer import Trainer


def str2bool(value: str) -> bool:
    return value.lower() in {"1", "true", "t", "yes", "y"}


def parse_args():
    p = argparse.ArgumentParser(description="Train baseline PromptDA with DPT variants")

    p.add_argument("--data_root", type=str, default="data/ARKitScenes/data/upsampling")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitb", "vitl"])
    p.add_argument(
        "--dpt_variant",
        type=str,
        default="legacy",
        choices=["legacy", "skip_concat_1x1", "hybrid_ca_shallow_concat"],
    )

    p.add_argument("--run_name", type=str, default="baseline_compare")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr_dpt", type=float, default=1e-4)
    p.add_argument("--eval_only", type=str2bool, default=False)

    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--resume", type=str, default=None)

    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--use_wandb", type=str2bool, default=True)
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_loader(data_root, split, batch_size, num_workers, shuffle):
    ds = MyARKitScenesDataset(root=data_root, split=split)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    Log.info(f"{split}: {len(ds)} samples")
    return loader


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Log.info(f"Device      : {device}")
    Log.info(f"Seed        : {args.seed}")
    Log.info(f"Run         : {args.run_name}")
    Log.info(f"Encoder     : {args.encoder}")
    Log.info(f"DPT variant : {args.dpt_variant}")
    Log.info(f"Eval only   : {args.eval_only}")

    val_loader = build_loader(args.data_root, "val", args.batch_size, args.num_workers, shuffle=False)

    model = PromptDA(
        encoder=args.encoder,
        dpt_variant=args.dpt_variant,
    ).to(device)

    ckpt_dir = f"{args.checkpoint_dir}/{args.run_name}_{args.encoder}_{args.dpt_variant}_{args.seed}"
    os.makedirs(ckpt_dir, exist_ok=True)

    wandb_run = None
    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Install it with: pip install wandb")
        wandb_run = wandb.init(
            project="ObjectPromptDA",
            entity="ObjectPromptDA",
            name=f"{args.run_name}_{args.encoder}_{args.dpt_variant}_seed{args.seed}",
            dir=ckpt_dir,
            mode=args.wandb_mode,
            config=vars(args),
            tags=["baseline", args.encoder, args.dpt_variant, "dpt_head_only"],
        )

    trainer = Trainer(
        model=model,
        optimizer=None,
        scheduler=None,
        device=device,
        ckpt_dir=ckpt_dir,
        wandb_run=wandb_run,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)
        Log.info(f"Resumed from: {args.resume}")

    if args.eval_only:
        val_loss, metrics = trainer.eval_epoch(val_loader, epoch=0)
        Log.info(
            f"[EVAL] L1={metrics['L1']:.4f} | "
            f"RMSE={metrics['RMSE']:.4f}"
        )
        trainer.log_wandb_metrics(
            epoch=0,
            train_loss=None,
            val_loss=val_loss,
            metrics=metrics,
            stage="eval",
        )
        trainer.save_checkpoint(epoch=0, metrics=metrics, tag="eval")
        if wandb_run is not None:
            wandb_run.finish()
        return

    train_loader = build_loader(args.data_root, "train", args.batch_size, args.num_workers, shuffle=True)

    optimizer = build_optimizer(model, lr_dpt=args.lr_dpt)
    scheduler = build_scheduler(
        optimizer,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        lr_dpt=args.lr_dpt,
    )
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler

    trainer.fit(train_loader, val_loader, epochs=args.epochs)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
