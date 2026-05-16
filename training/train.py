import argparse
import os
import random
import shlex
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose

try:
    import wandb
except ImportError:
    wandb = None

from data.ARKitScenes.depth_upsampling import dataset_keys as arkit_dataset_keys
from data.ARKitScenes.depth_upsampling import image_utils as arkit_image_utils

sys.modules.setdefault("dataset_keys", arkit_dataset_keys)
sys.modules.setdefault("image_utils", arkit_image_utils)

from data.ARKitScenes.depth_upsampling import transfroms
from data.ARKitScenes.depth_upsampling.logs.eval import eval_log
from data.ARKitScenes.depth_upsampling.logs.train import train_log
from data.ARKitScenes.depth_upsampling.sampler import MultiEpochSampler
from dataset.dataset import MyARKitScenesDataset, collate_fn
from dataset import dataset_keys
from promptda.promptda_baseline import PromptDA as ViTPromptDA
from promptda.promptda_pyramic import PromptDA as PyramidPromptDA
from promptda.promptda_uncertainty import PromptDAUncertainty
from promptda.utils.logger import Log
from training.optimizer import build_optimizer
from training.trainer import Trainer

TENSORBOARD_DIR = "tensorboard"
PYRAMID_ENCODERS = tuple(PyramidPromptDA.BACKBONES)
VIT_ENCODERS = ("vits", "vitb", "vitl")
ALL_ENCODERS = VIT_ENCODERS + PYRAMID_ENCODERS


def str2bool(value: str) -> bool:
    return value.lower() in {"1", "true", "t", "yes", "y"}


def parse_args():
    p = argparse.ArgumentParser(description="Train baseline PromptDA with DPT variants")

    p.add_argument("--data_root", type=str, default="data/ARKitScenes/data/upsampling")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument(
        "--encoder",
        type=str,
        default="vits",
        choices=ALL_ENCODERS,
    )
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

    p.add_argument("--uncertainty", type=str2bool, default=False,
                   help="Use 2-channel uncertainty head + Laplace NLL loss")

    p.add_argument("--use_wandb", type=str2bool, default=True)
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--promptda_ckpt", type=str, default=None,
               help="Path or HF repo for pretrained PromptDA checkpoint (default: auto from --encoder)")

    p.add_argument("--num_iter", type=int, default=20000)
    p.add_argument("--log_freq", type=int, default=200)
    p.add_argument("--eval_freq", type=int, default=1000)
    p.add_argument("--save_freq", type=int, default=5000)
    p.add_argument("--tbp", type=int, default=None)
    p.add_argument("--log_dir", type=str, default="log")

    p.add_argument("--use_smooth", type=str2bool, default=False,
                   help="Add edge-aware smoothness loss: |∂x D|·exp(-|∂x I|) + |∂y D|·exp(-|∂y I|)")
    p.add_argument("--smooth_weight", type=float, default=0.1,
                   help="Weight λ for the smoothness term (default: 0.1)")

    return p.parse_args()


def get_backbone_family(encoder: str) -> str:
    return "pyramid" if encoder in PYRAMID_ENCODERS else "vit"


def build_model(args):
    backbone_family = get_backbone_family(args.encoder)

    if args.uncertainty and backbone_family == "pyramid":
        raise ValueError("Uncertainty training is implemented only for DINOv2 ViT encoders.")

    if args.uncertainty:
        return PromptDAUncertainty.from_pretrained(
            pretrained_model_name_or_path=args.promptda_ckpt,
            encoder=args.encoder,
            dpt_variant=args.dpt_variant,
        )

    if backbone_family == "pyramid":
        return PyramidPromptDA(
            encoder=args.encoder,
            ckpt_path=args.promptda_ckpt,
            dpt_variant=args.dpt_variant,
            pretrained_backbone=True,
        )

    return ViTPromptDA(
        encoder=args.encoder,
        dpt_variant=args.dpt_variant,
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(split: str | None = None):
    if split == "train":
        return Compose([
            transfroms.RandomFilpLR(),
            transfroms.ValidDepthMask(gt_low_limit=0.01),
            transfroms.AsContiguousArray(),
        ])
    return Compose([
        transfroms.ModCrop(modulo=14),
        transfroms.ValidDepthMask(gt_low_limit=0.01),
    ])


def build_train_loader(args, start_itr: int):
    ds = MyARKitScenesDataset(
        root=args.data_root,
        split="train",
        max_samples=args.max_samples,
        seed=args.seed,
        transform=build_transforms('train'),
    )
    sampler = MultiEpochSampler(ds, args.num_iter, start_itr, args.batch_size)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    Log.info(f"train: {len(ds)} samples")
    return loader


def build_val_loader(args):
    ds = MyARKitScenesDataset(
        root=args.data_root,
        split="val",
        max_samples=None,
        seed=args.seed,
        transform=build_transforms("val"),
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    Log.info(f"val: {len(ds)} samples")
    return loader


def strip_boxes(batch):
    if dataset_keys.BOUNDING_BOX in batch or dataset_keys.BOUNDING_BOX_IMAGE in batch:
        return {
            k: v for k, v in batch.items()
            if k not in (dataset_keys.BOUNDING_BOX, dataset_keys.BOUNDING_BOX_IMAGE)
        }
    return batch


def iter_stripped(loader):
    for batch in loader:
        yield strip_boxes(batch)


class LoggingModel:
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def __call__(self, input_batch):
        image = input_batch[dataset_keys.COLOR_IMG].to(self.device)
        prompt = input_batch[dataset_keys.LOW_RES_DEPTH_IMG].to(self.device)
        pred = self.model(image, prompt)
        pred_depth = pred["mu"] if isinstance(pred, dict) else pred
        return {dataset_keys.PREDICTION_DEPTH_IMG: pred_depth}


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Log.info(f"Device      : {device}")
    Log.info(f"Seed        : {args.seed}")
    Log.info(f"Run         : {args.run_name}")
    Log.info(f"Encoder     : {args.encoder}")
    backbone_family = get_backbone_family(args.encoder)
    Log.info(f"Backbone    : {backbone_family}")
    Log.info(f"DPT variant : {args.dpt_variant}")
    Log.info(f"Uncertainty : {args.uncertainty}")
    Log.info(f"Use smooth  : {args.use_smooth} (λ={args.smooth_weight})")
    Log.info(f"Eval only   : {args.eval_only}")

    val_loader = build_val_loader(args)
    model = build_model(args)

    ckpt_dir = f"{args.checkpoint_dir}/{args.run_name}_{backbone_family}_{args.encoder}_{args.dpt_variant}_{args.seed}"
    if args.uncertainty:
        ckpt_dir += "_uncertainty"
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
            tags=[backbone_family, args.encoder, args.dpt_variant, "dpt_head_only"] + (["uncertainty"] if args.uncertainty else []),
        )

    trainer = Trainer(
        model=model,
        optimizer=None,
        scheduler=None,
        device=device,
        ckpt_dir=ckpt_dir,
        wandb_run=wandb_run,
        uncertainty=args.uncertainty,
        use_smooth=args.use_smooth,
        smooth_weight=args.smooth_weight,
    )

    if args.eval_only:
        if args.resume:
            trainer.load_checkpoint(args.resume)
            Log.info(f"Resumed from: {args.resume}")

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

    optimizer = build_optimizer(model, lr_dpt=args.lr_dpt)
    trainer.optimizer = optimizer

    if args.resume:
        trainer.load_checkpoint(args.resume)
        Log.info(f"Resumed from: {args.resume}")

    start_itr = trainer.global_step
    train_loader = build_train_loader(args, start_itr)

    if args.tbp is not None:
        Log.info("starting tensorboard")
        tensorboard_path = os.path.join(args.log_dir, TENSORBOARD_DIR)
        command = f"tensorboard --logdir {tensorboard_path} --port {args.tbp}"
        tensorboard_process = subprocess.Popen(shlex.split(command), env=os.environ.copy())
        train_tensorboard_writer = SummaryWriter(os.path.join(tensorboard_path, "train"), flush_secs=30)
        val_tensorboard_writer = SummaryWriter(os.path.join(tensorboard_path, "val"), flush_secs=30)
    else:
        Log.info("no tensorboard")
        tensorboard_process = None
        train_tensorboard_writer = None
        val_tensorboard_writer = None

    start_time = time.time()
    duration = 0
    step = start_itr + 1
    recent_train_losses = []  # accumulate train loss between eval steps

    Log.info("start training")
    for input_batch in train_loader:
        if step > args.num_iter:
            break

        before_op_time = time.time()
        input_batch = strip_boxes(input_batch)

        image = input_batch[dataset_keys.COLOR_IMG].to(device)
        depth_gt = input_batch[dataset_keys.HIGH_RES_DEPTH_IMG].to(device)
        prompt = input_batch[dataset_keys.LOW_RES_DEPTH_IMG].to(device)
        pred = model(image, prompt)

        pred_depth = pred["mu"] if isinstance(pred, dict) else pred
        assert pred_depth.shape[-2:] == depth_gt.shape[-2:], (
            f"[main loop] pred {pred_depth.shape[-2:]} ≠ GT {depth_gt.shape[-2:]}. "
            "Check dataset image/depth sizes."
        )

        loss, _ = trainer.loss_fn(pred, depth_gt, image if args.use_smooth else None)

        if torch.isnan(loss).any():
            raise RuntimeError("NaN in loss occurred. Aborting training.")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        duration += time.time() - before_op_time
        recent_train_losses.append(loss.item())

        # ── Log train loss (wandb + tensorboard) ──────────────────────────
        if step % args.log_freq == 0:
            avg_train_loss = sum(recent_train_losses) / len(recent_train_losses)
            recent_train_losses = []

            current_lr = optimizer.param_groups[0].get("lr", -1)

            # Tensorboard
            if train_tensorboard_writer is not None:
                train_tensorboard_writer.add_scalar("loss", avg_train_loss, step)
                train_tensorboard_writer.add_scalar("lr", current_lr, step)

            # wandb
            if wandb_run is not None:
                wandb_run.log({
                    "train/loss": avg_train_loss,
                    "train/lr": current_lr,
                    "step": step,
                })

            examples_per_sec = args.batch_size / duration * args.log_freq
            time_sofar = (time.time() - start_time) / 3600
            training_time_left = (args.num_iter / step - 1.0) * time_sofar
            print_string = "step={} | loss={:.4f} | examples/s: {:4.2f} | time elapsed: {:.2f}h | time left: {:.2f}h"
            print(print_string.format(step, avg_train_loss, examples_per_sec, time_sofar, training_time_left))
            duration = 0

        # ── Eval + log val metrics (wandb + tensorboard) ──────────────────
        if step % args.eval_freq == 0:
            val_loss, metrics = trainer.eval_epoch(val_loader, epoch=step)

            Log.info(
                f"[EVAL step={step}] val_loss={val_loss:.4f} | "
                f"L1={metrics['L1']:.4f} | RMSE={metrics['RMSE']:.4f}"
            )

            # Tensorboard
            if val_tensorboard_writer is not None:
                val_tensorboard_writer.add_scalar("loss", val_loss, step)
                val_tensorboard_writer.add_scalar("L1", metrics["L1"], step)
                val_tensorboard_writer.add_scalar("RMSE", metrics["RMSE"], step)

            # wandb — dùng trainer.log_wandb_metrics để nhất quán
            avg_train_loss_for_log = (
                sum(recent_train_losses) / len(recent_train_losses)
                if recent_train_losses else None
            )
            trainer.log_wandb_metrics(
                epoch=step,
                train_loss=avg_train_loss_for_log,
                val_loss=val_loss,
                metrics=metrics,
                stage="train",
            )

            # Save best checkpoint theo val L1
            if metrics["L1"] < trainer.best_l1:
                trainer.best_l1 = metrics["L1"]
                trainer.save_checkpoint(epoch=step, metrics=metrics, tag="best")
                Log.info(f"  ✓ best.pth saved → L1={trainer.best_l1:.4f}")

            model.train()

        # ── Save checkpoint ───────────────────────────────────────────────
        if step % args.save_freq == 0:
            checkpoint = {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_file = os.path.join(ckpt_dir, f"checkpoint_step-{step}")
            torch.save(checkpoint, save_file)
            Log.info(f"Checkpoint saved: {save_file}")

        trainer.global_step = step
        step += 1

    Log.info("finished training")
    if tensorboard_process is not None:
        tensorboard_process.terminate()

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
