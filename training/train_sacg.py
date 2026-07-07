from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from data.ARKitScenes.depth_upsampling import dataset_keys as arkit_dataset_keys
from data.ARKitScenes.depth_upsampling import image_utils as arkit_image_utils

sys.modules.setdefault("dataset_keys", arkit_dataset_keys)
sys.modules.setdefault("image_utils", arkit_image_utils)

from data.ARKitScenes.depth_upsampling import transfroms
from data.ARKitScenes.depth_upsampling.sampler import MultiEpochSampler
from dataset.dataset import MyARKitScenesDataset, collate_fn
from dataset import dataset_keys
from promptda.sacg import PromptDASACG, SACGLoss
from promptda.utils.logger import Log
from training.metrics import aggregate_metrics, compute_depth_metrics
from training.visualization import plot_sacg_comparison

TENSORBOARD_DIR = "tensorboard"


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "t", "yes", "y"}


def parse_args():
    p = argparse.ArgumentParser(description="Train SACG refinement on top of frozen official PromptDA")
    p.add_argument("--data_root", type=str, default="data/ARKitScenes/data/upsampling")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_val_samples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitb", "vitl"])
    p.add_argument(
        "--dpt_variant",
        type=str,
        default="legacy",
        choices=["legacy", "skip_concat_1x1", "hybrid_ca_shallow_concat"],
    )
    p.add_argument(
        "--promptda_ckpt",
        type=str,
        default=None,
        help="Local PromptDA checkpoint path or HF repo id. Defaults to official PromptDA checkpoint for --encoder.",
    )
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--learnable_lidar", type=str2bool, default=False)

    p.add_argument("--run_name", type=str, default="sacg")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr_sacg", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_iter", type=int, default=10000)
    p.add_argument("--log_freq", type=int, default=200)
    p.add_argument("--eval_freq", type=int, default=1000)
    p.add_argument("--save_freq", type=int, default=5000)
    p.add_argument("--visual_freq", type=int, default=5000)
    p.add_argument("--num_visuals", type=int, default=4)
    p.add_argument("--eval_only", type=str2bool, default=False)

    p.add_argument("--boundary_weight", type=float, default=0.5)
    p.add_argument("--gate_weight", type=float, default=0.1)
    p.add_argument("--gate_warmup_step", type=int, default=5000)
    p.add_argument("--use_boundary_loss", type=str2bool, default=True)

    p.add_argument("--use_wandb", type=str2bool, default=True)
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--log_dir", type=str, default="log")
    p.add_argument("--tbp", type=int, default=None)
    p.add_argument("--compare_after_train", type=str2bool, default=True)
    p.add_argument("--compare_save_visuals", type=str2bool, default=True)
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


def build_transforms(split: str):
    if split == "train":
        return Compose([
            transfroms.RandomFilpLR(),
            transfroms.ValidDepthMask(gt_low_limit=0.01),
            transfroms.AsContiguousArray(),
        ])
    return Compose([
        transfroms.ValidDepthMask(gt_low_limit=0.01),
    ])


def build_train_loader(args, start_itr: int):
    ds = MyARKitScenesDataset(
        root=args.data_root,
        split="train",
        max_samples=args.max_samples,
        seed=args.seed,
        transform=build_transforms("train"),
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
        max_samples=args.max_val_samples,
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


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def assert_output_size(output: dict[str, torch.Tensor], target: torch.Tensor):
    pred = output["refined_depth"]
    if pred.shape[-2:] != target.shape[-2:]:
        raise RuntimeError(
            f"SACG output shape {tuple(pred.shape[-2:])} does not match GT shape {tuple(target.shape[-2:])}. "
            "Dataset GT must be resized to PromptDA output/RGB size."
        )


def freeze_check(model: PromptDASACG):
    for name, param in model.promptda.named_parameters():
        if param.requires_grad:
            raise RuntimeError(f"PromptDA must be frozen, but '{name}' requires grad.")
    trainable = sum(p.numel() for p in model.sacg.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.sacg.parameters())
    Log.info(f"SACG trainable params: {trainable:,} / {total:,}")


def build_model(args, device: torch.device) -> PromptDASACG:
    model = PromptDASACG.from_pretrained(
        encoder=args.encoder,
        promptda_ckpt=args.promptda_ckpt,
        dpt_variant=args.dpt_variant,
        sacg_ckpt=None,
        learnable_lidar=args.learnable_lidar,
    ).to(device)
    freeze_check(model)
    return model


def save_checkpoint(
    path: Path,
    model: PromptDASACG,
    optimizer: torch.optim.Optimizer | None,
    scheduler,
    step: int,
    metrics: dict,
    best_l1: float,
):
    state = {
        "model": model.sacg.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "metrics": metrics,
        "best_l1": best_l1,
    }
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: PromptDASACG,
    optimizer: torch.optim.Optimizer | None,
    scheduler,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    model.sacg.load_state_dict(state_dict, strict=False)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    step = int(checkpoint.get("step", checkpoint.get("global_step", 0)))
    best_l1 = float(checkpoint.get("best_l1", float("inf")))
    Log.info(f"Loaded SACG checkpoint from {path} | step={step} | best L1={best_l1:.4f}")
    return step, best_l1


def compute_sacg_loss(loss_fn: SACGLoss, output: dict, target: torch.Tensor, step: int):
    valid_mask = target > 0
    return loss_fn(
        refined_depth=output["refined_depth"],
        coarse_depth=output["coarse_depth"],
        target=target,
        valid_mask=valid_mask,
        gate_map=output["gate_map"],
        c_grad=output["c_grad"],
        edge_strength=output["edge_strength"],
        epoch=step,
    )


@torch.no_grad()
def eval_sacg(model: PromptDASACG, loader, loss_fn: SACGLoss, device: torch.device, step: int):
    model.eval()
    losses = []
    metrics_list = []
    baseline_metrics_list = []
    for batch in tqdm(loader, desc=f"Val {step}", leave=False):
        batch = move_batch_to_device(batch, device)
        image = batch[dataset_keys.COLOR_IMG]
        target = batch[dataset_keys.HIGH_RES_DEPTH_IMG]
        prompt = batch[dataset_keys.LOW_RES_DEPTH_IMG]
        output = model(image, prompt)
        assert_output_size(output, target)
        loss, _ = compute_sacg_loss(loss_fn, output, target, step)
        losses.append(float(loss.item()))
        metrics_list.append(compute_depth_metrics(output["refined_depth"], target, rgb=image))
        baseline_metrics_list.append(compute_depth_metrics(output["coarse_depth"], target, rgb=image))
    return (
        sum(losses) / len(losses),
        aggregate_metrics(metrics_list),
        aggregate_metrics(baseline_metrics_list),
    )


@torch.no_grad()
def save_visuals(model: PromptDASACG, loader, device: torch.device, output_dir: Path, max_images: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    for batch_idx, batch in enumerate(loader):
        if saved >= max_images:
            break
        batch = move_batch_to_device(batch, device)
        image = batch[dataset_keys.COLOR_IMG]
        target = batch[dataset_keys.HIGH_RES_DEPTH_IMG]
        prompt = batch[dataset_keys.LOW_RES_DEPTH_IMG]
        identifiers = batch.get(dataset_keys.IDENTIFIER, [])
        output = model(image, prompt)
        assert_output_size(output, target)
        identifier = identifiers[0] if identifiers else f"{batch_idx:04d}"
        save_path = output_dir / f"{saved:04d}_{Path(identifier).stem}.png"
        plot_sacg_comparison(
            rgb=image[0],
            sparse_depth=prompt[0],
            target=target[0],
            baseline_depth=output["coarse_depth"][0],
            refined_depth=output["refined_depth"][0],
            gate_map=output["gate_map"][0],
            c_grad=output["c_grad"][0],
            f_lidar=output["f_lidar"][0],
            save_path=str(save_path),
        )
        saved += 1
    return saved


def run_compare_inference(args, ckpt_dir: Path):
    best_ckpt = ckpt_dir / "best.pth"
    if not best_ckpt.exists():
        Log.warn(f"Skipping compare_inference: best checkpoint not found at {best_ckpt}")
        return

    compare_script = Path(__file__).resolve().parents[1] / "compare_inference.py"
    command = [
        sys.executable,
        str(compare_script),
        "--data_root",
        args.data_root,
        "--split",
        "val",
        "--num_workers",
        str(args.num_workers),
        "--encoder",
        args.encoder,
        "--dpt_variant",
        args.dpt_variant,
        "--sacg_ckpt",
        str(best_ckpt),
        "--batch_size",
        "1",
        "--output_dir",
        str(ckpt_dir / "compare_best"),
        "--num_visuals",
        str(args.num_visuals),
    ]
    if args.promptda_ckpt is not None:
        command.extend(["--baseline_ckpt", args.promptda_ckpt])
    if args.max_val_samples is not None:
        command.extend(["--max_samples", str(args.max_val_samples)])
    if args.learnable_lidar:
        command.append("--learnable_lidar")
    if args.compare_save_visuals:
        command.append("--save_visuals")

    Log.info(f"Running compare_inference with best checkpoint: {best_ckpt}")
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(args.checkpoint_dir) / f"{args.run_name}_{args.encoder}_{args.dpt_variant}_{args.seed}_sacg"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    Log.info(f"Device      : {device}")
    Log.info(f"Run         : {args.run_name}")
    Log.info(f"Encoder     : {args.encoder}")
    Log.info(f"DPT variant : {args.dpt_variant}")
    Log.info(f"PromptDA    : {args.promptda_ckpt or 'official PromptDA default'}")
    Log.info(f"Eval only   : {args.eval_only}")

    val_loader = build_val_loader(args)
    model = build_model(args, device)
    loss_fn = SACGLoss(
        boundary_weight=args.boundary_weight,
        gate_weight=args.gate_weight,
        gate_warmup_epoch=args.gate_warmup_step,
        use_boundary_loss=args.use_boundary_loss,
    )

    optimizer = torch.optim.AdamW(
        model.sacg.parameters(),
        lr=args.lr_sacg,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.num_iter),
    )

    start_step = 0
    best_l1 = float("inf")
    if args.resume:
        start_step, best_l1 = load_checkpoint(args.resume, model, optimizer, scheduler, device)

    wandb_run = None
    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Install it with: pip install wandb")
        wandb_run = wandb.init(
            project="ObjectPromptDA",
            entity="ObjectPromptDA",
            name=f"{args.run_name}_{args.encoder}_{args.dpt_variant}_seed{args.seed}_sacg",
            dir=str(ckpt_dir),
            mode=args.wandb_mode,
            config=vars(args),
            tags=["sacg", args.encoder, args.dpt_variant, "promptda_frozen"],
        )

    if args.eval_only:
        val_loss, metrics, baseline_metrics = eval_sacg(model, val_loader, loss_fn, device, start_step)
        Log.info(f"[EVAL] loss={val_loss:.4f} | L1={metrics['L1']:.4f} | RMSE={metrics['RMSE']:.4f}")
        Log.info(f"[BASE] L1={baseline_metrics['L1']:.4f} | RMSE={baseline_metrics['RMSE']:.4f}")
        save_checkpoint(ckpt_dir / "eval.pth", model, None, None, start_step, metrics, best_l1)
        save_visuals(model, val_loader, device, ckpt_dir / "visualizations_eval", args.num_visuals)
        if wandb_run is not None:
            wandb_run.finish()
        return

    if args.tbp is not None:
        tensorboard_path = os.path.join(args.log_dir, TENSORBOARD_DIR, args.run_name)
        command = f"tensorboard --logdir {tensorboard_path} --port {args.tbp}"
        tensorboard_process = subprocess.Popen(shlex.split(command), env=os.environ.copy())
        train_writer = SummaryWriter(os.path.join(tensorboard_path, "train"), flush_secs=30)
        val_writer = SummaryWriter(os.path.join(tensorboard_path, "val"), flush_secs=30)
    else:
        tensorboard_process = None
        train_writer = None
        val_writer = None

    train_loader = build_train_loader(args, start_step)
    recent_losses = []
    recent_info = {"L_main": [], "L_boundary": [], "L_gate": []}
    start_time = time.time()
    duration = 0.0
    step = start_step + 1

    Log.info("start SACG training")
    model.train()
    for batch in train_loader:
        if step > args.num_iter:
            break

        before_op_time = time.time()
        batch = move_batch_to_device(batch, device)
        image = batch[dataset_keys.COLOR_IMG]
        target = batch[dataset_keys.HIGH_RES_DEPTH_IMG]
        prompt = batch[dataset_keys.LOW_RES_DEPTH_IMG]

        output = model(image, prompt)
        assert_output_size(output, target)
        loss, loss_info = compute_sacg_loss(loss_fn, output, target, step)
        if torch.isnan(loss).any():
            raise RuntimeError("NaN in SACG loss occurred. Aborting training.")

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.sacg.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        duration += time.time() - before_op_time
        recent_losses.append(loss.item())
        for key in recent_info:
            recent_info[key].append(loss_info[key])

        if step % args.log_freq == 0:
            avg_loss = sum(recent_losses) / len(recent_losses)
            avg_info = {
                key: sum(values) / len(values)
                for key, values in recent_info.items()
                if values
            }
            recent_losses = []
            recent_info = {"L_main": [], "L_boundary": [], "L_gate": []}
            current_lr = optimizer.param_groups[0].get("lr", -1)
            examples_per_sec = args.batch_size / max(duration, 1e-6) * args.log_freq
            time_sofar = (time.time() - start_time) / 3600
            time_left = (args.num_iter / step - 1.0) * time_sofar
            Log.info(
                f"step={step} | loss={avg_loss:.4f} | L_main={avg_info.get('L_main', 0):.4f} | "
                f"L_boundary={avg_info.get('L_boundary', 0):.4f} | L_gate={avg_info.get('L_gate', 0):.4f} | "
                f"examples/s={examples_per_sec:.2f} | elapsed={time_sofar:.2f}h | left={time_left:.2f}h"
            )
            if train_writer is not None:
                train_writer.add_scalar("loss", avg_loss, step)
                train_writer.add_scalar("lr", current_lr, step)
                for key, value in avg_info.items():
                    train_writer.add_scalar(key, value, step)
            if wandb_run is not None:
                wandb_run.log({
                    "step": step,
                    "train/loss": avg_loss,
                    "train/lr": current_lr,
                    **{f"train/{key}": value for key, value in avg_info.items()},
                })
            duration = 0.0

        if step % args.eval_freq == 0:
            val_loss, metrics, baseline_metrics = eval_sacg(model, val_loader, loss_fn, device, step)
            Log.info(
                f"[EVAL step={step}] val_loss={val_loss:.4f} | "
                f"L1={metrics['L1']:.4f} | RMSE={metrics['RMSE']:.4f} | AbsRel={metrics['AbsRel']:.4f}"
            )
            Log.info(
                f"[BASE step={step}] L1={baseline_metrics['L1']:.4f} | "
                f"RMSE={baseline_metrics['RMSE']:.4f} | AbsRel={baseline_metrics['AbsRel']:.4f}"
            )
            if val_writer is not None:
                val_writer.add_scalar("loss", val_loss, step)
                for key, value in metrics.items():
                    val_writer.add_scalar(key, value, step)
            if wandb_run is not None:
                wandb_run.log({
                    "step": step,
                    "val/loss": val_loss,
                    **{f"val/{key}": value for key, value in metrics.items()},
                    **{f"baseline/{key}": value for key, value in baseline_metrics.items()},
                })
            metrics_path = ckpt_dir / "metrics_history.jsonl"
            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "step": step,
                    "val_loss": val_loss,
                    "sacg": metrics,
                    "baseline": baseline_metrics,
                }) + "\n")

            if metrics["L1"] < best_l1:
                best_l1 = metrics["L1"]
                save_checkpoint(ckpt_dir / "best.pth", model, optimizer, scheduler, step, metrics, best_l1)
                Log.info(f"best.pth saved -> L1={best_l1:.4f}")
            model.train()

        if step % args.visual_freq == 0:
            saved = save_visuals(model, val_loader, device, ckpt_dir / f"visualizations_step-{step}", args.num_visuals)
            Log.info(f"Saved {saved} visualization(s) for step {step}")

        if step % args.save_freq == 0:
            save_checkpoint(
                ckpt_dir / f"checkpoint_step-{step}.pth",
                model,
                optimizer,
                scheduler,
                step,
                {},
                best_l1,
            )
            save_checkpoint(ckpt_dir / "latest.pth", model, optimizer, scheduler, step, {}, best_l1)
            Log.info(f"Checkpoint saved: {ckpt_dir / f'checkpoint_step-{step}.pth'}")

        step += 1

    save_checkpoint(ckpt_dir / "latest.pth", model, optimizer, scheduler, step - 1, {}, best_l1)
    Log.info(f"finished SACG training | best L1={best_l1:.4f}")
    if tensorboard_process is not None:
        tensorboard_process.terminate()
    if wandb_run is not None:
        wandb_run.finish()
    if args.compare_after_train:
        run_compare_inference(args, ckpt_dir)


if __name__ == "__main__":
    main()
