"""
Inference/evaluation for the official PromptDA baseline checkpoint.

This script keeps the evaluation pipeline aligned with our local training code:
  - MyARKitScenesDataset + collate_fn
  - validation transform: ValidDepthMask(gt_low_limit=0.01)
  - model call: model(color_img, low_res_depth_img)
  - GT is loaded at PromptDA output/RGB size, so prediction and GT match directly
  - same loss and metrics as Trainer.eval_epoch

By default it loads the official PromptDA checkpoint for the selected encoder.
You can also point `--checkpoint_path` to a local PromptDA checkpoint or a
Hugging Face repo id.

Example:
    python inference_baseline.py \
        --data_root data/ARKitScenes/data/upsampling \
        --encoder vitl \
        --output_dir results/promptda_baseline
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

# Match training/train.py import setup. Some ARKit transforms import the legacy
# top-level names directly, so register the package modules before importing them.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from data.ARKitScenes.depth_upsampling import dataset_keys as arkit_dataset_keys
from data.ARKitScenes.depth_upsampling import image_utils as arkit_image_utils

sys.modules.setdefault("dataset_keys", arkit_dataset_keys)
sys.modules.setdefault("image_utils", arkit_image_utils)

from data.ARKitScenes.depth_upsampling import transfroms
from dataset.dataset import MyARKitScenesDataset, collate_fn
from promptda.promptda import PromptDA
from promptda.promptda_uncertainty import PromptDAUncertainty
from promptda.utils.logger import Log
from training.loss import CombinedLoss
from training.loss_laplace import RobustLaplaceNLLLoss
from training.metrics import aggregate_metrics, compute_depth_metrics
from training.visualization import plot_uncertainty


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {value}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate PromptDA baseline checkpoints with the training pipeline"
    )

    # Data: names mirror training/train.py for less foot-gunning.
    p.add_argument("--data_root", type=str, default="data/ARKitScenes/data/upsampling")
    p.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Deprecated alias for --data_root; kept for old commands.",
    )
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    # Model: same architecture switches used in training/train.py.
    p.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitb", "vitl"])
    p.add_argument(
        "--dpt_variant",
        type=str,
        default="legacy",
        choices=["legacy", "skip_concat_1x1", "hybrid_ca_shallow_concat"],
    )
    p.add_argument(
        "--uncertainty",
        type=str2bool,
        default=False,
        help="Use PromptDAUncertainty architecture and Laplace NLL eval loss.",
    )
    # Checkpoint.
    p.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional local PromptDA checkpoint path or HF repo id. Defaults to the official PromptDA checkpoint for --encoder.",
    )

    # Output.
    p.add_argument("--output_dir", type=str, default="results/inference_baseline")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--save_depth", action="store_true")
    p.add_argument("--save_visuals", action="store_true")
    p.add_argument("--num_visuals", type=int, default=8)
    p.add_argument("--sigma_percentile", type=float, default=99.0)

    return p.parse_args()


# --------------------------------------------------------------------------- #
# Training-compatible builders
# --------------------------------------------------------------------------- #


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


def build_loader(args, device: torch.device) -> DataLoader:
    dataset = MyARKitScenesDataset(
        root=args.data_root,
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
        transform=build_transforms(args.split),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    Log.info(f"{args.split}: {len(dataset)} samples | batches: {len(loader)}")
    return loader


def build_model(args) -> torch.nn.Module:
    if args.uncertainty:
        return PromptDAUncertainty.from_pretrained(
            pretrained_model_name_or_path=args.checkpoint_path,
            encoder=args.encoder,
            dpt_variant=args.dpt_variant,
        )
    return PromptDA.from_pretrained(
        pretrained_model_name_or_path=args.checkpoint_path,
        encoder=args.encoder,
        use_mlf=False,
        dpt_variant=args.dpt_variant,
    )


# --------------------------------------------------------------------------- #
# Prediction helpers
# --------------------------------------------------------------------------- #


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def assert_prediction_size(pred, target_hw: tuple[int, int]):
    if isinstance(pred, dict):
        pred_depth = pred["mu"]
    else:
        pred_depth = pred

    if pred_depth.shape[-2:] != target_hw:
        raise RuntimeError(
            f"Prediction shape {tuple(pred_depth.shape[-2:])} does not match GT shape {target_hw}. "
            "Dataset GT should be resized to PromptDA output/RGB size."
        )
    return pred


def prediction_depth(pred) -> torch.Tensor:
    return pred["mu"] if isinstance(pred, dict) else pred


def prediction_sigma(pred) -> torch.Tensor | None:
    if not isinstance(pred, dict) or "s" not in pred:
        return None
    return torch.exp(torch.clamp(pred["s"], min=-10.0, max=10.0))


def safe_identifier(identifier: str, fallback: str) -> str:
    stem = Path(identifier).stem if identifier else fallback
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem) or fallback


def save_prediction_depths(pred_depth: torch.Tensor, identifiers: list[str], depth_dir: str, batch_idx: int):
    for i in range(pred_depth.shape[0]):
        name = safe_identifier(identifiers[i] if i < len(identifiers) else "", f"{batch_idx:04d}_{i:02d}")
        np.save(os.path.join(depth_dir, f"{name}.npy"), pred_depth[i, 0].detach().cpu().numpy())


def save_prediction_visuals(
    image: torch.Tensor,
    depth_gt: torch.Tensor,
    pred_depth: torch.Tensor,
    sigma: torch.Tensor | None,
    identifiers: list[str],
    visual_dir: str,
    batch_idx: int,
    saved_count: int,
    max_visuals: int,
    sigma_percentile: float,
) -> int:
    remaining = max_visuals - saved_count
    if remaining <= 0:
        return saved_count

    n_save = min(pred_depth.shape[0], remaining)
    for i in range(n_save):
        name = safe_identifier(identifiers[i] if i < len(identifiers) else "", f"{batch_idx:04d}_{i:02d}")
        out_path = os.path.join(visual_dir, f"{saved_count + i:04d}_{name}.png")
        plot_uncertainty(
            rgb=image[i],
            target=depth_gt[i],
            baseline=None if sigma is not None else pred_depth[i],
            mu=pred_depth[i] if sigma is not None else None,
            sigma=None if sigma is None else sigma[i],
            save_path=out_path,
            sigma_percentile=sigma_percentile,
        )
    return saved_count + n_save


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    args = parse_args()
    if args.data_path is not None:
        Log.warn("--data_path is deprecated; using it as --data_root for compatibility.")
        args.data_root = args.data_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Log.info(f"Device      : {device}")
    Log.info(f"Data root   : {args.data_root}")
    Log.info(f"Split       : {args.split}")
    Log.info(f"Encoder     : {args.encoder}")
    Log.info(f"DPT variant : {args.dpt_variant}")
    Log.info(f"Uncertainty : {args.uncertainty}")
    Log.info(f"Checkpoint  : {args.checkpoint_path or 'official PromptDA default'}")

    os.makedirs(args.output_dir, exist_ok=True)
    depth_dir = os.path.join(args.output_dir, "depth_maps")
    visual_dir = os.path.join(args.output_dir, "visualizations")
    if args.save_depth:
        os.makedirs(depth_dir, exist_ok=True)
    if args.save_visuals:
        os.makedirs(visual_dir, exist_ok=True)

    loader = build_loader(args, device)

    Log.info("Building model...")
    model = build_model(args)
    model = model.to(device).eval()

    loss_fn = RobustLaplaceNLLLoss() if args.uncertainty else CombinedLoss()
    metrics_list = []
    loss_values = []
    saved_visuals = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Inference")):
            batch = move_batch_to_device(batch, device)

            image = batch[arkit_dataset_keys.COLOR_IMG]
            depth_gt = batch[arkit_dataset_keys.HIGH_RES_DEPTH_IMG]
            prompt = batch[arkit_dataset_keys.LOW_RES_DEPTH_IMG]
            identifiers = batch.get(arkit_dataset_keys.IDENTIFIER, [])

            pred = model(image, prompt)
            pred = assert_prediction_size(pred, depth_gt.shape[-2:])
            pred_depth = prediction_depth(pred)
            sigma = prediction_sigma(pred)

            loss, _ = loss_fn(pred, depth_gt)
            loss_values.append(float(loss.item()))
            metrics_list.append(compute_depth_metrics(pred_depth, depth_gt))

            if args.save_depth:
                save_prediction_depths(pred_depth, identifiers, depth_dir, batch_idx)

            if args.save_visuals and saved_visuals < args.num_visuals:
                saved_visuals = save_prediction_visuals(
                    image=image,
                    depth_gt=depth_gt,
                    pred_depth=pred_depth,
                    sigma=sigma,
                    identifiers=identifiers,
                    visual_dir=visual_dir,
                    batch_idx=batch_idx,
                    saved_count=saved_visuals,
                    max_visuals=args.num_visuals,
                    sigma_percentile=args.sigma_percentile,
                )

    metrics = aggregate_metrics(metrics_list)
    avg_loss = sum(loss_values) / len(loss_values) if loss_values else float("nan")
    results = {
        "val_loss": avg_loss,
        **metrics,
    }

    Log.info("=" * 60)
    Log.info(f"PromptDA Baseline Inference [{args.split}]")
    Log.info("=" * 60)
    Log.info(f"  {'val_loss':20s}: {avg_loss:.6f}")
    for key, value in sorted(metrics.items()):
        Log.info(f"  {key:20s}: {value:.6f}")
    Log.info("=" * 60)

    txt_path = os.path.join(args.output_dir, "metrics.txt")
    with open(txt_path, "w") as f:
        f.write(f"PromptDA Baseline Inference ({args.split})\n")
        f.write(f"Data root: {args.data_root}\n")
        f.write(f"Encoder: {args.encoder}\n")
        f.write(f"DPT variant: {args.dpt_variant}\n")
        f.write(f"Uncertainty: {args.uncertainty}\n")
        f.write(f"Checkpoint: {args.checkpoint_path or 'official PromptDA default'}\n")
        f.write("=" * 60 + "\n")
        f.write(f"  {'val_loss':20s}: {avg_loss:.6f}\n")
        for key, value in sorted(metrics.items()):
            f.write(f"  {key:20s}: {value:.6f}\n")
        f.write("=" * 60 + "\n")
    Log.info(f"Saved -> {txt_path}")

    json_path = os.path.join(args.output_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    Log.info(f"Saved -> {json_path}")

    npz_path = os.path.join(args.output_dir, "metrics.npz")
    np.savez(npz_path, **results)
    Log.info(f"Saved -> {npz_path}")

    if args.save_depth:
        Log.info(f"Saved depth maps -> {depth_dir}")
    if args.save_visuals:
        Log.info(f"Saved {saved_visuals} visualization(s) -> {visual_dir}")

    return results


if __name__ == "__main__":
    main()
