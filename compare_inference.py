from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data.ARKitScenes.depth_upsampling import dataset_keys as arkit_dataset_keys
from data.ARKitScenes.depth_upsampling import image_utils as arkit_image_utils

sys.modules.setdefault("dataset_keys", arkit_dataset_keys)
sys.modules.setdefault("image_utils", arkit_image_utils)

from data.ARKitScenes.depth_upsampling import transfroms
from dataset.dataset import MyARKitScenesDataset, collate_fn
from promptda.promptda import PromptDA
from promptda.sacg import PromptDASACG
from promptda.utils.logger import Log
from training.metrics import aggregate_metrics, compute_depth_metrics
from training.visualization import plot_sacg_comparison


def parse_args():
    p = argparse.ArgumentParser(description="Compare official PromptDA baseline against SACG refinement")
    p.add_argument("--data_root", type=str, default="data/ARKitScenes/data/upsampling")
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl"])
    p.add_argument(
        "--dpt_variant",
        type=str,
        default="legacy",
        choices=["legacy", "skip_concat_1x1", "hybrid_ca_shallow_concat"],
    )
    p.add_argument(
        "--baseline_ckpt",
        type=str,
        default=None,
        help="Optional local PromptDA checkpoint path or HF repo id. Defaults to official PromptDA checkpoint.",
    )
    p.add_argument(
        "--sacg_ckpt",
        type=str,
        default=None,
        help="Optional SACG checkpoint. If omitted, SACG runs with random refinement weights for component debugging.",
    )
    p.add_argument("--learnable_lidar", action="store_true")

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--output_dir", type=str, default="results/sacg_compare")
    p.add_argument("--save_visuals", action="store_true")
    p.add_argument("--num_visuals", type=int, default=8)
    return p.parse_args()


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
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def assert_prediction_size(pred: torch.Tensor, target_hw: tuple[int, int], name: str):
    if pred.shape[-2:] != target_hw:
        raise RuntimeError(
            f"{name} shape {tuple(pred.shape[-2:])} does not match GT shape {target_hw}. "
            "Dataset GT should be resized to PromptDA output/RGB size."
        )


def save_visual(
    batch: dict,
    baseline_depth: torch.Tensor,
    sacg_output: dict[str, torch.Tensor],
    save_path: str,
):
    plot_sacg_comparison(
        rgb=batch[arkit_dataset_keys.COLOR_IMG][0],
        sparse_depth=batch[arkit_dataset_keys.LOW_RES_DEPTH_IMG][0],
        target=batch[arkit_dataset_keys.HIGH_RES_DEPTH_IMG][0],
        baseline_depth=baseline_depth[0],
        refined_depth=sacg_output["refined_depth"][0],
        gate_map=sacg_output["gate_map"][0],
        c_grad=sacg_output["c_grad"][0],
        f_lidar=sacg_output["f_lidar"][0],
        save_path=save_path,
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    visuals_dir = os.path.join(args.output_dir, "visualizations")
    if args.save_visuals:
        os.makedirs(visuals_dir, exist_ok=True)

    Log.info(f"Device        : {device}")
    Log.info(f"Data root     : {args.data_root}")
    Log.info(f"Split         : {args.split}")
    Log.info(f"Encoder       : {args.encoder}")
    Log.info(f"Baseline ckpt : {args.baseline_ckpt or 'official PromptDA default'}")
    Log.info(f"SACG ckpt     : {args.sacg_ckpt or 'None (random SACG refine head)'}")

    loader = build_loader(args, device)

    baseline = PromptDA.from_pretrained(
        pretrained_model_name_or_path=args.baseline_ckpt,
        encoder=args.encoder,
        use_mlf=False,
        dpt_variant=args.dpt_variant,
    ).to(device).eval()

    sacg_model = PromptDASACG.from_pretrained(
        encoder=args.encoder,
        promptda_ckpt=args.baseline_ckpt,
        dpt_variant=args.dpt_variant,
        sacg_ckpt=args.sacg_ckpt,
        learnable_lidar=args.learnable_lidar,
    ).to(device).eval()

    baseline_metrics = []
    sacg_metrics = []
    saved_visuals = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Compare")):
            batch = move_batch_to_device(batch, device)
            image = batch[arkit_dataset_keys.COLOR_IMG]
            depth_gt = batch[arkit_dataset_keys.HIGH_RES_DEPTH_IMG]
            prompt = batch[arkit_dataset_keys.LOW_RES_DEPTH_IMG]
            identifiers = batch.get(arkit_dataset_keys.IDENTIFIER, [])

            baseline_depth = baseline(image, prompt)
            sacg_output = sacg_model(image, prompt)
            assert_prediction_size(baseline_depth, depth_gt.shape[-2:], "baseline_depth")
            assert_prediction_size(sacg_output["refined_depth"], depth_gt.shape[-2:], "refined_depth")

            baseline_metrics.append(compute_depth_metrics(baseline_depth, depth_gt, rgb=image))
            sacg_metrics.append(
                compute_depth_metrics(sacg_output["refined_depth"], depth_gt, rgb=image)
            )

            if args.save_visuals and saved_visuals < args.num_visuals:
                identifier = identifiers[0] if identifiers else f"{batch_idx:04d}"
                save_path = os.path.join(visuals_dir, f"{saved_visuals:04d}_{Path(identifier).stem}.png")
                save_visual(batch, baseline_depth, sacg_output, save_path)
                saved_visuals += 1

    baseline_agg = aggregate_metrics(baseline_metrics)
    sacg_agg = aggregate_metrics(sacg_metrics)

    improvements = {}
    lower_is_better = {"L1", "RMSE", "AbsRel", "BoundaryAbsRel"}
    higher_is_better = {"delta1", "delta2", "delta3"}
    for key in sorted(set(baseline_agg.keys()) | set(sacg_agg.keys())):
        if key not in baseline_agg or key not in sacg_agg:
            continue
        base_val = baseline_agg[key]
        sacg_val = sacg_agg[key]
        if np.isnan(base_val) or np.isnan(sacg_val):
            improvements[key] = float("nan")
            continue
        if key in lower_is_better:
            improvements[key] = (base_val - sacg_val) / (abs(base_val) + 1e-8) * 100.0
        elif key in higher_is_better:
            improvements[key] = (sacg_val - base_val) / (abs(base_val) + 1e-8) * 100.0
        else:
            improvements[key] = sacg_val - base_val

    results = {
        "baseline": baseline_agg,
        "sacg": sacg_agg,
        "improvements_percent": improvements,
    }

    Log.info("=" * 88)
    Log.info("PromptDA Baseline vs SACG")
    Log.info("=" * 88)
    Log.info(f"{'Metric':<18} {'Baseline':<18} {'SACG':<18} {'Improvement':<18}")
    Log.info("-" * 88)
    for key in sorted(results["baseline"].keys()):
        base_val = results["baseline"][key]
        sacg_val = results["sacg"].get(key, float("nan"))
        imp = results["improvements_percent"].get(key, float("nan"))
        if np.isnan(imp):
            imp_str = "nan"
        elif key in lower_is_better or key in higher_is_better:
            imp_str = f"{imp:+.2f}%"
        else:
            imp_str = f"{imp:+.6f}"
        Log.info(f"{key:<18} {base_val:<18.6f} {sacg_val:<18.6f} {imp_str:<18}")
    Log.info("=" * 88)

    json_path = os.path.join(args.output_dir, "comparison_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    txt_path = os.path.join(args.output_dir, "comparison_results.txt")
    with open(txt_path, "w") as f:
        f.write("PromptDA Baseline vs SACG\n")
        f.write("=" * 88 + "\n")
        f.write(f"{'Metric':<18} {'Baseline':<18} {'SACG':<18} {'Improvement':<18}\n")
        f.write("-" * 88 + "\n")
        for key in sorted(results["baseline"].keys()):
            base_val = results["baseline"][key]
            sacg_val = results["sacg"].get(key, float("nan"))
            imp = results["improvements_percent"].get(key, float("nan"))
            if np.isnan(imp):
                imp_str = "nan"
            elif key in lower_is_better or key in higher_is_better:
                imp_str = f"{imp:+.2f}%"
            else:
                imp_str = f"{imp:+.6f}"
            f.write(f"{key:<18} {base_val:<18.6f} {sacg_val:<18.6f} {imp_str:<18}\n")
        f.write("=" * 88 + "\n")

    Log.info(f"Saved -> {json_path}")
    Log.info(f"Saved -> {txt_path}")
    if args.save_visuals:
        Log.info(f"Saved {saved_visuals} visualization(s) -> {visuals_dir}")

    return results


if __name__ == "__main__":
    main()
