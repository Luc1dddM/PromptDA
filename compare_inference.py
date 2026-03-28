"""
Comparison script: Baseline vs MLF inference.

Runs both baseline and MLF models on the same dataset and compares metrics side-by-side.

Usage:
    python compare_inference.py --data_root data/ARKitScenes/data/upsampling \
                                --encoder vitl \
                                --mlf_checkpoint checkpoints/experiment/best.pth \
                                --output_dir results/comparison
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dataset.arkitscene import ARKitScenesDataset, collate_fn
from promptda.promptda import PromptDA
from promptda.utils.logger import Log
from training.metrics import compute_depth_metrics, aggregate_metrics


def parse_args():
    p = argparse.ArgumentParser(description="PromptDA Baseline vs MLF Comparison")

    # Data
    p.add_argument("--data_root", type=str, default="data/ARKitScenes/data/upsampling")
    p.add_argument("--split", type=str, default="Validation", choices=["Training", "Validation"])
    p.add_argument("--image_size", type=int, nargs=2, default=[196, 252])
    p.add_argument("--num_workers", type=int, default=4)

    # Model
    p.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl"])
    p.add_argument("--pretrained_path", type=str, default=None,
                   help="Baseline pretrained checkpoint")
    p.add_argument("--mlf_checkpoint", type=str, required=True,
                   help="MLF trained checkpoint")

    # Output
    p.add_argument("--output_dir", type=str, default="results/comparison")
    p.add_argument("--batch_size", type=int, default=4)

    return p.parse_args()


def load_checkpoint(ckpt_path, model, device):
    """Load checkpoint weights."""
    if not os.path.exists(ckpt_path):
        Log.warn(f"Checkpoint not found: {ckpt_path}")
        return False
    
    Log.info(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    
    if "model" in state:
        state_dict = state["model"]
    else:
        state_dict = state
    
    model.load_state_dict(state_dict, strict=False)
    return True


def run_inference(model, loader, device, model_name):
    """Run inference and collect metrics."""
    Log.info(f"Running {model_name} inference...")
    all_metrics = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=model_name, leave=False):
            image = batch["image"].to(device)
            depth_gt = batch["depth_gt"].to(device)
            prompt = batch["prompt"].to(device)
            boxes = [b.to(device) for b in batch["boxes"]]

            pred = model(image, prompt, boxes=boxes if boxes[0].numel() > 0 else None)
            metrics = compute_depth_metrics(pred, depth_gt)
            all_metrics.append(metrics)

    return aggregate_metrics(all_metrics)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Log.info(f"Device: {device}")
    Log.info(f"Encoder: {args.encoder}")
    Log.info(f"Split: {args.split}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    Log.info(f"Loading {args.split} dataset...")
    dataset = ARKitScenesDataset(
        data_root=args.data_root,
        split=args.split,
        image_size=tuple(args.image_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    Log.info(f"Dataset size: {len(dataset)}")

    # Load baseline model
    Log.info(f"Loading baseline model (no MLF)...")
    baseline_model = PromptDA.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_path,
        encoder=args.encoder,
        use_mlf=False,
    ).to(device).eval()

    # Load MLF model
    Log.info(f"Loading MLF model...")
    mlf_model = PromptDA.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_path,
        encoder=args.encoder,
        use_mlf=True,
    ).to(device).eval()

    # Load MLF checkpoint
    if not load_checkpoint(args.mlf_checkpoint, mlf_model, device):
        Log.warn("Using random MLF weights (this will give poor results)")

    # Run inference for both models
    baseline_metrics = run_inference(baseline_model, loader, device, "Baseline")
    mlf_metrics = run_inference(mlf_model, loader, device, "MLF")

    # Compute improvements
    improvements = {}
    for key in baseline_metrics.keys():
        baseline_val = baseline_metrics[key]
        mlf_val = mlf_metrics[key]
        
        # For metrics where lower is better (loss, error metrics)
        if key in ["AbsRel", "MAE", "RMSE", "Log10", "SILog"]:
            improvement = (baseline_val - mlf_val) / (baseline_val + 1e-8) * 100
            improvements[key] = improvement
        # For metrics where higher is better (delta)
        else:
            improvement = (mlf_val - baseline_val) / (baseline_val + 1e-8) * 100
            improvements[key] = improvement

    # Print comparison
    Log.info("=" * 80)
    Log.info(f"Baseline vs MLF Comparison ({args.split})")
    Log.info("=" * 80)
    Log.info(f"{'Metric':<15} {'Baseline':<15} {'MLF':<15} {'Improvement':<15}")
    Log.info("-" * 80)
    
    for key in sorted(baseline_metrics.keys()):
        baseline_val = baseline_metrics[key]
        mlf_val = mlf_metrics[key]
        improvement = improvements[key]
        
        improvement_sign = "↓" if improvement < 0 else "↑"
        log_str = f"{key:<15} {baseline_val:<15.6f} {mlf_val:<15.6f} {improvement_sign} {abs(improvement):>10.2f}%"
        Log.info(log_str)
    
    Log.info("=" * 80)

    # Save results
    results = {
        "baseline": baseline_metrics,
        "mlf": mlf_metrics,
        "improvements": improvements,
    }
    
    results_file = os.path.join(args.output_dir, "comparison_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    Log.info(f"Results saved to: {results_file}")

    # Save comparison as text
    txt_file = os.path.join(args.output_dir, "comparison_results.txt")
    with open(txt_file, "w") as f:
        f.write(f"Baseline vs MLF Comparison ({args.split})\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Metric':<15} {'Baseline':<15} {'MLF':<15} {'Improvement':<15}\n")
        f.write("-" * 80 + "\n")
        for key in sorted(baseline_metrics.keys()):
            baseline_val = baseline_metrics[key]
            mlf_val = mlf_metrics[key]
            improvement = improvements[key]
            improvement_sign = "↓" if improvement < 0 else "↑"
            f.write(f"{key:<15} {baseline_val:<15.6f} {mlf_val:<15.6f} {improvement_sign} {abs(improvement):>10.2f}%\n")
        f.write("=" * 80 + "\n")
    Log.info(f"Comparison saved to: {txt_file}")

    return results


if __name__ == "__main__":
    main()
