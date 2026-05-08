"""
Inference script for depth upsampling models trained with depth_upsampling/train.py.
Supports models trained with different losses and network types (MSG, MSPF).

Usage:
    # For MSG model
    python inference_baseline.py \
        --data_path data/ARKitScenes/data/upsampling \
        --network MSG \
        --checkpoint_path log/checkpoint_step-20000 \
        --upsample_factor 2 \
        --output_dir results/msg

    # For MSPF model
    python inference_baseline.py \
        --data_path data/ARKitScenes/data/upsampling \
        --network MSPF \
        --checkpoint_path log/checkpoint_step-20000 \
        --upsample_factor 2 \
        --output_dir results/mspf
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.transforms import Compose

# Add depth_upsampling to path
depth_upsampling_path = os.path.join(os.path.dirname(__file__), 'data', 'ARKitScenes', 'depth_upsampling')
sys.path.insert(0, depth_upsampling_path)

import transfroms
from dataset import ARKitScenesDataset
from models import get_network
from data_utils import batch_to_cuda
from training.metrics import compute_depth_metrics, aggregate_metrics
from promptda.utils.logger import Log


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Depth Upsampling Inference")

    # Data
    p.add_argument("--data_path", type=str, default="data/ARKitScenes/data/upsampling",
                   help="Path to ARKitScenes dataset")
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)

    # Network
    p.add_argument("--network", type=str, required=True,
                   choices=["MSG", "MSPF"],
                   help="Network model class")
    p.add_argument("--upsample_factor", type=int, default=2,
                   choices=[2, 4, 8],
                   help="Upsample scale from low to high resolution")

    # Checkpoint
    p.add_argument("--checkpoint_path", type=str, required=True,
                   help="Path to checkpoint to load")

    # Output
    p.add_argument("--output_dir", type=str, default="results/inference")
    p.add_argument("--save_depth", action="store_true")
    p.add_argument("--batch_size", type=int, default=1)

    return p.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Log.info(f"Device        : {device}")
    Log.info(f"Network       : {args.network}")
    Log.info(f"Split         : {args.split}")
    Log.info(f"Batch size    : {args.batch_size}")
    Log.info(f"Upsample factor: {args.upsample_factor}")
    Log.info(f"Checkpoint    : {args.checkpoint_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_depth:
        depth_dir = os.path.join(args.output_dir, "depth_maps")
        os.makedirs(depth_dir, exist_ok=True)

    # ── Transform ───────────────────────────────────────────────────────── #
    # Use ModCrop for validation (same as training script)
    patch_size = 256 if args.upsample_factor == 2 else 512
    transform = Compose([
        transfroms.ModCrop(modulo=32),
        transfroms.ValidDepthMask(gt_low_limit=0.01)
    ])

    # ── Dataset ─────────────────────────────────────────────────────────── #
    Log.info(f"Loading '{args.split}' dataset from: {args.data_path}")

    dataset = ARKitScenesDataset(
        root=args.data_path,
        split=args.split,
        upsample_factor=args.upsample_factor,
        transform=transform
    )

    if args.max_samples is not None:
        dataset = torch.utils.data.Subset(
            dataset, range(min(args.max_samples, len(dataset)))
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    Log.info(f"Dataset size  : {len(dataset)} | Batches: {len(loader)}")

    # ── Model ────────────────────────────────────────────────────────────── #
    Log.info("Loading model...")
    model = get_network(args.network, args.upsample_factor)

    # Load checkpoint
    if os.path.exists(args.checkpoint_path):
        Log.info(f"Loading checkpoint from: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        Log.info("Checkpoint loaded successfully.")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    cudnn.benchmark = True
    Log.info("Model ready.")

    # ── Inference ────────────────────────────────────────────────────────── #
    all_metrics = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Inference")):

            # Move batch to device
            batch = batch_to_cuda(batch)

            # Get inputs
            color_img = batch['color_img']           # (B, 3, H, W)
            depth_gt = batch['high_res_depth_img']   # (B, 1, H, W)
            low_res_depth = batch['low_res_depth_img'] # (B, 1, h, w)

            # Concatenate low-res depth with color image (as in training)
            # The model expects concatenated input: [color_img, low_res_depth]
            # This is the standard format for depth upsampling networks
            input_tensor = torch.cat([color_img, low_res_depth], dim=1)  # (B, 4, H, W)

            # Forward pass
            pred = model(input_tensor)  # (B, 1, H, W)

            # Ensure prediction matches ground truth size
            if pred.shape[-2:] != depth_gt.shape[-2:]:
                pred = torch.nn.functional.interpolate(
                    pred,
                    size=depth_gt.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            # Compute metrics
            all_metrics.append(compute_depth_metrics(pred, depth_gt))

            # Save depth maps if requested
            if args.save_depth:
                for i in range(pred.shape[0]):
                    fname = f"batch_{batch_idx:04d}_sample_{i:02d}.npy"
                    np.save(os.path.join(depth_dir, fname), pred[i, 0].cpu().numpy())

    # ── Results ──────────────────────────────────────────────────────────── #
    agg = aggregate_metrics(all_metrics)

    Log.info("=" * 60)
    Log.info(f"Inference Results  [{args.split}]")
    Log.info("=" * 60)
    for k, v in sorted(agg.items()):
        Log.info(f"  {k:20s}: {v:.6f}")
    Log.info("=" * 60)

    txt_path = os.path.join(args.output_dir, "metrics.txt")
    with open(txt_path, "w") as f:
        f.write(f"Inference Results ({args.split})\n")
        f.write(f"Network: {args.network}\n")
        f.write(f"Upsample factor: {args.upsample_factor}\n")
        f.write(f"Checkpoint: {args.checkpoint_path}\n")
        f.write("=" * 60 + "\n")
        for k, v in sorted(agg.items()):
            f.write(f"  {k:20s}: {v:.6f}\n")
        f.write("=" * 60 + "\n")
    Log.info(f"Saved -> {txt_path}")

    npz_path = os.path.join(args.output_dir, "metrics.npz")
    np.savez(npz_path, **agg)
    Log.info(f"Saved -> {npz_path}")

    return agg


if __name__ == "__main__":
    main()
