"""
dataset/arkitscenes.py

ARKitScenes Dataset Loader for Prompt-DA + MLF Training.

Each __getitem__ returns:
    - image      : RGB image tensor [3, H, W], normalized
    - depth_gt   : High-res GT depth tensor [1, H, W], in meters
    - prompt     : Sparse LiDAR depth tensor [1, H, W] (point prompts, in meters)
    - boxes      : List of 2D bounding boxes in feature-map coords [N, 4] xyxy
                   (empty tensor if no sidecar file found)

ARKitScenes folder structure expected:
    data_root/
    └── {split}/                        # e.g., Training / Validation
        └── {scene_id}/
            ├── lowres_wide/            # RGB 256x192, 60 FPS, filenames = timestamps
            ├── lowres_depth/           # LiDAR sparse depth 256x192 (uint16, mm)
            ├── highres_depth/          # GT depth (uint16, mm) — same timestamps as lowres
            ├── lowres_wide_intrinsics/ # .pincam files per frame
            └── lowres_wide.traj        # camera poses (axis-angle + translation)

Bounding boxes (offline YOLOv8):
    Sidecar JSON files live alongside lowres_wide images:
        data_root/{split}/{scene_id}/boxes/{timestamp}.json
    Format: list of [x1, y1, x2, y2] in feature-map pixel coordinates
    Generate them with: scripts/precompute_boxes.py
"""

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.utils.arkitscene import load_depth_png, sparse_depth_to_prompt, load_boxes, parse_pincam

class ARKitScenesDataset(Dataset):
    """
    Args:
        data_root   : Root directory of ARKitScenes download.
        split       : "Training" or "Validation".
        image_size  : (H, W) to resize RGB and depths to. Default (192, 256).
        max_depth   : Clip GT depth values above this (meters). Default 10.0.
        feat_scale  : Ratio of feature-map size to image size, used to convert
                      bounding boxes from image coords → feature-map coords.
                      For ViT-B/14 at 518×518: feat_h/img_h ≈ 37/518 ≈ 0.0714.
                      At lowres (192×256): feat_h = round(192 * feat_scale).
                      Default 0.0714.
    """

    IMAGE_MEAN = [0.485, 0.456, 0.406]
    IMAGE_STD  = [0.229, 0.224, 0.225]

    PATCH_SIZE = 14  # DINOv2 patch size

    def __init__(
        self,
        data_root: str,
        split: str = "Training",
        image_size: tuple = (196, 252),  # H=14x14, W=18x14
        max_depth: float = 10.0,
    ):
        h, w = image_size
        assert h % self.PATCH_SIZE == 0, (
            f"image_size height {h} must be divisible by {self.PATCH_SIZE}. "
            f"Suggestion: {(h // self.PATCH_SIZE) * self.PATCH_SIZE} or "
            f"{((h // self.PATCH_SIZE) + 1) * self.PATCH_SIZE}"
        )
        assert w % self.PATCH_SIZE == 0, (
            f"image_size width {w} must be divisible by {self.PATCH_SIZE}. "
            f"Suggestion: {(w // self.PATCH_SIZE) * self.PATCH_SIZE} or "
            f"{((w // self.PATCH_SIZE) + 1) * self.PATCH_SIZE}"
        )

        self.data_root  = Path(data_root) / split
        self.image_size = image_size
        self.max_depth  = max_depth
        self.split      = split

        self.img_h, self.img_w = image_size
        self.feat_h = h // self.PATCH_SIZE
        self.feat_w = w // self.PATCH_SIZE

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=self.IMAGE_MEAN, std=self.IMAGE_STD)
        # `feat_h` and `feat_w` are exact by integer division.

        # Build flat list of (scene_id, timestamp) pairs
        self.samples = self._collect_samples()

    # ── internal ──────────────────────────────────────────────────────────────

    def _collect_samples(self):
        samples = []
        for scene_dir in sorted(self.data_root.iterdir()):
            if not scene_dir.is_dir():
                continue
            rgb_dir   = scene_dir / "wide"
            depth_dir = scene_dir / "lowres_depth"
            gt_dir    = scene_dir / "highres_depth"
            bbox      = scene_dir / "boxes"
            if not (rgb_dir.exists() and depth_dir.exists() and gt_dir.exists() and bbox.exists()):
                continue

            # Timestamps are the stems of depth files (most restrictive source)
            for depth_file in sorted(depth_dir.glob("*.png")):
                ts = depth_file.stem          # e.g., "47333462_6845.80601079"
                rgb_path = rgb_dir / f"{ts}.png"
                gt_path  = gt_dir  / f"{ts}.png"
                bbox_path = bbox / f"{ts}.boxes.json"
                if rgb_path.exists() and gt_path.exists() and bbox_path.exists():
                    samples.append((scene_dir, ts))
        return samples

    def _get_pincam(self, scene_dir: Path, ts: str) -> tuple:
        """Find the closest .pincam file to the given timestamp."""
        intrinsics_dir = scene_dir / "lowres_wide_intrinsics"
        # Timestamps in pincam filenames may differ slightly — find nearest
        ts_float = float(ts.split("_")[-1]) if "_" in ts else float(ts)
        pincam_files = sorted(intrinsics_dir.glob("*.pincam"))
        if not pincam_files:
            # Fallback: use a fixed intrinsic (256×192, approximate)
            return 256, 192, 211.9, 211.9, 127.9, 95.9
        best = min(
            pincam_files,
            key=lambda p: abs(float(p.stem.split("_")[-1]) - ts_float)
        )
        return parse_pincam(str(best))

    def _boxes_image_to_feat(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Scale bounding boxes from image pixel space → feature-map pixel space.
        boxes: [N, 4] xyxy in image coords (at self.image_size resolution).
        """
        if boxes.numel() == 0:
            return boxes
        scale_x = self.feat_w / self.img_w
        scale_y = self.feat_h / self.img_h
        scale = boxes.new_tensor([scale_x, scale_y, scale_x, scale_y])
        return boxes * scale

    # ── public ────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene_dir, ts = self.samples[idx]

        # ── 1. Load RGB ───────────────────────────────────────────────────────
        rgb_path = scene_dir / "wide" / f"{ts}.png"
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        image = self.normalize(self.to_tensor(rgb))  # [3, H, W]

        # ── 2. Load GT depth (high-res Faro laser scanner) ───────────────────
        gt_path  = scene_dir / "highres_depth" / f"{ts}.png"
        depth_gt = load_depth_png(str(gt_path))
        depth_gt = cv2.resize(depth_gt, (self.img_w, self.img_h),
                              interpolation=cv2.INTER_NEAREST)
        depth_gt = np.clip(np.nan_to_num(depth_gt, nan=0.0), 0.0, self.max_depth)
        depth_gt = torch.from_numpy(depth_gt).unsqueeze(0).float()  # [1, H, W]

        # ── 3. Load LiDAR sparse depth → point prompts ────────────────────────
        lidar_path = scene_dir / "lowres_depth" / f"{ts}.png"
        lidar_raw  = load_depth_png(str(lidar_path))
        lidar_raw  = cv2.resize(lidar_raw, (self.img_w, self.img_h),
                                interpolation=cv2.INTER_NEAREST)
        prompt = sparse_depth_to_prompt(lidar_raw)  # [1, H, W], 0 = no info

        # ── 4. Load bounding boxes (YOLOv8 offline sidecar) ──────────────────
        # boxes_xyxy_feature inside the JSON is already in feature-map coords
        box_path   = scene_dir / "boxes" / f"{ts}.boxes.json"
        boxes_feat = load_boxes(str(box_path))  # [N, 4] xyxy, feature-map coords

        return {
            "image":     image,       # [3, H, W]  float32, ImageNet-normalized
            "depth_gt":  depth_gt,    # [1, H, W]  float32, meters (0 = invalid)
            "prompt":    prompt,       # [1, H, W]  float32, meters (0 = no LiDAR)
            "boxes":     boxes_feat,  # [N, 4]     float32, xyxy feature-map coords
            "scene_id":  scene_dir.name,
            "timestamp": ts,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Collate function  (needed because "boxes" has variable N per sample)
# ──────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """
    Custom collate that stacks tensors of fixed shape normally, and keeps
    variable-length "boxes" as a Python list of tensors (one per image).
    This is the format expected by MaskedLocalFusion.forward(f_global, boxes).
    """
    images    = torch.stack([s["image"]    for s in batch])
    depth_gts = torch.stack([s["depth_gt"] for s in batch])
    prompts   = torch.stack([s["prompt"]   for s in batch])
    boxes     = [s["boxes"] for s in batch]   # list of [N_i, 4]

    return {
        "image":    images,
        "depth_gt": depth_gts,
        "prompt":   prompts,
        "boxes":    boxes,
        "scene_id": [s["scene_id"]  for s in batch],
        "timestamp":[s["timestamp"] for s in batch],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    data_root = sys.argv[1] if len(sys.argv) > 1 else "/media/lam/Volume3/114MasterYZU/Sem1/CV/PromptDA/data/ARKitScenes/data/upsampling"
    ds = ARKitScenesDataset(data_root=data_root, split="Training")
    print(f"Dataset size: {len(ds)} samples")

    loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_fn,
                        num_workers=0)
    batch = next(iter(loader))

    print("image    :", batch["image"].shape)     # [2, 3, 192, 256]
    print("depth_gt :", batch["depth_gt"].shape)  # [2, 1, 192, 256]
    print("prompt   :", batch["prompt"].shape)    # [2, 1, 192, 256]
    print("boxes[0] :", batch["boxes"][0].shape)  # [N, 4]
    print("boxes[1] :", batch["boxes"][1].shape)
    print("Smoke-test passed ✓")           