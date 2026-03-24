import json
import os
from pathlib import Path
 
import cv2
import numpy as np
import torch


def parse_pincam(pincam_path: str):
    """
    Parse a .pincam intrinsics file.
    Returns: (w, h, fx, fy, cx, cy) as floats.
    """
    with open(pincam_path, "r") as f:
        vals = f.read().strip().split()
    w, h, fx, fy, cx, cy = [float(v) for v in vals]
    return w, h, fx, fy, cx, cy


def load_depth_png(depth_path: str) -> np.ndarray:
    """
    Load a uint16 depth PNG (millimeters) and convert to float32 meters.
    Invalid pixels (value == 0) become NaN.
    """
    depth_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth_m = depth_mm / 1000.0
    depth_m[depth_m == 0.0] = np.nan
    return depth_m


def sparse_depth_to_prompt(depth_m: np.ndarray) -> torch.Tensor:
    """
    Convert a dense LiDAR depth map (with NaNs for invalid) to a sparse
    prompt tensor [1, H, W]. NaN → 0.0 so the network knows those pixels
    carry no metric information.
    """
    prompt = np.nan_to_num(depth_m, nan=0.0)
    return torch.from_numpy(prompt).unsqueeze(0).float()


def load_boxes(box_path: str, device="cpu") -> torch.Tensor:
    """
    Load pre-computed YOLOv8 bounding boxes from a JSON sidecar file.

    Expected format:
        {
            "image_path": "...",
            "image_size": [H, W],
            "feature_size": [feat_H, feat_W],
            "boxes_xyxy_feature": [[x1, y1, x2, y2], ...]
        }

    Returns a [N, 4] float tensor in xyxy feature-map coords.
    Returns an empty [0, 4] tensor if file does not exist or boxes are empty.
    """
    if not os.path.exists(box_path):
        return torch.zeros((0, 4), dtype=torch.float32)
    with open(box_path, "r") as f:
        data = json.load(f)
    boxes = data.get("boxes_xyxy_feature", [])
    if len(boxes) == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)