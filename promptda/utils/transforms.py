"""
transforms.py  -  Pre-processing transforms for PromptDA inference.

Có 2 transform tách biệt:
    ImageTransform      : nhận image tensor (C, H, W) — dùng cho dataset.transform
    BatchResizeTransform: nhận cả batch dict sau DataLoader — resize đồng bộ image + depth

Lý do tách:
    Dataset gọi self.transform(image_tensor), không phải transform(sample_dict).
    Nên không thể resize depth_gt trong dataset transform.
    -> Resize toàn bộ dict ở bước collate/batch thay thế.
"""

import torch
import torch.nn.functional as F
import numpy as np


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _floor_to_multiple(x: float, multiple: int) -> int:
    """Floor x xuống bội số gần nhất của multiple."""
    return max(int(x // multiple) * multiple, multiple)


def _target_hw(h: int, w: int, max_size: int, patch_size: int) -> tuple[int, int]:
    """Tính (H, W) target: scale xuống nếu cần, floor về bội số patch_size."""
    scale = 1.0
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
    tar_h = _floor_to_multiple(h * scale, patch_size)
    tar_w = _floor_to_multiple(w * scale, patch_size)
    return tar_h, tar_w


# --------------------------------------------------------------------------- #
# 1. ImageTransform — dùng cho dataset.transform (chỉ nhận image tensor)
# --------------------------------------------------------------------------- #

class ImageTransform:
    """
    Resize image tensor (C, H, W) sao cho H, W là bội số của patch_size.

    Dataset gọi: sample["color_img"] = self.transform(sample["color_img"])
    Nên class này chỉ nhận và trả về image tensor, không động đến depth.

    Args:
        max_size   (int): Max kích thước cạnh dài. Default 1008 = 72*14.
        patch_size (int): Patch size của ViT. Default 14.
    """

    def __init__(self, max_size: int = 1008, patch_size: int = 14):
        self.max_size   = _floor_to_multiple(max_size, patch_size)
        self.patch_size = patch_size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (C, H, W) float tensor
        Returns:
            image: (C, H', W') với H', W' là bội số của patch_size
        """
        C, H, W = image.shape
        tar_h, tar_w = _target_hw(H, W, self.max_size, self.patch_size)

        if (H, W) == (tar_h, tar_w):
            return image

        return F.interpolate(
            torch.from_numpy(np.expand_dims(image, 0)),          # (1, C, H, W)
            size=(tar_h, tar_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)                     # (C, H', W')

    def __repr__(self):
        return f"ImageTransform(max_size={self.max_size}, patch_size={self.patch_size})"


# --------------------------------------------------------------------------- #
# 2. BatchResizeTransform — dùng sau DataLoader để resize depth đồng bộ
# --------------------------------------------------------------------------- #

class BatchResizeTransform:
    """
    Resize toàn bộ batch dict để đồng bộ spatial size giữa image và depth_gt.

    Dùng trong inference loop:
        batch = BatchResizeTransform()(batch)

    Lý do cần:
        dataset.transform chỉ resize image, còn depth_gt vẫn giữ size gốc.
        Nếu image và depth_gt khác size thì compute_depth_metrics sẽ lỗi.

    Args:
        patch_size (int): Đảm bảo depth_gt cũng là bội số của patch_size.
    """

    def __init__(self, patch_size: int = 14):
        self.patch_size = patch_size

    def __call__(self, batch: dict) -> dict:
        """
        Args:
            batch: dict với keys "color_img", "high_res_depth_img", "low_res_depth_img"
                   Mỗi tensor shape (B, C, H, W)
        Returns:
            batch: depth_gt resize khớp với image; prompt align patch grid
        """
        image    = batch["color_img"]           # (B, 3, H, W)
        depth_gt = batch["high_res_depth_img"]  # (B, 1, H_d, W_d)
        prompt   = batch["low_res_depth_img"]   # (B, 1, h, w)

        _, _, H, W = image.shape

        # Resize depth_gt khớp với image (image đã được ImageTransform resize)
        if depth_gt.shape[-2:] != (H, W):
            depth_gt = F.interpolate(
                depth_gt,
                size=(H, W),
                mode="nearest",   # nearest để tránh tạo ra depth value không thực
            )

        # Align prompt về bội số patch_size (thường đã nhỏ nên không cần scale)
        pH, pW = prompt.shape[-2:]
        p_tar_h = _floor_to_multiple(pH, self.patch_size)
        p_tar_w = _floor_to_multiple(pW, self.patch_size)
        if (pH, pW) != (p_tar_h, p_tar_w):
            prompt = F.interpolate(
                prompt,
                size=(p_tar_h, p_tar_w),
                mode="nearest",
            )

        batch = dict(batch)
        batch["color_img"]           = image
        batch["high_res_depth_img"]  = depth_gt
        batch["low_res_depth_img"]   = prompt
        return batch

    def __repr__(self):
        return f"BatchResizeTransform(patch_size={self.patch_size})"