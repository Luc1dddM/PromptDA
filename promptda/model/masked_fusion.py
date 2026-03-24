from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


class MaskedLocalFusion(nn.Module):
    def __init__(
        self,
        in_channels: int,
        roi_output_size: int = 7,
        sampling_ratio: int = 2,
    ):
        super().__init__()
        self.roi_output_size = roi_output_size
        self.sampling_ratio = sampling_ratio
        self.projector = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def _clamp_boxes(self, boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes

        clamped = boxes.clone()
        clamped[:, [0, 2]] = clamped[:, [0, 2]].clamp(min=0, max=width)
        clamped[:, [1, 3]] = clamped[:, [1, 3]].clamp(min=0, max=height)

        # Ensure non-zero extent.
        clamped[:, 2] = torch.maximum(clamped[:, 2], clamped[:, 0] + 1.0)
        clamped[:, 3] = torch.maximum(clamped[:, 3], clamped[:, 1] + 1.0)
        clamped[:, 2] = clamped[:, 2].clamp(max=width)
        clamped[:, 3] = clamped[:, 3].clamp(max=height)
        return clamped

    def forward(
        self,
        f_global: torch.Tensor,
        boxes: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if boxes is None or len(boxes) == 0:
            # Keep an autograd path through MLF parameters even when no boxes are provided.
            # Numerical output is unchanged.
            return f_global + 0.0 * self.projector(f_global)

        batch_size, _, height, width = f_global.shape
        if len(boxes) != batch_size:
            raise ValueError(f"Expected {batch_size} box tensors, got {len(boxes)}")

        if all(b.numel() == 0 for b in boxes):
            # Keep an autograd path through MLF parameters even when all box tensors are empty.
            # Numerical output is unchanged.
            return f_global + 0.0 * self.projector(f_global)

        device = f_global.device
        dtype = f_global.dtype

        canvas = torch.zeros_like(f_global)
        mask = torch.zeros((batch_size, 1, height, width), device=device, dtype=dtype)

        rois = []
        roi_index_map = []
        clamped_boxes_per_image: List[torch.Tensor] = []

        for b_idx, b in enumerate(boxes):
            if b.numel() == 0:
                clamped = torch.empty((0, 4), device=device, dtype=dtype)
                clamped_boxes_per_image.append(clamped)
                continue

            b = b.to(device=device, dtype=dtype)
            clamped = self._clamp_boxes(b, height=height, width=width)
            clamped_boxes_per_image.append(clamped)
            for i in range(clamped.shape[0]):
                rois.append(
                    torch.tensor(
                        [
                            float(b_idx),
                            clamped[i, 0].item(),
                            clamped[i, 1].item(),
                            clamped[i, 2].item(),
                            clamped[i, 3].item(),
                        ],
                        device=device,
                        dtype=dtype,
                    )
                )
                roi_index_map.append((b_idx, i))

        if len(rois) == 0:
            # Safe fallback that preserves gradient flow to MLF parameters.
            return f_global + 0.0 * self.projector(f_global)

        rois_tensor = torch.stack(rois, dim=0)
        roi_patches = roi_align(
            input=f_global,
            boxes=rois_tensor,
            output_size=(self.roi_output_size, self.roi_output_size),
            spatial_scale=1.0,
            sampling_ratio=self.sampling_ratio,
            aligned=True,
        )

        for patch_idx, (b_idx, local_idx) in enumerate(roi_index_map):
            box = clamped_boxes_per_image[b_idx][local_idx]
            x1 = int(torch.floor(box[0]).item())
            y1 = int(torch.floor(box[1]).item())
            x2 = int(torch.ceil(box[2]).item())
            y2 = int(torch.ceil(box[3]).item())

            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(x1 + 1, min(x2, width))
            y2 = max(y1 + 1, min(y2, height))

            target_h = y2 - y1
            target_w = x2 - x1
            resized_patch = F.interpolate(
                roi_patches[patch_idx : patch_idx + 1],
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
            canvas[b_idx : b_idx + 1, :, y1:y2, x1:x2] = (
                canvas[b_idx : b_idx + 1, :, y1:y2, x1:x2] + resized_patch
            )
            mask[b_idx : b_idx + 1, :, y1:y2, x1:x2] = 1.0

        projected = self.projector(canvas)
        f_enhanced = f_global + projected * mask
        return f_enhanced
