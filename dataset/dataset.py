import cv2
import numpy as np
import os
import json
from . import dataset_keys

from data.ARKitScenes.depth_upsampling.dataset import ARKitScenesDataset
from data.ARKitScenes.depth_upsampling.data_utils import image_hwc_to_chw, expand_channel_dim
from promptda.utils.io_wrapper import ensure_multiple_of

MILLIMETER_TO_METER = 1000
WIDE = 'wide'
PATCH_SIZE = 14  # DINOv2 patch size


class MyARKitScenesDataset(ARKitScenesDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ── Image loading ─────────────────────────────────────────────────────

    def load_image(self, path, shape, is_depth, sky_direction):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = ARKitScenesDataset.rotate_image(img, sky_direction)

        if is_depth:
            img = expand_channel_dim(
                np.asarray(img / MILLIMETER_TO_METER, np.float32)
            )
        else:
            img = img / 255.0
            max_size = 1008 // PATCH_SIZE * PATCH_SIZE  # 1008, already multiple of 14

            if max(img.shape[:2]) > max_size:
                h, w = img.shape[:2]
                scale = max_size / max(h, w)
                tar_h = ensure_multiple_of(h * scale)
                tar_w = ensure_multiple_of(w * scale)
                img = cv2.resize(img, (tar_w, tar_h), interpolation=cv2.INTER_AREA)

            img = image_hwc_to_chw(np.asarray(img, np.float32))  # (3, H, W)
        return img

    # ── Bounding box loading ──────────────────────────────────────────────

    @staticmethod
    def load_bounding_box(box_path: str) -> np.ndarray:
        """
        Load YOLOv8 boxes from JSON.
        Returns: float32 array (N, 4) in xyxy, image-pixel coordinates.
        """
        if not os.path.exists(box_path):
            return np.zeros((0, 4), dtype=np.float32)

        with open(box_path, "r") as f:
            data = json.load(f)

        boxes = data.get("boxes_xyxy_feature", [])
        if not boxes:
            return np.zeros((0, 4), dtype=np.float32)

        arr = np.array(boxes, dtype=np.float32)
        assert arr.ndim == 2 and arr.shape[1] == 4, \
            f"Expected (N,4) boxes, got shape {arr.shape}"
        return arr

    @staticmethod
    def scale_boxes_to_feature_space(
        boxes: np.ndarray,
        img_h: int,
        img_w: int,
        patch_size: int = PATCH_SIZE,
    ) -> np.ndarray:
        """
        Scale boxes từ image pixel space → DINOv2 feature map space.

        image:        (img_h, img_w)
        feature map:  (img_h // patch_size,  img_w // patch_size)
        """
        if boxes.shape[0] == 0:
            return boxes

        feat_h = img_h // patch_size
        feat_w = img_w // patch_size
        scale_x = feat_w / img_w
        scale_y = feat_h / img_h

        scaled = boxes.copy()
        scaled[:, [0, 2]] *= scale_x  # x1, x2
        scaled[:, [1, 3]] *= scale_y  # y1, y2
        return scaled

    # ── __getitem__ ───────────────────────────────────────────────────────

    def __getitem__(self, index: int):
        video_id, sample_id, direction = self.samples[index]
        sample = {dataset_keys.IDENTIFIER: str(sample_id)}

        # ── File paths ────────────────────────────────────────────────────
        rgb_file   = os.path.join(self.dataset_folder, video_id, WIDE, sample_id)
        depth_file = os.path.join(self.dataset_folder, video_id, 'highres_depth', sample_id)
        apple_file = os.path.join(self.dataset_folder, video_id, 'lowres_depth', sample_id)
        box_file   = os.path.join(
            self.dataset_folder, video_id, 'boxes',
            sample_id.replace('.png', '.json')
        )

        # ── Load images ───────────────────────────────────────────────────
        color_img = self.load_image(rgb_file, self.high_res, False, direction)
        sample[dataset_keys.COLOR_IMG]          = color_img           # (3, H, W)
        sample[dataset_keys.HIGH_RES_DEPTH_IMG] = self.load_image(depth_file, self.high_res, True, direction)
        sample[dataset_keys.LOW_RES_DEPTH_IMG]  = self.load_image(apple_file, self.low_res,  True, direction)

        # ── Load + scale boxes → feature map space ────────────────────────
        _, img_h, img_w = color_img.shape
        boxes_px = self.load_bounding_box(box_file)                   # (N, 4) pixel
        boxes_feat = self.scale_boxes_to_feature_space(               # (N, 4) feat
            boxes_px, img_h, img_w
        )
        sample[dataset_keys.BOUNDING_BOX] = boxes_feat

        if self.transform is not None:
            sample[dataset_keys.COLOR_IMG] = self.transform(sample[dataset_keys.COLOR_IMG])

        return sample
    
def collate_fn(batch):
    import torch
    keys_to_stack = [
        dataset_keys.COLOR_IMG,
        dataset_keys.HIGH_RES_DEPTH_IMG,
        dataset_keys.LOW_RES_DEPTH_IMG,
    ]
    result = {}

    for k in keys_to_stack:
        result[k] = torch.from_numpy(
            np.stack([s[k] for s in batch], axis=0)
        ).float()

    # Boxes: list of Tensor (N_i, 4) — N_i khác nhau
    result[dataset_keys.BOUNDING_BOX] = [
        torch.from_numpy(s[dataset_keys.BOUNDING_BOX]).float()
        for s in batch
    ]

    result[dataset_keys.IDENTIFIER] = [s[dataset_keys.IDENTIFIER] for s in batch]
    return result