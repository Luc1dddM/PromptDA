"""ScanNet++ iPhone prompts paired with rendered supervision, in metres."""
from pathlib import Path
from dataset.experiment import scene_subset
import warnings

import cv2
import numpy as np
from torch.utils.data import Dataset

from dataset import dataset_keys as keys


class ScanNetPPDataset(Dataset):
    def __init__(self, root, split='train', max_samples=None, seed=42,
                 transform=None, target_hw=(756, 1008), depth_scale=1000.0,
                 use_rgb_masks=True):
        self.root = Path(root)
        self.transform = transform
        self.target_hw = tuple(target_hw)
        self.depth_scale = float(depth_scale)
        self.use_rgb_masks = use_rgb_masks
        if self.depth_scale <= 0 or not np.isfinite(self.depth_scale):
            raise ValueError('depth_scale must be finite and positive')
        if len(self.target_hw) != 2 or any(n <= 0 or n % 14 for n in self.target_hw):
            raise ValueError('target_hw must contain positive multiples of 14')
        if max_samples is not None and max_samples <= 0:
            raise ValueError('max_samples must be positive')
        if split not in ('train', 'val'):
            raise ValueError('split must be train or val')
        split_path = self.root / 'splits' / f'{split}.txt'
        scenes = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
        if len(scenes) != len(set(scenes)):
            raise ValueError(f'Duplicate scenes in {split_path}')
        other_path = self.root / 'splits' / ('val.txt' if split == 'train' else 'train.txt')
        if other_path.exists() and set(scenes) & set(other_path.read_text().split()):
            raise ValueError('Train and validation scene splits overlap')
        self.samples = []
        self.skipped_scenes = []
        for scene in scenes:
            if Path(scene).name != scene:
                raise ValueError(f'Expected a scene ID in {split_path}: {scene}')
            folder = self.root / 'data' / scene / 'iphone'
            targets = sorted((folder / 'render_depth').glob('*.png'))
            if not targets:
                self.skipped_scenes.append(scene)
                continue
            rgb_files = {p.name for p in (folder / 'rgb').iterdir()} if (folder / 'rgb').is_dir() else set()
            prompt_files = {p.name for p in (folder / 'depth').iterdir()} if (folder / 'depth').is_dir() else set()
            mask_files = ({p.name for p in (folder / 'rgb_masks').iterdir()}
                          if use_rgb_masks and (folder / 'rgb_masks').is_dir() else set())
            for target in targets:
                stem = target.stem
                rgb = folder / 'rgb' / f'{stem}.jpg'
                if rgb.name not in rgb_files:
                    rgb = folder / 'rgb' / f'{stem}.png'
                prompt = folder / 'depth' / f'{stem}.png'
                mask = folder / 'rgb_masks' / f'{stem}.png'
                required_assets = [(rgb, rgb_files), (prompt, prompt_files)]
                if use_rgb_masks:
                    required_assets.append((mask, mask_files))
                for required, available in required_assets:
                    if required.name not in available:
                        raise FileNotFoundError(f'Missing paired asset: {required}')
                self.samples.append((f'{scene}/{stem}', rgb, prompt, target, mask))
        if self.skipped_scenes:
            warnings.warn(f"Skipping {len(self.skipped_scenes)} scenes without rendered targets: {self.skipped_scenes}", stacklevel=2)
        if not self.samples:
            raise ValueError(f'No paired frames for {split_path}')
        if max_samples is not None and len(self.samples) > max_samples:
            self.samples = scene_subset(self.samples, max_samples, seed)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def read(path):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f'Cannot read image: {path}')
        return image

    def __getitem__(self, index):
        identifier, rgb_path, prompt_path, target_path, mask_path = self.samples[index]
        rgb = self.read(rgb_path)
        target = self.read(target_path)
        prompt = self.read(prompt_path)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or target.ndim != 2 or prompt.ndim != 2:
            raise ValueError(f'Expected RGB and single-channel depth: {identifier}')
        if target.shape != rgb.shape[:2]:
            raise ValueError(f'RGB and rendered depth are not aligned: {identifier}')
        target = target.astype(np.float32) / self.depth_scale
        prompt = prompt.astype(np.float32) / self.depth_scale
        target[~np.isfinite(target) | (target <= 0.01)] = 0
        prompt[~np.isfinite(prompt) | (prompt <= 0)] = 0
        if self.use_rgb_masks:
            mask = self.read(mask_path)
            if mask.shape != target.shape:
                raise ValueError(f'RGB mask is not aligned: {identifier}')
            # ScanNet++ masks: zero excludes pixels; nonzero keeps pixels.
            target[mask == 0] = 0
            prompt_mask = cv2.resize(mask, (prompt.shape[1], prompt.shape[0]), interpolation=cv2.INTER_NEAREST)
            prompt[prompt_mask == 0] = 0
        height, width = self.target_hw
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        target = cv2.resize(target, (width, height), interpolation=cv2.INTER_NEAREST)
        sample = {
            keys.IDENTIFIER: identifier,
            keys.COLOR_IMG: np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0,
            keys.HIGH_RES_DEPTH_IMG: target[None],
            keys.LOW_RES_DEPTH_IMG: prompt[None],
        }
        if self.transform is not None:
            sample = self.transform(sample)
        sample[keys.VALID_MASK_IMG] = sample[keys.HIGH_RES_DEPTH_IMG] > 0
        return sample
