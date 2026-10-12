import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from dataset.scannetpp import ScanNetPPDataset
from dataset import dataset_keys as keys


class ScanNetPPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'splits').mkdir()
        (self.root / 'splits/train.txt').write_text('scene1\n')
        (self.root / 'splits/val.txt').write_text('scene2\n')
        self.folder = self.root / 'data/scene1/iphone'
        for name in ('rgb', 'depth', 'render_depth', 'rgb_masks'):
            (self.folder / name).mkdir(parents=True)
        for index in range(3):
            stem = f'frame_{index:06d}'
            rgb = np.zeros((28, 42, 3), np.uint8)
            rgb[:, :, 2] = 255
            cv2.imwrite(str(self.folder / 'rgb' / f'{stem}.png'), rgb)
            cv2.imwrite(str(self.folder / 'depth' / f'{stem}.png'), np.full((4, 6), 2000, np.uint16))
            cv2.imwrite(str(self.folder / 'render_depth' / f'{stem}.png'), np.full((28, 42), 3000, np.uint16))
            mask = np.full((28, 42), 255, np.uint8)
            mask[:, :21] = 0
            cv2.imwrite(str(self.folder / 'rgb_masks' / f'{stem}.png'), mask)

    def test_units_color_masks_and_native_prompt(self):
        sample = ScanNetPPDataset(self.root, target_hw=(14, 28))[0]
        self.assertEqual(sample[keys.COLOR_IMG].shape, (3, 14, 28))
        self.assertEqual(sample[keys.LOW_RES_DEPTH_IMG].shape, (1, 4, 6))
        np.testing.assert_equal(sample[keys.COLOR_IMG][0], 1)
        np.testing.assert_equal(sample[keys.COLOR_IMG][2], 0)
        np.testing.assert_equal(sample[keys.HIGH_RES_DEPTH_IMG][:, :, :14], 0)
        np.testing.assert_equal(sample[keys.HIGH_RES_DEPTH_IMG][:, :, 14:], 3)
        np.testing.assert_equal(sample[keys.LOW_RES_DEPTH_IMG][:, :, :3], 0)
        np.testing.assert_equal(sample[keys.LOW_RES_DEPTH_IMG][:, :, 3:], 2)
        self.assertEqual(sample[keys.IDENTIFIER], 'scene1/frame_000000')

    def test_sampling_and_masks_disabled(self):
        a = ScanNetPPDataset(self.root, max_samples=2, seed=7, use_rgb_masks=False)
        b = ScanNetPPDataset(self.root, max_samples=2, seed=7)
        self.assertEqual(a.samples, b.samples)
        np.testing.assert_equal(a[0][keys.HIGH_RES_DEPTH_IMG], 3)

    def test_split_leakage(self):
        (self.root / 'splits/val.txt').write_text('scene1\n')
        with self.assertRaisesRegex(ValueError, 'overlap'):
            ScanNetPPDataset(self.root)

    def test_missing_pair_fails(self):
        (self.folder / 'depth/frame_000000.png').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'paired asset'):
            ScanNetPPDataset(self.root)

    def test_missing_render_scene_warns(self):
        (self.root / 'splits/train.txt').write_text('scene1\nempty\n')
        with self.assertWarnsRegex(UserWarning, 'without rendered targets'):
            ds = ScanNetPPDataset(self.root)
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds.skipped_scenes, ['empty'])


if __name__ == '__main__':
    unittest.main()
