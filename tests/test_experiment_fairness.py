import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset.dataset import MyARKitScenesDataset
from dataset.experiment import scene_subset
from training.metrics import compute_depth_metrics, compute_depth_metrics_per_image, aggregate_metrics


class FairnessTests(unittest.TestCase):
    def test_arkit_rgb_and_native_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'metadata.csv').write_text('video_id,sky_direction,fold\n1,Up,Training\n')
            for folder in ('wide', 'lowres_depth', 'highres_depth'):
                (root / 'Training/1' / folder).mkdir(parents=True)
            rgb = np.zeros((28, 42, 3), dtype=np.uint8)
            rgb[..., 2] = 255
            for folder, value in [('wide', rgb), ('lowres_depth', np.full((4, 6), 2000, np.uint16)),
                                  ('highres_depth', np.full((28, 42), 3000, np.uint16))]:
                cv2.imwrite(str(root / 'Training/1' / folder / 'frame.png'), value)
            sample = MyARKitScenesDataset(str(root), target_hw=(14, 28))[0]
            from dataset import dataset_keys as keys
            np.testing.assert_equal(sample[keys.COLOR_IMG][0], 1)
            np.testing.assert_equal(sample[keys.COLOR_IMG][2], 0)
            self.assertEqual(sample[keys.LOW_RES_DEPTH_IMG].shape, (1, 4, 6))
            np.testing.assert_equal(sample[keys.LOW_RES_DEPTH_IMG], 2)
            self.assertEqual(sample[keys.IDENTIFIER], '1/frame.png')
            ds = MyARKitScenesDataset(str(root), target_hw=(14, 28))
            ds.samples[0][2] = 'Left'
            self.assertEqual(ds[0][keys.LOW_RES_DEPTH_IMG].shape, (1, 4, 6))

    def test_scene_coverage_and_reproducibility(self):
        samples = [(f'{scene}/frame_{i:06d}',) for scene, count in [('a', 100), ('b', 10), ('c', 2)] for i in range(count)]
        selected = scene_subset(samples, 18, 26)
        self.assertEqual(selected, scene_subset(list(reversed(samples)), 18, 26))
        self.assertEqual(len(set(selected)), 18)
        self.assertEqual([sum(s[0].startswith(c + '/') for s in selected) for c in 'abc'], [8, 8, 2])

    def test_metrics_batch_independence_and_delta(self):
        gt = torch.ones(3, 1, 8, 8)
        pred = gt * torch.tensor([1.05, 1.2, 2.0]).view(3, 1, 1, 1)
        gt[1, :, :4] = 0
        original_gt, original_pred = gt.clone(), pred.clone()
        one = aggregate_metrics(compute_depth_metrics_per_image(pred, gt))
        split = aggregate_metrics(compute_depth_metrics_per_image(pred[:2], gt[:2]) + compute_depth_metrics_per_image(pred[2:], gt[2:]))
        self.assertEqual(one, split)
        self.assertAlmostEqual(one['delta0.5'], 1 / 3)
        self.assertAlmostEqual(one['delta1'], 2 / 3)
        self.assertAlmostEqual(one['L1'], (0.05 + 0.2 + 1) / 3, places=6)
        torch.testing.assert_close(gt, original_gt)
        torch.testing.assert_close(pred, original_pred)

    def test_valid_mask_and_nonfinite_predictions(self):
        gt = torch.ones(1, 1, 8, 8)
        pred = gt.clone()
        pred[..., 0, 0] = 50
        mask = torch.ones_like(gt, dtype=torch.bool)
        mask[..., 0, 0] = False
        self.assertEqual(compute_depth_metrics(pred, gt, valid_mask=mask)['L1'], 0)
        pred[..., 1, 1] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            compute_depth_metrics(pred, gt)


if __name__ == '__main__':
    unittest.main()
