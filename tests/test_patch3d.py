import unittest

import numpy as np

from tesi_m3d.evaluation import f1_iou, threshold_heatmap
from tesi_m3d.patches import (
    PatchGrid,
    binary_cube_mask,
    extract_patches,
    gaussian_patch_weights,
    labels_from_mask,
    patch_label,
    reconstruct_heatmap,
    soft_scores_from_mask,
)


class Patch3DTests(unittest.TestCase):
    def test_extract_patches_regular_grid(self):
        volume = np.arange(8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
        grid = PatchGrid(volume_shape=volume.shape, patch_shape=(4, 4, 4), stride=(2, 2, 2))

        patches = extract_patches(volume, grid)

        self.assertEqual(patches.shape, (27, 4, 4, 4))
        np.testing.assert_array_equal(patches[0], volume[:4, :4, :4])
        np.testing.assert_array_equal(patches[-1], volume[4:8, 4:8, 4:8])

    def test_synthetic_64_cube_produces_27_patches(self):
        volume = np.zeros((64, 64, 64), dtype=np.float32)
        mask = binary_cube_mask((64, 64, 64), start=(16, 16, 16), size=(32, 32, 32))
        grid = PatchGrid(volume_shape=volume.shape, patch_shape=(32, 32, 32), stride=(16, 16, 16))

        patches = extract_patches(volume, grid)
        labels = labels_from_mask(mask, grid, positive_overlap_fraction=0.05)
        soft_scores = soft_scores_from_mask(mask, grid)

        self.assertEqual(len(grid), 27)
        self.assertEqual(patches.shape, (27, 32, 32, 32))
        # With a centered 32^3 cube and stride 16, every patch overlaps the cube by at least 12.5%.
        self.assertEqual(labels.count(1), 27)
        self.assertEqual(labels.count(0), 0)
        self.assertEqual(float(soft_scores.max()), 1.0)

    def test_patch_label_thresholds(self):
        empty = np.zeros((10, 10, 10), dtype=bool)
        tiny = empty.copy()
        tiny[:1, :1, :1] = True
        threshold = empty.copy()
        threshold[:5, :10, :10] = True

        self.assertEqual(patch_label(empty, positive_overlap_fraction=0.05), 0)
        self.assertIsNone(patch_label(tiny, positive_overlap_fraction=0.05))
        self.assertEqual(patch_label(threshold, positive_overlap_fraction=0.05), 1)

    def test_reconstruct_heatmap_average_and_gaussian(self):
        mask = binary_cube_mask((64, 64, 64), start=(16, 16, 16), size=(32, 32, 32))
        grid = PatchGrid(volume_shape=mask.shape, patch_shape=(32, 32, 32), stride=(16, 16, 16))
        scores = soft_scores_from_mask(mask, grid)

        average = reconstruct_heatmap(scores, grid, mode="average")
        gaussian = reconstruct_heatmap(scores, grid, mode="gaussian")

        self.assertEqual(average.shape, (64, 64, 64))
        self.assertEqual(gaussian.shape, (64, 64, 64))
        self.assertGreater(average[32, 32, 32], average[0, 0, 0])
        self.assertGreater(gaussian[32, 32, 32], gaussian[0, 0, 0])

    def test_gaussian_weights_peak_at_center(self):
        weights = gaussian_patch_weights((9, 9, 9))

        self.assertEqual(weights.shape, (9, 9, 9))
        self.assertAlmostEqual(float(weights[4, 4, 4]), 1.0, places=6)
        self.assertLess(float(weights[0, 0, 0]), float(weights[4, 4, 4]))

    def test_labels_and_metrics_on_synthetic_cube(self):
        mask = binary_cube_mask((16, 16, 16), start=(6, 6, 6), size=(4, 4, 4))
        grid = PatchGrid(volume_shape=mask.shape, patch_shape=(8, 8, 8), stride=(4, 4, 4))
        labels = labels_from_mask(mask, grid, positive_overlap_fraction=0.05)
        scores = np.array([1.0 if label == 1 else 0.0 for label in labels], dtype=np.float32)

        heatmap = reconstruct_heatmap(scores, grid)
        prediction = threshold_heatmap(heatmap, threshold=0.5)
        f1, iou = f1_iou(mask, prediction)

        self.assertGreater(f1, 0.0)
        self.assertGreater(iou, 0.0)
        self.assertGreater(heatmap[7, 7, 7], heatmap[0, 0, 0])


if __name__ == "__main__":
    unittest.main()
