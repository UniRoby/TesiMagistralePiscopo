import unittest

import numpy as np

from tesi_m3d.evaluation import (
    balanced_accuracy,
    binary_localization_metrics,
    topk_detection_score,
    volume_auc_ba,
    voxel_auc_ap,
)
from tesi_m3d.patches import binary_cube_mask


class EvaluationTests(unittest.TestCase):
    def test_binary_localization_metrics(self):
        mask = np.array([1, 1, 0, 0], dtype=bool)
        pred = np.array([1, 0, 1, 0], dtype=bool)

        metrics = binary_localization_metrics(mask, pred)

        self.assertAlmostEqual(metrics.precision, 0.5)
        self.assertAlmostEqual(metrics.recall, 0.5)
        self.assertAlmostEqual(metrics.f1, 0.5)
        self.assertAlmostEqual(metrics.iou, 1 / 3)

    def test_voxel_auc_ap_on_synthetic_heatmap(self):
        mask = binary_cube_mask((8, 8, 8), start=(2, 2, 2), size=(3, 3, 3))
        heatmap = mask.astype(np.float32)

        auc, ap = voxel_auc_ap(mask, heatmap)

        self.assertAlmostEqual(auc, 1.0)
        self.assertAlmostEqual(ap, 1.0)

    def test_volume_scores_and_balanced_accuracy(self):
        y_true = np.array([0, 0, 1, 1], dtype=bool)
        y_score = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)

        auc, ba = volume_auc_ba(y_true, y_score, threshold=0.5)

        self.assertAlmostEqual(auc, 1.0)
        self.assertAlmostEqual(ba, 1.0)
        self.assertAlmostEqual(balanced_accuracy(y_true, y_score, threshold=0.5), 1.0)

    def test_topk_detection_score(self):
        heatmap = np.arange(100, dtype=np.float32).reshape(10, 10)

        score = topk_detection_score(heatmap, fraction=0.1)

        self.assertAlmostEqual(score, float(np.mean(np.arange(90, 100))))


if __name__ == "__main__":
    unittest.main()
