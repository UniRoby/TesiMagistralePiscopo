import unittest

import numpy as np

from tesi_m3d.evaluation import (
    _max_balanced_accuracy_from_roc,
    balanced_accuracy,
    best_heatmap_threshold_by_f1,
    best_threshold_by_balanced_accuracy,
    binary_localization_metrics,
    topk_detection_score,
    volume_detection_scores,
    volume_auc_ba,
    voxel_auc_ap,
    voxel_auc_max_balanced_accuracy,
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

    def test_per_volume_auc_and_max_ba(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn not installed")
        mask = np.array([0, 0, 1, 1], dtype=bool)

        auc, max_ba, threshold = voxel_auc_max_balanced_accuracy(
            mask, np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
        )

        self.assertAlmostEqual(auc, 1.0)
        self.assertAlmostEqual(max_ba, 1.0)
        self.assertAlmostEqual(threshold, 0.8)

    def test_max_ba_from_roc_prefers_highest_tied_threshold(self):
        max_ba, threshold = _max_balanced_accuracy_from_roc(
            np.array([0.0, 0.0, 0.5, 1.0]),
            np.array([0.0, 0.5, 1.0, 1.0]),
            np.array([np.inf, 0.8, 0.4, 0.1]),
        )

        self.assertAlmostEqual(max_ba, 0.75)
        self.assertAlmostEqual(threshold, 0.8)

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

    def test_topk_volume_score_ignores_single_voxel_spike(self):
        clean = np.zeros((10, 10), dtype=np.float32)
        clean[0, 0] = 1.0
        coherent = np.zeros((10, 10), dtype=np.float32)
        coherent[:2, :5] = 0.8

        max_scores = volume_detection_scores([clean, coherent], mode="max")
        topk_scores = volume_detection_scores([clean, coherent], mode="topk_mean", topk_fraction=0.1)

        self.assertGreater(max_scores[0], max_scores[1])
        self.assertLess(topk_scores[0], topk_scores[1])

    def test_calibrates_highest_balanced_accuracy_tie_threshold(self):
        truth = np.array([0, 1, 0, 1], dtype=bool)
        scores = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)

        threshold, value = best_threshold_by_balanced_accuracy(truth, scores)

        self.assertAlmostEqual(threshold, 0.8)
        self.assertAlmostEqual(value, 0.75)

    def test_calibrates_micro_f1_for_heatmaps(self):
        mask = np.array([1, 1, 0, 0], dtype=bool)
        heatmap = np.array([0.9, 0.8, 0.7, 0.1], dtype=np.float32)

        threshold, metrics = best_heatmap_threshold_by_f1([mask], [heatmap], thresholds=[0.5, 0.8, 0.9])

        self.assertAlmostEqual(threshold, 0.8)
        self.assertAlmostEqual(metrics.f1, 1.0)


if __name__ == "__main__":
    unittest.main()
