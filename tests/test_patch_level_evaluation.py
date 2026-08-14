import numpy as np

from tesi_m3d.evaluate_patch_level import (
    best_f1_threshold,
    binary_counts,
    patch_center,
    patch_overlap_fractions,
    topk_hits,
)
from tesi_m3d.patches import PatchGrid
from tesi_m3d.mine_hard_negatives import select_hard_negative_indices


def test_patch_level_ranking_and_threshold_metrics():
    truth = np.array([0, 1, 0, 1], dtype=bool)
    scores = np.array([0.1, 0.9, 0.8, 0.7], dtype=np.float32)

    threshold, metrics = best_f1_threshold(truth, scores)

    assert np.isclose(threshold, 0.7)
    assert np.isclose(metrics["f1"], 0.8)
    assert binary_counts(truth, scores, threshold) == {"tp": 2, "fp": 1, "tn": 1, "fn": 0}


def test_topk_hit_uses_any_mask_intersection():
    overlaps = np.array([0.0, 0.001, 0.0, 0.2])
    scores = np.array([0.9, 0.8, 0.7, 0.6])

    assert topk_hits(overlaps, scores) == {1: False, 3: True, 5: True}
    np.testing.assert_allclose(patch_center((slice(0, 32), slice(16, 48), slice(4, 36))), [15.5, 31.5, 19.5])


def test_patch_overlap_fractions_match_patch_volume():
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[:2, :2, :2] = True
    grid = PatchGrid((4, 4, 4), patch_shape=(2, 2, 2), stride=(2, 2, 2))

    np.testing.assert_allclose(patch_overlap_fractions(mask, grid), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_hard_negative_mining_keeps_only_clean_highest_scores():
    scores = np.array([0.2, 0.95, 0.8, 0.9, 0.7], dtype=np.float32)
    overlaps = np.array([0.0, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)

    np.testing.assert_array_equal(select_hard_negative_indices(scores, overlaps, 3), [3, 2, 4])
