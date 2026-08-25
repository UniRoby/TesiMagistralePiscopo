"""Tests for defensive CT-GAN trace measurements."""

import unittest

import numpy as np

from tesi_m3d.audit_ctgan_traces import residual_region_stats, select_audit_records
from tesi_m3d.dataset import M3DSynthRecord
from tesi_m3d.evaluate_unet import auc_ap_from_histograms, best_histogram_threshold, score_histograms


class CTGANTraceAuditTests(unittest.TestCase):
    def test_balanced_selection_and_residual_regions(self) -> None:
        records = [
            M3DSynthRecord(f"{ty}_{i}", ty, "pix2pix", f"P{ty}{i}", "1", 4, 4, 4, "train")
            for ty in ("inj", "rem") for i in range(4)
        ]
        selected = select_audit_records(records, "pix2pix", 6)
        self.assertEqual(sum(record.ty == "inj" for record in selected), 3)
        self.assertEqual(sum(record.ty == "rem" for record in selected), 3)

        pristine = np.zeros((9, 9, 9), dtype=np.uint16)
        manipulated = pristine.copy()
        mask = np.zeros_like(pristine, dtype=bool)
        mask[3:6, 3:6, 3:6] = True
        manipulated[mask] = 10
        stats = residual_region_stats(pristine, manipulated, mask)
        self.assertEqual(stats["inside"]["changed_fraction"], 1.0)
        self.assertEqual(stats["background"]["changed_fraction"], 0.0)

    def test_histogram_metrics_are_perfect_for_separated_scores(self) -> None:
        mask = np.array([0, 0, 1, 1], dtype=bool)
        scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
        positive, negative = score_histograms(mask, scores, bins=10)
        _, threshold, dice = best_histogram_threshold(positive, negative)
        auc, ap = auc_ap_from_histograms(positive, negative)
        self.assertGreaterEqual(threshold, 0.3)
        self.assertEqual(dice, 1.0)
        self.assertEqual(auc, 1.0)
        self.assertEqual(ap, 1.0)


if __name__ == "__main__":
    unittest.main()
