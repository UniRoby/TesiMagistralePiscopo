"""Tests for memory-bounded patch inference."""

from __future__ import annotations

import unittest

import numpy as np

from tesi_m3d.inference import infer_heatmap, predict_patch_scores
from tesi_m3d.model import Patch3DModelConfig, build_patch3d_classifier
from tesi_m3d.patches import PatchGrid


class InferenceTests(unittest.TestCase):
    def test_streamed_scores_and_heatmap_cover_the_full_volume(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        model = build_patch3d_classifier(Patch3DModelConfig(base_channels=4))
        volume = np.zeros((64, 64, 64), dtype=np.float32)
        grid = PatchGrid(volume.shape, patch_shape=(32, 32, 32), stride=(16, 16, 16))
        scores = predict_patch_scores(model, volume, grid, batch_size=5, device="cpu")
        heatmap = infer_heatmap(
            model, volume, patch_shape=(32, 32, 32), stride=(16, 16, 16), batch_size=5
        )

        self.assertEqual(scores.shape, (27,))
        self.assertEqual(heatmap.shape, volume.shape)
        self.assertTrue(np.isfinite(heatmap).all())


if __name__ == "__main__":
    unittest.main()
