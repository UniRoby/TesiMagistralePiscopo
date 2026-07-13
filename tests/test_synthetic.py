import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tesi_m3d.synthetic import SyntheticPatchDataset, make_synthetic_volume, make_synthetic_volumes


class SyntheticDataTests(unittest.TestCase):
    def test_make_synthetic_volume_shapes(self):
        item = make_synthetic_volume(shape=(32, 32, 32), cube_start=(8, 8, 8), cube_size=(8, 8, 8))

        self.assertEqual(item.volume.shape, (32, 32, 32))
        self.assertEqual(item.mask.shape, (32, 32, 32))
        self.assertEqual(item.volume.dtype, np.float32)
        self.assertGreater(item.mask.sum(), 0)
        self.assertGreater(float(item.volume[item.mask].mean()), float(item.volume[~item.mask].mean()))

    def test_synthetic_patch_dataset_returns_tensors(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed")

        volumes = make_synthetic_volumes(generators=("pix2pix",), volumes_per_generator=1)
        dataset = SyntheticPatchDataset(volumes, patch_shape=(32, 32, 32), stride=(16, 16, 16))
        sample = dataset[0]

        self.assertGreater(len(dataset), 0)
        self.assertEqual(tuple(sample["image"].shape), (1, 32, 32, 32))
        self.assertEqual(tuple(sample["label"].shape), (1,))

    def test_smoke_synthetic_training_script_creates_checkpoint(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed")

        project = Path(__file__).resolve().parents[1]
        script = project / "scripts" / "smoke_synthetic_training.py"
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=tmp,
                check=True,
                text=True,
                capture_output=True,
            )
            checkpoint = Path(tmp) / "outputs" / "synthetic_smoke" / "synthetic_patch3d.pt"
            heatmap = Path(tmp) / "outputs" / "synthetic_smoke" / "synthetic_heatmap.npy"
            self.assertIn("train_loss", result.stdout)
            self.assertTrue(checkpoint.exists())
            self.assertTrue(heatmap.exists())


if __name__ == "__main__":
    unittest.main()
