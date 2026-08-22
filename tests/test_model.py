import unittest

from tesi_m3d.losses import SegmentationBCEDiceLoss
from tesi_m3d.model import (
    Patch3DModelConfig,
    TorchDependencyError,
    UNet3DModelConfig,
    build_patch3d_classifier,
    build_unet3d,
)


class ModelTests(unittest.TestCase):
    def test_default_model_outputs_single_logit(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")

        model = build_patch3d_classifier(Patch3DModelConfig(base_channels=4))
        x = torch.zeros((2, 1, 32, 32, 32), dtype=torch.float32)
        y = model(x)

        self.assertEqual(tuple(y.shape), (2, 1))

    def test_model_build_or_clear_dependency_error(self):
        try:
            model = build_patch3d_classifier(Patch3DModelConfig(base_channels=4))
        except TorchDependencyError as exc:
            self.assertIn("PyTorch is required", str(exc))
        else:
            self.assertTrue(hasattr(model, "forward"))

    def test_unet_outputs_voxel_probabilities_and_loss(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")

        model = build_unet3d(UNet3DModelConfig(base_channels=2))
        image = torch.zeros((1, 1, 16, 16, 16), dtype=torch.float32)
        target = torch.zeros_like(image)
        probabilities = model(image)

        self.assertEqual(tuple(probabilities.shape), tuple(target.shape))
        self.assertTrue(torch.all((probabilities >= 0) & (probabilities <= 1)))
        self.assertTrue(torch.isfinite(SegmentationBCEDiceLoss()(probabilities, target)))


if __name__ == "__main__":
    unittest.main()
