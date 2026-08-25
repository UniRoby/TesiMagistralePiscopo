import unittest

from tesi_m3d.losses import SegmentationBCEDiceLoss, SegmentationFocalDiceLoss
from tesi_m3d.model import (
    Patch3DModelConfig,
    TorchDependencyError,
    UNet3DModelConfig,
    build_patch3d_classifier,
    build_unet3d,
    highpass_residual3d,
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

    def test_highpass_is_zero_on_constant_input_and_preserves_gradients(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")

        image = torch.ones((1, 1, 8, 8, 8), requires_grad=True)
        residual = highpass_residual3d(image)
        self.assertEqual(tuple(residual.shape), tuple(image.shape))
        self.assertTrue(torch.allclose(residual, torch.zeros_like(residual)))
        residual.sum().backward()
        self.assertIsNotNone(image.grad)

    def test_highpass_unet_and_focal_dice(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")

        model = build_unet3d(UNet3DModelConfig(base_channels=2, input_mode="ct_highpass"))
        image = torch.rand((1, 1, 16, 16, 16))
        target = torch.zeros_like(image)
        target[:, :, 6:10, 6:10, 6:10] = 1
        probabilities = model(image)
        loss = SegmentationFocalDiceLoss()(probabilities, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
