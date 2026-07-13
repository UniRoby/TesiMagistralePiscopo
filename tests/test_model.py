import unittest

from tesi_m3d.model import Patch3DModelConfig, TorchDependencyError, build_patch3d_classifier


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


if __name__ == "__main__":
    unittest.main()
