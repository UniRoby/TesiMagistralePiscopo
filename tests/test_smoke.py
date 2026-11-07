import unittest

from tesi_m3d.model import Patch3DModelConfig


class SmokeTests(unittest.TestCase):
    def test_import_and_model_config(self):
        config = Patch3DModelConfig(in_channels=1, num_classes=1)

        self.assertEqual(config.in_channels, 1)
        self.assertEqual(config.num_classes, 1)
        self.assertEqual(config.name, "resnet18_medicalnet")
        self.assertTrue(config.medicalnet_weights.endswith("resnet_18_23dataset.pth"))


if __name__ == "__main__":
    unittest.main()
