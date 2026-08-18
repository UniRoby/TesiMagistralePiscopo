import unittest

import numpy as np

from tesi_m3d.isotropic import resample_volume, scale_coordinate


class IsotropicResamplingTests(unittest.TestCase):
    def test_scan_and_mask_share_target_geometry(self):
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("scipy not installed")
        scan = np.arange(4 * 6 * 8, dtype=np.uint16).reshape(4, 6, 8)
        mask = scan > 100
        spacing = (2.0, 0.5, 0.5)

        resampled_scan = resample_volume(scan, spacing, 1.0, order=1)
        resampled_mask = resample_volume(mask, spacing, 1.0, order=0)

        self.assertEqual(resampled_scan.shape, (8, 3, 4))
        self.assertEqual(resampled_mask.shape, resampled_scan.shape)
        self.assertEqual(resampled_scan.dtype, np.uint16)
        self.assertEqual(resampled_mask.dtype, np.bool_)
        self.assertEqual(scale_coordinate("3", spacing[0], 1.0), "6")


if __name__ == "__main__":
    unittest.main()
