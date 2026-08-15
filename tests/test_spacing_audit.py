import unittest

from tesi_m3d.audit_spacing import physical_patch_geometry, summarize_spacing_rows


class SpacingAuditTests(unittest.TestCase):
    def test_physical_patch_geometry_respects_zyx_spacing(self):
        patch_mm, target_vox = physical_patch_geometry((2.0, 0.5, 0.5), (32, 32, 32), 32.0)

        self.assertEqual(patch_mm, (64.0, 16.0, 16.0))
        self.assertEqual(target_vox, (16.0, 64.0, 64.0))

    def test_summary_groups_split_and_generator(self):
        base = {
            "spacing_z": 2.0, "spacing_y": 0.5, "spacing_x": 0.5,
            "patch_mm_z": 64.0, "patch_mm_y": 16.0, "patch_mm_x": 16.0,
            "target_vox_z": 16.0, "target_vox_y": 64.0, "target_vox_x": 64.0,
        }
        rows = [
            {**base, "split": "train", "generator": "pix2pix"},
            {**base, "split": "valid", "generator": "cycle"},
        ]

        summary = summarize_spacing_rows(rows)

        self.assertEqual(summary["overall"]["n_records"], 2)
        self.assertEqual(set(summary["by_split_generator"]), {"train:pix2pix", "valid:cycle"})


if __name__ == "__main__":
    unittest.main()
