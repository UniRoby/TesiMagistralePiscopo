import unittest

from tesi_m3d.dataset import (
    M3DSynthRecord,
    assert_no_orig_id_leakage,
    filter_records,
    cross_generator_records,
    leave_generator_out_records,
)


def record(img_id, mod, orig_id, split):
    return M3DSynthRecord(
        img_id=img_id,
        ty="inject",
        mod=mod,
        orig_id=orig_id,
        sdir_id="0",
        coord_z=0,
        coord_y=0,
        coord_x=0,
        split=split,
    )


class DatasetSplitTests(unittest.TestCase):
    def test_filter_records_keeps_real_when_requested(self):
        records = [
            record("a", "pix2pix", "orig-a", "train"),
            record("b", "real", "orig-b", "train"),
            record("c", "cycle", "orig-c", "test"),
        ]

        filtered = filter_records(records, mods=["pix2pix"], splits=["train"], include_real=True)

        self.assertEqual([item.img_id for item in filtered], ["a", "b"])


    def test_cross_generator_train_one_tests_two(self):
        records = [
            record("a", "pix2pix", "orig-a", "train"),
            record("b", "cycle", "orig-b", "train"),
            record("c", "diffusion", "orig-c", "train"),
            record("d", "real", "orig-d", "train"),
            record("e", "pix2pix", "orig-e", "valid"),
            record("f", "real", "orig-f", "valid"),
            record("g", "cycle", "orig-g", "test"),
            record("h", "diffusion", "orig-h", "test"),
            record("i", "real", "orig-i", "test"),
        ]

        groups = cross_generator_records(
            records,
            train_mods=["pix2pix"],
            valid_mods=["pix2pix"],
            test_mods=["cycle", "diffusion"],
        )

        self.assertEqual({item.mod for item in groups["train"]}, {"pix2pix", "real"})
        self.assertEqual({item.mod for item in groups["valid"]}, {"pix2pix", "real"})
        self.assertEqual({item.mod for item in groups["test"]}, {"cycle", "diffusion", "real"})

    def test_leave_generator_out_has_no_orig_id_overlap(self):
        records = [
            record("a", "pix2pix", "orig-a", "train"),
            record("b", "cycle", "orig-b", "train"),
            record("c", "real", "orig-c", "train"),
            record("d", "diffusion", "orig-d", "test"),
            record("e", "real", "orig-e", "test"),
        ]

        groups = leave_generator_out_records(
            records,
            heldout_mod="diffusion",
            train_mods=["pix2pix", "cycle"],
            valid_splits=("valid",),
        )

        self.assertEqual({item.mod for item in groups["train"]}, {"pix2pix", "cycle", "real"})
        self.assertEqual({item.mod for item in groups["test"]}, {"diffusion", "real"})

    def test_orig_id_leakage_raises(self):
        groups = {
            "train": [record("a", "pix2pix", "shared", "train")],
            "test": [record("b", "diffusion", "shared", "test")],
        }

        with self.assertRaises(ValueError):
            assert_no_orig_id_leakage(groups)


if __name__ == "__main__":
    unittest.main()
