"""Tests for training-side wiring: seeding, subsetting, AMP, dataset contract."""

from __future__ import annotations

import pickle
import tempfile
import unittest

import numpy as np

from tesi_m3d.dataset import M3DSynthPatchDataset, M3DSynthRecord
from tesi_m3d.patch_index import build_patch_index
from tesi_m3d.model import Patch3DModelConfig, build_patch3d_classifier
from tesi_m3d.train import _checkpoint_payload, _load_checkpoint, load_yaml_config, set_global_seed, subset_records

from _tiff_fixtures import make_fake_corpus


def _records_with_types() -> list[M3DSynthRecord]:
    """Return records grouped by type, as data.csv actually orders them."""

    records = []
    for i in range(50):
        records.append(
            M3DSynthRecord(
                img_id=f"rem_{i}", ty="rem", mod="pix2pix", orig_id=f"O{i}",
                sdir_id="1", coord_z=0, coord_y=0, coord_x=0, split="train",
            )
        )
    for i in range(50):
        records.append(
            M3DSynthRecord(
                img_id=f"inj_{i}", ty="inj", mod="pix2pix", orig_id=f"P{i}",
                sdir_id="1", coord_z=0, coord_y=0, coord_x=0, split="train",
            )
        )
    return records


class TestSubsetRecords(unittest.TestCase):
    def test_subset_is_deterministic(self) -> None:
        records = _records_with_types()
        first = [r.img_id for r in subset_records(records, 10, seed=21)]
        second = [r.img_id for r in subset_records(records, 10, seed=21)]
        self.assertEqual(first, second)

    def test_subset_keeps_both_manipulation_types(self) -> None:
        """A plain head slice would return only 'rem' records."""

        subset = subset_records(_records_with_types(), 20, seed=21)
        types = {r.ty for r in subset}
        self.assertEqual(types, {"rem", "inj"})

    def test_none_and_oversized_limits_return_everything(self) -> None:
        records = _records_with_types()
        self.assertEqual(len(subset_records(records, None)), len(records))
        self.assertEqual(len(subset_records(records, 999)), len(records))


class TestSeeding(unittest.TestCase):
    def test_set_global_seed_makes_numpy_reproducible(self) -> None:
        set_global_seed(21)
        first = np.random.rand(5)
        set_global_seed(21)
        np.testing.assert_array_equal(first, np.random.rand(5))


class TestBaselineAndResume(unittest.TestCase):
    def test_baseline_config_has_the_bounded_training_budget(self) -> None:
        config = load_yaml_config("configs/train_pix2pix_baseline.yaml")
        self.assertEqual(config["data"]["max_train_records"], 256)
        self.assertEqual(config["training"]["batch_size"], 32)
        self.assertEqual(config["training"]["max_patches_per_epoch"], 4096)
        self.assertEqual(config["patches"]["inference_stride"], [16, 16, 16])
        self.assertAlmostEqual(config["patches"]["positive_volume_fraction"], 0.67)
        self.assertEqual(config["evaluation"]["detection_score"], "auto")

    def test_checkpoint_restores_model_optimizer_and_epoch(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        class Objects:
            pass

        source = Objects()
        source.model = build_patch3d_classifier(Patch3DModelConfig(base_channels=4))
        source.optimizer = torch.optim.AdamW(source.model.parameters(), lr=1e-3)
        source.scaler = torch.amp.GradScaler("cuda", enabled=False)
        payload = _checkpoint_payload(source, {"model": {"base_channels": 4}}, epoch=3, best_ap=0.7)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = f"{tmp}/resume.pt"
            torch.save(payload, checkpoint)
            target = Objects()
            target.model = build_patch3d_classifier(Patch3DModelConfig(base_channels=4))
            target.optimizer = torch.optim.AdamW(target.model.parameters(), lr=1e-3)
            target.scaler = torch.amp.GradScaler("cuda", enabled=False)
            epoch, best_ap, without_improvement = _load_checkpoint(checkpoint, target)

        self.assertEqual(epoch, 3)
        self.assertEqual(best_ap, 0.7)
        self.assertEqual(without_improvement, 0)
        for expected, actual in zip(source.model.parameters(), target.model.parameters()):
            self.assertTrue(torch.equal(expected, actual))


class TestDatasetContract(unittest.TestCase):
    def _dataset(self, tmp):
        root, records = make_fake_corpus(tmp, n_records=2, scan_z=40, shape=(48, 48))
        index = build_patch_index(
            records, root, patch_shape=(16, 16, 16), stride=(16, 16, 16), progress=False
        )
        return M3DSynthPatchDataset(
            records, data_root=root, examples=index,
            patch_shape=(16, 16, 16), stride=(16, 16, 16),
        )

    def test_item_shapes_and_keys(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._dataset(tmp)
            sample = dataset[0]
            self.assertEqual(set(sample), {"image", "label", "soft_score"})
            self.assertEqual(tuple(sample["image"].shape), (1, 16, 16, 16))
            self.assertEqual(tuple(sample["label"].shape), (1,))
            self.assertEqual(tuple(sample["soft_score"].shape), (1,))

    def test_no_padding_warning_for_a_well_formed_index(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._dataset(tmp)
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                for index in range(len(dataset)):
                    dataset[index]

    def test_labels_property_matches_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._dataset(tmp)
            np.testing.assert_array_equal(dataset.labels, dataset.examples.label)

    def test_volume_key_groups_by_scan_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._dataset(tmp)
            keys = {dataset.volume_key(i) for i in range(len(dataset))}
            self.assertEqual(len(keys), 2)

    def test_pickle_drops_resident_volumes(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._dataset(tmp)
            dataset[0]
            self.assertGreater(len(dataset._volume_cache), 0)
            restored = pickle.loads(pickle.dumps(dataset))
            self.assertEqual(len(restored._volume_cache), 0)
            self.assertEqual(len(restored), len(dataset))

    def test_volume_cache_prevents_reloading(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._dataset(tmp)
            same_volume = [
                i for i in range(len(dataset))
                if dataset.volume_key(i) == dataset.volume_key(0)
            ][:10]
            for index in same_volume:
                dataset[index]
            self.assertEqual(dataset._volume_cache.misses, 1)
            self.assertEqual(dataset._volume_cache.hits, len(same_volume) - 1)


if __name__ == "__main__":
    unittest.main()
