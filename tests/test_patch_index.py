"""Tests for the patch index, its disk cache, and the scan-shape grid fix."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tesi_m3d import patch_index as patch_index_module
from tesi_m3d.dataset import PatchExample, build_patch_examples
from tesi_m3d.patch_index import (
    PatchIndex,
    build_patch_index,
    index_cache_key,
    load_or_build_patch_index,
)

from _tiff_fixtures import make_fake_corpus


class TestGridUsesScanShape(unittest.TestCase):
    def test_coords_never_exceed_scan_extent(self) -> None:
        """Regression: the grid used to be built from the mask, which is z+1."""

        with tempfile.TemporaryDirectory() as tmp:
            scan_z, patch_z = 40, 32
            root, records = make_fake_corpus(tmp, n_records=1, scan_z=scan_z, mask_z_offset=1)
            index = build_patch_index(
                records, root, patch_shape=(32, 32, 32), stride=(8, 8, 8), progress=False
            )
            max_z_start = int(index.coord[:, 0].max())
            self.assertEqual(max_z_start, scan_z - patch_z)
            self.assertLessEqual(max_z_start + patch_z, scan_z)

    def test_positive_and_negative_patches_are_both_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, records = make_fake_corpus(tmp, n_records=2, scan_z=40)
            index = build_patch_index(
                records, root, patch_shape=(16, 16, 16), stride=(8, 8, 8), progress=False
            )
            self.assertGreater(index.n_positive, 0)
            self.assertGreater(len(index) - index.n_positive, 0)
            self.assertTrue(np.all(index.soft_score[index.label == 0] == 0.0))


class TestRealRecords(unittest.TestCase):
    def test_real_records_never_load_a_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, records = make_fake_corpus(tmp, n_records=0, n_real=2, scan_z=40)
            original = patch_index_module.load_label_mask
            patch_index_module.load_label_mask = lambda path: self.fail(
                "real records must not read a label stack"
            )
            try:
                index = build_patch_index(
                    records, root, patch_shape=(16, 16, 16), stride=(16, 16, 16), progress=False
                )
            finally:
                patch_index_module.load_label_mask = original
            self.assertEqual(index.n_positive, 0)
            self.assertGreater(len(index), 0)

    def test_shared_real_directory_probed_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, records = make_fake_corpus(
                tmp, n_records=0, n_real=4, real_shared_dirs=1, scan_z=40
            )
            calls: list[str] = []
            original = patch_index_module.probe_volume_shape

            def counting_probe(path):
                calls.append(str(path))
                return original(path)

            patch_index_module.probe_volume_shape = counting_probe
            try:
                build_patch_index(
                    records, root, patch_shape=(16, 16, 16), stride=(16, 16, 16), progress=False
                )
            finally:
                patch_index_module.probe_volume_shape = original
            # Four records, one shared directory: probe exactly once.
            self.assertEqual(len(calls), 1)


class TestCacheKeyAndPersistence(unittest.TestCase):
    def test_key_changes_with_every_relevant_parameter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, records = make_fake_corpus(tmp, n_records=2, scan_z=40)
            base = index_cache_key(records, (32, 32, 32), (32, 32, 32), 0.05)
            self.assertNotEqual(base, index_cache_key(records, (16, 16, 16), (32, 32, 32), 0.05))
            self.assertNotEqual(base, index_cache_key(records, (32, 32, 32), (16, 16, 16), 0.05))
            self.assertNotEqual(base, index_cache_key(records, (32, 32, 32), (32, 32, 32), 0.10))
            self.assertNotEqual(base, index_cache_key(records[:1], (32, 32, 32), (32, 32, 32), 0.05))
            self.assertEqual(base, index_cache_key(records, (32, 32, 32), (32, 32, 32), 0.05))

    def test_round_trip_preserves_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, records = make_fake_corpus(tmp, n_records=2, scan_z=40)
            index = build_patch_index(
                records, root, patch_shape=(16, 16, 16), stride=(16, 16, 16), progress=False
            )
            path = Path(tmp) / "index.npz"
            index.save(path)
            restored = PatchIndex.load(path)
            np.testing.assert_array_equal(index.record_index, restored.record_index)
            np.testing.assert_array_equal(index.coord, restored.coord)
            np.testing.assert_array_equal(index.label, restored.label)
            np.testing.assert_allclose(index.soft_score, restored.soft_score)

    def test_second_call_uses_cache_without_touching_tiffs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, records = make_fake_corpus(tmp, n_records=2, scan_z=40)
            cache_dir = Path(tmp) / "cache"
            kwargs = dict(patch_shape=(16, 16, 16), stride=(16, 16, 16), progress=False)
            first = load_or_build_patch_index(records, root, cache_dir, **kwargs)

            original = patch_index_module.probe_volume_shape
            patch_index_module.probe_volume_shape = lambda path: self.fail("cache miss")
            try:
                second = load_or_build_patch_index(records, root, cache_dir, **kwargs)
            finally:
                patch_index_module.probe_volume_shape = original
            np.testing.assert_array_equal(first.coord, second.coord)


class TestBackwardCompatibility(unittest.TestCase):
    def test_iteration_yields_patch_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, records = make_fake_corpus(tmp, n_records=1, scan_z=40)
            index = build_patch_index(
                records, root, patch_shape=(16, 16, 16), stride=(16, 16, 16), progress=False
            )
            first = index[0]
            self.assertIsInstance(first, PatchExample)
            self.assertIsInstance(first.coord, tuple)
            self.assertEqual(len(list(index)), len(index))

    def test_build_patch_examples_still_returns_a_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, records = make_fake_corpus(tmp, n_records=1, scan_z=40)
            examples = build_patch_examples(
                records, root, patch_shape=(16, 16, 16), stride=(16, 16, 16)
            )
            self.assertIsInstance(examples, list)
            self.assertTrue(all(isinstance(e, PatchExample) for e in examples))


if __name__ == "__main__":
    unittest.main()
