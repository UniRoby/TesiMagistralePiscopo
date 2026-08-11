"""Tests for cheap volume probing, mask alignment, and the volume cache."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tesi_m3d.dataset import load_tiff_stack
from tesi_m3d.volume_io import (
    SLICE_TEMPLATE,
    VolumeCache,
    align_mask_to_scan,
    count_contiguous_slices,
    load_label_mask,
    probe_volume_shape,
)

from _tiff_fixtures import make_fake_corpus, write_tiff_stack


class TestProbing(unittest.TestCase):
    def test_probe_matches_full_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = make_fake_corpus(tmp, n_records=1, scan_z=12, shape=(20, 24))
            scan_dir = root / "pix2pix" / "scan" / "inj_1"
            self.assertEqual(probe_volume_shape(scan_dir), load_tiff_stack(scan_dir).shape)

    def test_count_stops_at_first_gap_like_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "gappy"
            write_tiff_stack(target, np.zeros((3, 8, 8), dtype=np.uint16))
            # A file past the gap must be ignored, exactly as load_tiff_stack does.
            write_tiff_stack(Path(tmp) / "extra", np.zeros((1, 8, 8), dtype=np.uint16))
            (Path(tmp) / "extra" / SLICE_TEMPLATE.format(0)).replace(target / SLICE_TEMPLATE.format(4))
            self.assertEqual(count_contiguous_slices(target), 3)
            self.assertEqual(load_tiff_stack(target).shape[0], 3)

    def test_label_mask_loads_as_bool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = make_fake_corpus(tmp, n_records=1, scan_z=10, shape=(16, 16))
            mask = load_label_mask(root / "pix2pix" / "label" / "inj_1")
            self.assertEqual(mask.dtype, np.dtype(bool))
            self.assertTrue(mask.any())


class TestAlignment(unittest.TestCase):
    def test_trims_longer_mask(self) -> None:
        mask = np.ones((11, 4, 4), dtype=bool)
        self.assertEqual(align_mask_to_scan(mask, (10, 4, 4)).shape, (10, 4, 4))

    def test_pads_shorter_mask_with_background(self) -> None:
        mask = np.ones((8, 4, 4), dtype=bool)
        aligned = align_mask_to_scan(mask, (10, 4, 4))
        self.assertEqual(aligned.shape, (10, 4, 4))
        self.assertFalse(aligned[8:].any())

    def test_rejects_spatial_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            align_mask_to_scan(np.ones((10, 4, 5), dtype=bool), (10, 4, 4), img_id="inj_1")


class TestVolumeCache(unittest.TestCase):
    def test_loader_called_once_while_resident(self) -> None:
        cache = VolumeCache(maxsize=2)
        calls = []

        def loader():
            calls.append(1)
            return np.zeros(4)

        cache.get("a", loader)
        cache.get("a", loader)
        self.assertEqual(len(calls), 1)
        self.assertEqual(cache.hits, 1)

    def test_evicts_least_recently_used(self) -> None:
        cache = VolumeCache(maxsize=2)
        for key in ("a", "b", "c"):
            cache.get(key, lambda: np.zeros(4))
        self.assertEqual(len(cache), 2)
        # "a" was evicted, so fetching it must call the loader again.
        calls = []
        cache.get("a", lambda: (calls.append(1), np.zeros(4))[1])
        self.assertEqual(len(calls), 1)

    def test_pickle_drops_cached_arrays(self) -> None:
        cache = VolumeCache(maxsize=3)
        cache.get("a", lambda: np.zeros(1000))
        restored = pickle.loads(pickle.dumps(cache))
        self.assertEqual(len(restored), 0)
        self.assertEqual(restored.maxsize, 3)


if __name__ == "__main__":
    unittest.main()
