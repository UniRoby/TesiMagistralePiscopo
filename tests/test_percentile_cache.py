"""Persistent percentiles must save work without changing model inputs."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import patch

import numpy as np
import pytest

from _tiff_fixtures import make_fake_corpus, write_tiff_stack
from tesi_m3d.dataset import M3DSynthPatchDataset, PatchExample, normalize_percentile, percentile_bounds
from tesi_m3d.volume_io import load_normalized_scan, normalize_scan


def test_reuses_bounds_after_volume_eviction_and_dataset_recreation(tmp_path):
    pytest.importorskip("torch")
    root, records = make_fake_corpus(tmp_path / "data", n_records=2, scan_z=3, shape=(8, 8))
    examples = [PatchExample(i, (0, 0, 0), 0, 0) for i in range(2)]
    kwargs = dict(examples=examples, patch_shape=(2, 2, 2), volume_cache_size=1, cache_dir=tmp_path / "cache")
    with patch("tesi_m3d.dataset.percentile_bounds", wraps=percentile_bounds) as compute:
        dataset = M3DSynthPatchDataset(records, root, **kwargs)
        first = dataset[0]["image"].clone()
        dataset[1]
        np.testing.assert_array_equal(dataset[0]["image"].numpy(), first.numpy())
        recreated = M3DSynthPatchDataset(records, root, **kwargs)
        np.testing.assert_array_equal(recreated[0]["image"].numpy(), first.numpy())
        assert compute.call_count == 2
        assert dataset._volume_cache.misses == 3


@pytest.mark.parametrize("kind", ["outliers", "zero", "constant"])
def test_matches_original_normalization_exactly(tmp_path, kind):
    raw = np.arange(3 * 8 * 8, dtype=np.uint16).reshape(3, 8, 8)
    raw[-1, -1, -1] = 65535
    if kind == "zero":
        raw.fill(0)
    elif kind == "constant":
        raw.fill(100)
    dirname = write_tiff_stack(tmp_path / "scan", raw)
    expected = normalize_percentile(raw)
    if kind == "outliers":
        assert expected.min() < 0 and expected.max() > 1  # Preserve the absence of clipping.
    with patch("tesi_m3d.dataset.percentile_bounds", wraps=percentile_bounds) as compute:
        for _ in range(2):
            actual = normalize_scan(raw, dirname, tmp_path / "cache")
            assert actual.dtype == np.float32
            np.testing.assert_array_equal(actual, expected)
        assert compute.call_count == 1


def test_invalidates_on_middle_slice_edit_and_percentile_change(tmp_path):
    raw = np.arange(3 * 8 * 8, dtype=np.uint16).reshape(3, 8, 8)
    dirname = write_tiff_stack(tmp_path / "scan", raw)
    cache = tmp_path / "cache"
    normalize_scan(raw, dirname, cache)
    raw[1] = 3000
    write_tiff_stack(tmp_path / "replacement", raw[1:2])
    (tmp_path / "replacement/slide0000.tiff").replace(dirname / "slide0001.tiff")
    # Guarantee a distinct timestamp even on filesystems with coarse clocks.
    middle = dirname / "slide0001.tiff"
    stat = middle.stat()
    os.utime(middle, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
    expected = normalize_percentile(raw)
    expected95 = normalize_percentile(raw, high=95)
    with patch("tesi_m3d.dataset.percentile_bounds", wraps=percentile_bounds) as compute:
        np.testing.assert_array_equal(normalize_scan(raw, dirname, cache), expected)
        np.testing.assert_array_equal(normalize_scan(raw, dirname, cache, high=95), expected95)
        assert compute.call_count == 2
    assert len(list((cache / "percentiles").glob("*.json"))) == 3


def test_real_aliases_share_one_entry_and_added_slice_invalidates(tmp_path):
    root, records = make_fake_corpus(tmp_path / "data", n_records=0, n_real=2, scan_z=3, shape=(8, 8))
    cache = tmp_path / "cache"
    with patch("tesi_m3d.dataset.percentile_bounds", wraps=percentile_bounds) as compute:
        first = load_normalized_scan(root, records[0], cache_dir=cache)
        np.testing.assert_array_equal(load_normalized_scan(root, records[1], cache_dir=cache), first)
        assert compute.call_count == 1
        dirname = next((root / "real/scan").iterdir())
        write_tiff_stack(tmp_path / "extra", np.full((1, 8, 8), 6000, dtype=np.uint16))
        (tmp_path / "extra/slide0000.tiff").replace(dirname / "slide0003.tiff")
        assert load_normalized_scan(root, records[0], cache_dir=cache).shape[0] == 4
        assert compute.call_count == 2


@pytest.mark.parametrize("damaged", ["broken json", "[NaN, 1]", "[5, 1]", "{}", "[[1, 2]]"])
def test_rebuilds_damaged_cache(tmp_path, damaged):
    raw = np.arange(64, dtype=np.uint16).reshape(1, 8, 8)
    dirname = write_tiff_stack(tmp_path / "scan", raw)
    expected = normalize_scan(raw, dirname, tmp_path / "cache")
    path = next((tmp_path / "cache/percentiles").glob("*.json"))
    path.write_text(damaged)
    np.testing.assert_array_equal(normalize_scan(raw, dirname, tmp_path / "cache"), expected)
    assert len(json.loads(path.read_text())) == 2


def test_concurrent_writers_publish_complete_cache(tmp_path):
    raw = np.arange(64, dtype=np.uint16).reshape(1, 8, 8)
    dirname = write_tiff_stack(tmp_path / "scan", raw)
    expected = normalize_percentile(raw)
    barrier = Barrier(2, timeout=10)

    def compute(*args):
        barrier.wait()
        return percentile_bounds(*args)

    with patch("tesi_m3d.dataset.percentile_bounds", side_effect=compute):
        with ThreadPoolExecutor(max_workers=2) as pool:
            outputs = list(pool.map(lambda _: normalize_scan(raw, dirname, tmp_path / "cache"), range(2)))
    for output in outputs:
        np.testing.assert_array_equal(output, expected)
    assert len(list((tmp_path / "cache/percentiles").iterdir())) == 1
    with patch("tesi_m3d.dataset.percentile_bounds", side_effect=AssertionError("cache miss")):
        np.testing.assert_array_equal(normalize_scan(raw, dirname, tmp_path / "cache"), expected)


def test_cache_write_failure_warns_without_changing_input(tmp_path):
    raw = np.arange(64, dtype=np.uint16).reshape(1, 8, 8)
    dirname = write_tiff_stack(tmp_path / "scan", raw)
    with patch("tesi_m3d.volume_io.os.replace", side_effect=PermissionError("read-only cache")):
        with pytest.warns(RuntimeWarning, match="Cannot save percentile cache"):
            actual = normalize_scan(raw, dirname, tmp_path / "cache")
    np.testing.assert_array_equal(actual, normalize_percentile(raw))
    assert not list((tmp_path / "cache/percentiles").iterdir())
