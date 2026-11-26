"""Lung ROI calculations must not cast the entire CT or label volume to int64."""
import numpy as np
import pytest

from tesi_m3d import lung


@pytest.mark.parametrize("values", [
    np.zeros((0, 3, 4), dtype=np.uint16),
    np.zeros((2, 3, 4), dtype=np.uint16),
    np.array([0, 24, 24, 24, 1024, 65535] * 15, dtype=np.uint16).reshape(3, 5, 6),
    np.arange(240, dtype=np.int32).reshape(4, 6, 10)[:, ::-1, ::2],
])
def test_chunked_histogram_and_offset_match_dense_without_large_temporaries(monkeypatch, values):
    bincount = np.bincount
    expected = bincount(values.reshape(-1))
    expected_offset = lung.estimate_ct_offset_from_histogram(expected)
    calls = []

    def bounded_bincount(chunk, **kwargs):
        calls.append(chunk.size)
        assert chunk.size <= 17
        return bincount(chunk, **kwargs)

    monkeypatch.setattr(lung, "_HISTOGRAM_CHUNK_SIZE", 17)
    monkeypatch.setattr(np, "bincount", bounded_bincount)
    np.testing.assert_array_equal(lung._chunked_bincount(values), expected)
    assert lung.estimate_ct_offset(values) == expected_offset
    assert sum(calls) == 2 * values.size


def test_lung_roi_is_identical_to_dense_counting(monkeypatch):
    scan = np.zeros((8, 32, 32), dtype=np.uint16)
    scan[:, 2:30, 2:30] = 3072
    scan[:, 8:18, 6:14] = 2072
    scan[:, 8:18, 18:26] = 2072
    chunked_bincount = lung._chunked_bincount
    monkeypatch.setattr(lung, "_chunked_bincount", lambda values: np.bincount(values.reshape(-1)))
    expected = lung.lung_mask_from_ct(scan)
    monkeypatch.setattr(lung, "_chunked_bincount", chunked_bincount)
    monkeypatch.setattr(lung, "_HISTOGRAM_CHUNK_SIZE", 257)
    actual = lung.lung_mask_from_ct(scan)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.bool_ and actual.any()
