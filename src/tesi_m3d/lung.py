"""Heuristic lung region extraction from stored uint16 chest CT volumes."""

from __future__ import annotations

import numpy as np

ALLOWED_OFFSETS = (0, 512, 1024, 2000, 2048, 3024, 3048, 3072)
LUNG_PEAK_HU = -1000
_HISTOGRAM_CHUNK_SIZE = 1 << 20


def _chunked_bincount(values: np.ndarray) -> np.ndarray:
    """Count exactly without bincount's full-volume int64 temporary."""
    histogram = np.zeros(0, dtype=np.int64)
    for start in range(0, values.size, _HISTOGRAM_CHUNK_SIZE):
        counts = np.bincount(values.flat[start:start + _HISTOGRAM_CHUNK_SIZE],
                             minlength=histogram.size)
        counts[:histogram.size] += histogram
        histogram = counts
    return histogram


def estimate_ct_offset_from_histogram(histogram: np.ndarray) -> tuple[int, int]:
    """Estimate the uint16 storage offset from a stored-value histogram."""

    histogram = np.asarray(histogram).reshape(-1)
    if histogram.size <= 1 or not np.any(histogram[1:]):
        return 0, 0
    peak = int(np.argmax(histogram[1:])) + 1
    guess = peak - LUNG_PEAK_HU
    offset = min(ALLOWED_OFFSETS, key=lambda candidate: abs(candidate - guess))
    return int(offset), peak


def estimate_ct_offset(scan: np.ndarray) -> tuple[int, int]:
    """Estimate the uint16 storage offset from the dominant non-zero air peak."""

    return estimate_ct_offset_from_histogram(_chunked_bincount(np.asarray(scan)))


def lung_mask_from_ct(scan: np.ndarray) -> np.ndarray:
    """Return a conservative lung mask from a raw stored-value CT stack.

    The body envelope prevents exterior air from entering the ROI. The two
    largest 3-D air components retain the lungs and reject small cavities.
    """

    try:
        from scipy import ndimage
    except ImportError as exc:  # pragma: no cover - optional train environment
        raise RuntimeError("SciPy is required to calculate lung-only metrics") from exc

    scan = np.asarray(scan)
    if scan.ndim != 3:
        raise ValueError("lung segmentation requires a 3-D CT volume")
    offset, _ = estimate_ct_offset(scan)
    hu = scan.astype(np.int32) - offset
    tissue = hu > -500
    envelope = np.stack([ndimage.binary_fill_holes(plane) for plane in tissue], axis=0)
    candidate = envelope & (hu >= -1000) & (hu <= -300)
    candidate = ndimage.binary_closing(
        candidate, structure=np.ones((1, 3, 3), dtype=bool), iterations=1
    )
    labels, count = ndimage.label(candidate)
    if count == 0:
        return np.zeros(scan.shape, dtype=bool)
    sizes = _chunked_bincount(labels)[1:]
    keep = np.argsort(sizes)[-2:] + 1
    lungs = np.isin(labels, keep)
    return ndimage.binary_dilation(lungs, iterations=1)
