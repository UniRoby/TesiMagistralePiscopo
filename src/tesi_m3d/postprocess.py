"""Post-processing utilities for volumetric tampering heatmaps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PostprocessResult:
    """Post-processed heatmap and binary mask."""

    heatmap: np.ndarray
    binary_mask: np.ndarray


def normalize01(values: np.ndarray) -> np.ndarray:
    """Normalize an array to ``[0, 1]`` while handling constant arrays."""

    values = np.asarray(values, dtype=np.float32)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32, copy=False)


def postprocess_heatmap(
    heatmap: np.ndarray,
    threshold: float = 0.5,
    smoothing_sigma: float | None = None,
    min_component_size: int = 0,
) -> PostprocessResult:
    """Smooth, normalize, threshold, and optionally clean small components.

    Args:
        heatmap: Raw 3D heatmap with higher values near suspected tampering.
        threshold: Threshold applied after optional smoothing and normalization.
        smoothing_sigma: Optional Gaussian sigma; ``None`` or ``0`` disables it.
        min_component_size: Remove connected components smaller than this size.

    Returns:
        ``PostprocessResult`` with normalized heatmap and binary mask.
    """

    processed = np.asarray(heatmap, dtype=np.float32)
    if smoothing_sigma is not None and smoothing_sigma > 0:
        try:
            from scipy import ndimage
        except ImportError as exc:  # pragma: no cover - optional train env
            raise RuntimeError("scipy is required for smoothing") from exc
        processed = ndimage.gaussian_filter(processed, sigma=float(smoothing_sigma)).astype(np.float32)

    processed = normalize01(processed)
    binary = processed >= float(threshold)
    if min_component_size > 0:
        binary = remove_small_components(binary, min_component_size=min_component_size)
    return PostprocessResult(heatmap=processed, binary_mask=binary.astype(bool, copy=False))


def remove_small_components(binary_mask: np.ndarray, min_component_size: int) -> np.ndarray:
    """Remove connected 3D components smaller than ``min_component_size`` voxels."""

    try:
        from scipy import ndimage
    except ImportError as exc:  # pragma: no cover - optional train env
        raise RuntimeError("scipy is required for connected-component filtering") from exc

    if min_component_size <= 0:
        return np.asarray(binary_mask).astype(bool)
    binary = np.asarray(binary_mask).astype(bool)
    labeled, count = ndimage.label(binary)
    keep = np.zeros_like(binary, dtype=bool)
    for component_id in range(1, count + 1):
        component = labeled == component_id
        # Component sizes are counted in voxels because M3Dsynth masks live on voxel grids.
        if int(component.sum()) >= int(min_component_size):
            keep |= component
    return keep
