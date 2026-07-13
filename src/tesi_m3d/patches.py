"""Patch extraction, labeling, and heatmap reconstruction for 3D CT volumes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class PatchGrid:
    """Regular 3D sliding-window grid in z, y, x order."""

    volume_shape: tuple[int, int, int]
    patch_shape: tuple[int, int, int] = (32, 32, 32)
    stride: tuple[int, int, int] = (16, 16, 16)

    def __post_init__(self) -> None:
        normalized = {
            "volume_shape": tuple(int(v) for v in self.volume_shape),
            "patch_shape": tuple(int(v) for v in self.patch_shape),
            "stride": tuple(int(v) for v in self.stride),
        }
        for name, values in normalized.items():
            if len(values) != 3:
                raise ValueError(f"{name} must contain exactly three dimensions")
            if any(v <= 0 for v in values):
                raise ValueError(f"{name} values must be positive")
        if any(p > v for p, v in zip(normalized["patch_shape"], normalized["volume_shape"])):
            raise ValueError("patch_shape cannot be larger than volume_shape")
        object.__setattr__(self, "volume_shape", normalized["volume_shape"])
        object.__setattr__(self, "patch_shape", normalized["patch_shape"])
        object.__setattr__(self, "stride", normalized["stride"])

    @property
    def starts(self) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return tuple(
            _axis_starts(size, patch, stride)
            for size, patch, stride in zip(self.volume_shape, self.patch_shape, self.stride)
        )  # type: ignore[return-value]

    def iter_slices(self) -> Iterator[tuple[slice, slice, slice]]:
        z_starts, y_starts, x_starts = self.starts
        dz, dy, dx = self.patch_shape
        for z in z_starts:
            for y in y_starts:
                for x in x_starts:
                    yield (slice(z, z + dz), slice(y, y + dy), slice(x, x + dx))

    def iter_coords(self) -> Iterator[tuple[int, int, int]]:
        z_starts, y_starts, x_starts = self.starts
        for z in z_starts:
            for y in y_starts:
                for x in x_starts:
                    yield (z, y, x)

    def __len__(self) -> int:
        z_starts, y_starts, x_starts = self.starts
        return len(z_starts) * len(y_starts) * len(x_starts)


def _axis_starts(size: int, patch: int, stride: int) -> tuple[int, ...]:
    if patch > size:
        raise ValueError("patch cannot be larger than axis size")
    starts = list(range(0, size - patch + 1, stride))
    last = size - patch
    if starts[-1] != last:
        starts.append(last)
    return tuple(starts)


def extract_patches(volume: np.ndarray, grid: PatchGrid) -> np.ndarray:
    """Extract all patches from a 3D volume using a regular grid."""

    if volume.ndim != 3:
        raise ValueError("volume must be a 3D array in z, y, x order")
    if tuple(volume.shape) != grid.volume_shape:
        raise ValueError("volume shape does not match grid.volume_shape")
    return np.stack([volume[slc] for slc in grid.iter_slices()], axis=0)


def patch_overlap_fraction(mask_patch: np.ndarray) -> float:
    """Fraction of manipulated voxels inside one patch."""

    if mask_patch.size == 0:
        raise ValueError("mask_patch cannot be empty")
    return float(np.count_nonzero(mask_patch) / mask_patch.size)


def patch_soft_score(mask_patch: np.ndarray) -> float:
    """Continuous soft label y in [0, 1] for ablation or regression losses."""

    return patch_overlap_fraction(mask_patch)


def patch_label(mask_patch: np.ndarray, positive_overlap_fraction: float = 0.05) -> int | None:
    """Binary label from overlap ratio.

    Returns 1 for positive synthetic patches, 0 for clean patches, and None for
    ambiguous boundary patches with non-zero overlap below threshold.
    """

    overlap = patch_overlap_fraction(mask_patch)
    if overlap >= positive_overlap_fraction:
        return 1
    if overlap == 0.0:
        return 0
    return None


def labels_from_mask(
    mask: np.ndarray,
    grid: PatchGrid,
    positive_overlap_fraction: float = 0.05,
) -> list[int | None]:
    """Compute binary patch labels for every grid position."""

    if mask.ndim != 3:
        raise ValueError("mask must be a 3D array in z, y, x order")
    if tuple(mask.shape) != grid.volume_shape:
        raise ValueError("mask shape does not match grid.volume_shape")
    return [
        patch_label(mask[slc], positive_overlap_fraction=positive_overlap_fraction)
        for slc in grid.iter_slices()
    ]


def soft_scores_from_mask(mask: np.ndarray, grid: PatchGrid) -> np.ndarray:
    """Compute continuous manipulated-voxel ratio for every patch."""

    if mask.ndim != 3:
        raise ValueError("mask must be a 3D array in z, y, x order")
    if tuple(mask.shape) != grid.volume_shape:
        raise ValueError("mask shape does not match grid.volume_shape")
    return np.asarray([patch_soft_score(mask[slc]) for slc in grid.iter_slices()], dtype=np.float32)


def gaussian_patch_weights(patch_shape: Sequence[int], sigma_scale: float = 0.125) -> np.ndarray:
    """Create separable 3D gaussian weights centered in a patch."""

    patch_shape = tuple(int(v) for v in patch_shape)
    if len(patch_shape) != 3 or any(v <= 0 for v in patch_shape):
        raise ValueError("patch_shape must contain three positive dimensions")
    axes = []
    for size in patch_shape:
        center = (size - 1) / 2.0
        sigma = max(size * float(sigma_scale), 1e-6)
        axis = np.arange(size, dtype=np.float32)
        axes.append(np.exp(-0.5 * ((axis - center) / sigma) ** 2))
    weights = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    weights /= np.max(weights)
    return weights.astype(np.float32, copy=False)


def reconstruct_heatmap(
    scores: Sequence[float] | np.ndarray,
    grid: PatchGrid,
    mode: str = "average",
    sigma_scale: float = 0.125,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Reconstruct a 3D heatmap from patch scores.

    mode="average" is baseline overlap averaging. mode="gaussian" weights each
    patch center more strongly, matching MONAI sliding-window practice.
    """

    scores_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(scores_arr) != len(grid):
        raise ValueError(f"expected {len(grid)} scores, got {len(scores_arr)}")
    if mode not in {"average", "gaussian"}:
        raise ValueError("mode must be 'average' or 'gaussian'")

    heatmap = np.zeros(grid.volume_shape, dtype=np.float32)
    norm = np.zeros(grid.volume_shape, dtype=np.float32)
    weights = np.ones(grid.patch_shape, dtype=np.float32)
    if mode == "gaussian":
        weights = gaussian_patch_weights(grid.patch_shape, sigma_scale=sigma_scale)

    for score, slc in zip(scores_arr, grid.iter_slices()):
        heatmap[slc] += float(score) * weights
        norm[slc] += weights

    if np.any(norm == 0):
        raise RuntimeError("grid reconstruction left uncovered voxels")
    return (heatmap / norm).astype(dtype, copy=False)


def binary_cube_mask(
    volume_shape: Sequence[int],
    start: Sequence[int],
    size: Sequence[int],
) -> np.ndarray:
    """Build a simple cuboid binary mask for tests and smoke experiments."""

    if len(volume_shape) != 3 or len(start) != 3 or len(size) != 3:
        raise ValueError("volume_shape, start and size must be 3D")
    slices = []
    for axis, (origin, width, limit) in enumerate(zip(start, size, volume_shape)):
        origin = int(origin)
        width = int(width)
        limit = int(limit)
        if origin < 0 or width <= 0 or origin + width > limit:
            raise ValueError(f"cube is outside volume on axis {axis}")
        slices.append(slice(origin, origin + width))
    mask = np.zeros(tuple(int(v) for v in volume_shape), dtype=bool)
    mask[tuple(slices)] = True
    return mask
