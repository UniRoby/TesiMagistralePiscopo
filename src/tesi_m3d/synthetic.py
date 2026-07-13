"""Synthetic NumPy data for smoke-testing the full 3D patch pipeline.

These volumes are not medical data and must not be used as scientific results.
They only validate geometry, labels, model forward/backward, heatmap reconstruction,
and metric code before the real M3Dsynth/LIDC data are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .patches import PatchGrid, binary_cube_mask, labels_from_mask, patch_overlap_fraction


@dataclass(frozen=True)
class SyntheticVolume:
    """One fake 3D volume and its known manipulation mask."""

    volume: np.ndarray
    mask: np.ndarray
    generator: str
    orig_id: str


@dataclass(frozen=True)
class SyntheticPatchExample:
    """One patch-level sample from a synthetic volume."""

    volume_index: int
    coord: tuple[int, int, int]
    label: int
    soft_score: float


def make_synthetic_volume(
    shape: Sequence[int] = (64, 64, 64),
    generator: str = "pix2pix",
    cube_start: Sequence[int] = (16, 16, 16),
    cube_size: Sequence[int] = (24, 24, 24),
    seed: int = 0,
) -> SyntheticVolume:
    """Create one fake CT-like volume with one known manipulated cuboid."""

    rng = np.random.default_rng(seed)
    shape = tuple(int(v) for v in shape)
    base = rng.normal(loc=0.1, scale=0.02, size=shape).astype(np.float32)
    mask = binary_cube_mask(shape, cube_start, cube_size)
    z, y, x = np.indices(shape, dtype=np.float32)

    if generator == "pix2pix":
        pattern = 0.35 + 0.05 * np.sin(x / 3.0)
    elif generator == "cycle":
        pattern = 0.35 + 0.05 * np.cos((y + z) / 4.0)
    elif generator == "diffusion":
        pattern = 0.35 + rng.normal(loc=0.0, scale=0.03, size=shape).astype(np.float32)
    else:
        pattern = np.full(shape, 0.35, dtype=np.float32)

    volume = base.copy()
    # Pattern is injected only inside the known mask, mimicking local manipulation.
    volume[mask] = np.asarray(pattern, dtype=np.float32)[mask]
    volume = np.clip(volume, 0.0, 1.0).astype(np.float32)
    return SyntheticVolume(volume=volume, mask=mask, generator=generator, orig_id=f"synthetic_{generator}_{seed}")


def make_synthetic_volumes(
    generators: Sequence[str] = ("pix2pix", "cycle", "diffusion"),
    volumes_per_generator: int = 2,
    shape: Sequence[int] = (64, 64, 64),
) -> list[SyntheticVolume]:
    """Create several fake volumes across fake generator names."""

    volumes: list[SyntheticVolume] = []
    for generator_index, generator in enumerate(generators):
        for sample_index in range(volumes_per_generator):
            offset = 8 + (sample_index * 4)
            volumes.append(
                make_synthetic_volume(
                    shape=shape,
                    generator=generator,
                    cube_start=(offset, 16, 16),
                    cube_size=(24, 24, 24),
                    seed=(generator_index * 100) + sample_index,
                )
            )
    return volumes


def build_synthetic_patch_examples(
    volumes: Sequence[SyntheticVolume],
    patch_shape: tuple[int, int, int] = (32, 32, 32),
    stride: tuple[int, int, int] = (16, 16, 16),
    positive_overlap_fraction: float = 0.05,
) -> list[SyntheticPatchExample]:
    """Build patch-level samples and skip ambiguous boundary patches."""

    examples: list[SyntheticPatchExample] = []
    for volume_index, item in enumerate(volumes):
        grid = PatchGrid(tuple(item.volume.shape), patch_shape=patch_shape, stride=stride)
        labels = labels_from_mask(item.mask, grid, positive_overlap_fraction=positive_overlap_fraction)
        for coord, slc, label in zip(grid.iter_coords(), grid.iter_slices(), labels):
            if label is None:
                # Tiny partial-overlap patches are excluded to avoid noisy synthetic supervision.
                continue
            examples.append(
                SyntheticPatchExample(
                    volume_index=volume_index,
                    coord=coord,
                    label=int(label),
                    soft_score=patch_overlap_fraction(item.mask[slc]),
                )
            )
    return examples


class SyntheticPatchDataset:
    """Torch-compatible patch dataset built from synthetic NumPy volumes."""

    def __init__(
        self,
        volumes: Sequence[SyntheticVolume],
        patch_shape: tuple[int, int, int] = (32, 32, 32),
        stride: tuple[int, int, int] = (16, 16, 16),
        positive_overlap_fraction: float = 0.05,
    ) -> None:
        """Store fake volumes and precompute patch index."""

        self.volumes = list(volumes)
        self.patch_shape = patch_shape
        self.examples = build_synthetic_patch_examples(
            self.volumes,
            patch_shape=patch_shape,
            stride=stride,
            positive_overlap_fraction=positive_overlap_fraction,
        )

    def __len__(self) -> int:
        """Return number of non-ambiguous patch examples."""

        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        """Return one patch as image, label, and metadata."""

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on train env
            raise RuntimeError("PyTorch is required for SyntheticPatchDataset") from exc

        example = self.examples[index]
        item = self.volumes[example.volume_index]
        z, y, x = example.coord
        dz, dy, dx = self.patch_shape
        patch = item.volume[z : z + dz, y : y + dy, x : x + dx]
        image = torch.from_numpy(patch[None].astype(np.float32, copy=False))
        label = torch.tensor([float(example.label)], dtype=torch.float32)
        soft_score = torch.tensor([float(example.soft_score)], dtype=torch.float32)
        return {
            "image": image,
            "label": label,
            "soft_score": soft_score,
            "generator": item.generator,
            "orig_id": item.orig_id,
            "coord": example.coord,
        }
