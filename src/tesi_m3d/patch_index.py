"""Patch index: which patch of which volume, with what label.

Two problems with the previous ``build_patch_examples`` are addressed here.

**Memory.** A full pix2pix+real training index is ~4 million patches. As
``PatchExample`` dataclass instances that is several hundred MB of Python
objects; as parallel numpy arrays it is ~60 MB. :class:`PatchIndex` stores the
arrays and materializes a ``PatchExample`` only when one is actually asked for.

**Startup cost.** The old builder re-read every label stack from disk on every
process start and persisted nothing, and it loaded whole *scan* stacks just to
read ``.shape``. Here the grid comes from :func:`~tesi_m3d.volume_io.probe_volume_shape`
(one slice header), real volumes read no mask at all, and the finished index is
cached on disk under a key derived from the records and patch parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence
import hashlib
import json
import os

import numpy as np

from .dataset import M3DSynthRecord, PatchExample, label_dir, scan_dir
from .patches import PatchGrid
from .volume_io import align_mask_to_scan, load_label_mask, probe_volume_shape

PATCH_INDEX_FORMAT_VERSION = 1


@dataclass(frozen=True)
class PatchIndex:
    """Struct-of-arrays patch index.

    Indexing and iteration yield :class:`~tesi_m3d.dataset.PatchExample`, so code
    written against the old ``list[PatchExample]`` keeps working unchanged.
    """

    record_index: np.ndarray  # int32 (N,)
    coord: np.ndarray  # int32 (N, 3)
    label: np.ndarray  # uint8 (N,)
    soft_score: np.ndarray  # float32 (N,)

    def __post_init__(self) -> None:
        """Validate that all four arrays describe the same patches."""

        lengths = {
            len(self.record_index),
            len(self.coord),
            len(self.label),
            len(self.soft_score),
        }
        if len(lengths) != 1:
            raise ValueError(f"PatchIndex arrays have inconsistent lengths: {lengths}")
        if self.coord.ndim != 2 or self.coord.shape[1] != 3:
            raise ValueError("coord must have shape (N, 3)")

    def __len__(self) -> int:
        """Return the number of indexed patches."""

        return len(self.record_index)

    def __getitem__(self, index: int) -> PatchExample:
        """Materialize one :class:`PatchExample` on demand."""

        return PatchExample(
            record_index=int(self.record_index[index]),
            coord=(int(self.coord[index, 0]), int(self.coord[index, 1]), int(self.coord[index, 2])),
            label=int(self.label[index]),
            soft_score=float(self.soft_score[index]),
        )

    def __iter__(self) -> Iterator[PatchExample]:
        """Iterate patches as :class:`PatchExample` instances."""

        for index in range(len(self)):
            yield self[index]

    def subset(self, indices: np.ndarray) -> "PatchIndex":
        """Return a new index containing only ``indices``."""

        indices = np.asarray(indices)
        return PatchIndex(
            record_index=self.record_index[indices],
            coord=self.coord[indices],
            label=self.label[indices],
            soft_score=self.soft_score[indices],
        )

    @property
    def n_positive(self) -> int:
        """Return how many indexed patches are labelled positive."""

        return int(np.count_nonzero(self.label))

    def save(self, path: str | Path) -> None:
        """Write the index to ``path`` atomically as a compressed npz."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary name first: an interrupted run must never leave a
        # truncated npz that a later run would happily load as a valid cache.
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as handle:
            np.savez_compressed(
                handle,
                version=np.asarray(PATCH_INDEX_FORMAT_VERSION),
                record_index=self.record_index,
                coord=self.coord,
                label=self.label,
                soft_score=self.soft_score,
            )
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path) -> "PatchIndex":
        """Read an index written by :meth:`save`."""

        with np.load(Path(path)) as data:
            version = int(data["version"])
            if version != PATCH_INDEX_FORMAT_VERSION:
                raise ValueError(
                    f"patch index at {path} has format version {version}, "
                    f"expected {PATCH_INDEX_FORMAT_VERSION}"
                )
            return cls(
                record_index=data["record_index"],
                coord=data["coord"],
                label=data["label"],
                soft_score=data["soft_score"],
            )


def index_cache_key(
    records: Sequence[M3DSynthRecord],
    patch_shape: tuple[int, int, int],
    stride: tuple[int, int, int],
    positive_overlap_fraction: float,
) -> str:
    """Return a stable short hash identifying one index configuration.

    Everything that changes the index content is in the digest, including the
    format version, so a stale cache can never be paired with changed settings.
    """

    payload = {
        "version": PATCH_INDEX_FORMAT_VERSION,
        "patch_shape": [int(v) for v in patch_shape],
        "stride": [int(v) for v in stride],
        "positive_overlap_fraction": round(float(positive_overlap_fraction), 6),
        "records": [(r.mod, r.img_id, r.orig_id, r.sdir_id) for r in records],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def _grid_entries_for_mask(
    mask: np.ndarray,
    grid: PatchGrid,
    positive_overlap_fraction: float,
) -> tuple[list[tuple[int, int, int]], list[int], list[float]]:
    """Return coords, labels and soft scores for the supervised patches of one mask.

    Patches with a non-zero but sub-threshold overlap are ambiguous supervision
    and are dropped, matching :func:`~tesi_m3d.patches.patch_label` returning
    ``None``. The overlap fraction is computed once and reused for both the
    label and the soft score, where the old builder computed it twice.
    """

    coords: list[tuple[int, int, int]] = []
    labels: list[int] = []
    softs: list[float] = []
    patch_voxels = int(np.prod(grid.patch_shape))
    for coord, slc in zip(grid.iter_coords(), grid.iter_slices()):
        overlap = float(np.count_nonzero(mask[slc]) / patch_voxels)
        if overlap >= positive_overlap_fraction:
            label = 1
        elif overlap == 0.0:
            label = 0
        else:
            continue
        coords.append(coord)
        labels.append(label)
        softs.append(overlap)
    return coords, labels, softs


def build_patch_index(
    records: Sequence[M3DSynthRecord],
    data_root: str | Path,
    patch_shape: tuple[int, int, int] = (32, 32, 32),
    stride: tuple[int, int, int] = (32, 32, 32),
    positive_overlap_fraction: float = 0.05,
    progress: bool = True,
) -> PatchIndex:
    """Build the patch index without loading a single scan voxel.

    The grid is built from the *scan* shape, never the mask shape. pix2pix label
    stacks carry one extra z-slice, so a mask-derived grid emitted a final row of
    z-coordinates one slice past the end of the scan; those patches were then
    silently zero-padded at load time and could still be labelled positive.

    Real records share scan directories (1787 training records point at 489
    directories), so their grids are computed once per unique shape and replayed.
    """

    data_root = Path(data_root)
    patch_shape = tuple(int(v) for v in patch_shape)
    stride = tuple(int(v) for v in stride)

    iterator: Sequence[tuple[int, M3DSynthRecord]] = list(enumerate(records))
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="indexing volumes", unit="vol")
        except ImportError:  # pragma: no cover - tqdm is optional
            pass

    shape_by_dir: dict[str, tuple[int, int, int]] = {}
    real_by_shape: dict[tuple[int, int, int], tuple[list, list, list]] = {}

    record_index_parts: list[np.ndarray] = []
    coord_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    soft_parts: list[np.ndarray] = []

    for record_index, record in iterator:
        scan_path = str(scan_dir(data_root, record).resolve())
        scan_shape = shape_by_dir.get(scan_path)
        if scan_shape is None:
            scan_shape = probe_volume_shape(scan_path)
            shape_by_dir[scan_path] = scan_shape

        if record.is_real:
            cached = real_by_shape.get(scan_shape)
            if cached is None:
                grid = PatchGrid(scan_shape, patch_shape=patch_shape, stride=stride)
                coords = list(grid.iter_coords())
                cached = (coords, [0] * len(coords), [0.0] * len(coords))
                real_by_shape[scan_shape] = cached
            coords, labels, softs = cached
        else:
            grid = PatchGrid(scan_shape, patch_shape=patch_shape, stride=stride)
            mask = align_mask_to_scan(
                load_label_mask(label_dir(data_root, record)),
                scan_shape,
                img_id=record.img_id,
            )
            coords, labels, softs = _grid_entries_for_mask(mask, grid, positive_overlap_fraction)

        if not coords:
            continue
        record_index_parts.append(np.full(len(coords), record_index, dtype=np.int32))
        coord_parts.append(np.asarray(coords, dtype=np.int32))
        label_parts.append(np.asarray(labels, dtype=np.uint8))
        soft_parts.append(np.asarray(softs, dtype=np.float32))

    if not record_index_parts:
        return PatchIndex(
            record_index=np.empty(0, dtype=np.int32),
            coord=np.empty((0, 3), dtype=np.int32),
            label=np.empty(0, dtype=np.uint8),
            soft_score=np.empty(0, dtype=np.float32),
        )
    return PatchIndex(
        record_index=np.concatenate(record_index_parts),
        coord=np.concatenate(coord_parts),
        label=np.concatenate(label_parts),
        soft_score=np.concatenate(soft_parts),
    )


def load_or_build_patch_index(
    records: Sequence[M3DSynthRecord],
    data_root: str | Path,
    cache_dir: str | Path | None = None,
    *,
    patch_shape: tuple[int, int, int] = (32, 32, 32),
    stride: tuple[int, int, int] = (32, 32, 32),
    positive_overlap_fraction: float = 0.05,
    rebuild: bool = False,
    progress: bool = True,
) -> PatchIndex:
    """Return a cached patch index, building and caching it on a miss.

    Building the training index reads ~1500 label stacks and takes minutes; that
    cost should be paid once, not on every restart or failed run.
    """

    key = index_cache_key(records, patch_shape, stride, positive_overlap_fraction)
    if cache_dir is None:
        return build_patch_index(
            records,
            data_root,
            patch_shape=patch_shape,
            stride=stride,
            positive_overlap_fraction=positive_overlap_fraction,
            progress=progress,
        )

    cache_dir = Path(cache_dir)
    index_path = cache_dir / f"patch_index_{key}.npz"
    if index_path.exists() and not rebuild:
        return PatchIndex.load(index_path)

    index = build_patch_index(
        records,
        data_root,
        patch_shape=patch_shape,
        stride=stride,
        positive_overlap_fraction=positive_overlap_fraction,
        progress=progress,
    )
    index.save(index_path)
    # Sidecar so a human can tell what a cache file belongs to without loading it.
    sidecar = {
        "version": PATCH_INDEX_FORMAT_VERSION,
        "key": key,
        "data_root": str(data_root),
        "patch_shape": [int(v) for v in patch_shape],
        "stride": [int(v) for v in stride],
        "positive_overlap_fraction": float(positive_overlap_fraction),
        "n_records": len(records),
        "n_patches": len(index),
        "n_positive": index.n_positive,
    }
    index_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    return index
