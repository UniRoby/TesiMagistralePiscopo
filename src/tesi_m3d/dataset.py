"""Dataset helpers for M3Dsynth metadata, TIFF stacks, and patch samples.

M3Dsynth data are expected under ``data_root/<mod>/scan/<img_id>/`` and
``data_root/<mod>/label/<img_id>/``. The project keeps this convention isolated
here so training, inference, and tests can share the same path logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import warnings
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

from .volume_io import align_mask_to_scan, load_label_mask, load_normalized_scan, VolumeCache

if TYPE_CHECKING:  # pragma: no cover - import cycle is resolved lazily at runtime
    from .patch_index import PatchIndex


@dataclass(frozen=True)
class M3DSynthRecord:
    """One row from official M3Dsynth metadata."""

    img_id: str
    ty: str
    mod: str
    orig_id: str
    sdir_id: str
    coord_z: int
    coord_y: int
    coord_x: int
    split: str | None = None

    @property
    def is_real(self) -> bool:
        """Return True when record is an unmanipulated real CT volume."""

        return self.mod == "real"

    @property
    def is_manipulated(self) -> bool:
        """Return True when record belongs to a synthetic manipulation generator."""

        return not self.is_real

    @property
    def coord(self) -> tuple[int, int, int]:
        """Return manipulation coordinate in z, y, x order from metadata."""

        return (self.coord_z, self.coord_y, self.coord_x)


@dataclass(frozen=True)
class PatchExample:
    """Index entry describing one patch extracted from one volume."""

    record_index: int
    coord: tuple[int, int, int]
    label: int
    soft_score: float


def read_records(metadata_dir: str | Path, splits: Sequence[str] | None = None) -> list[M3DSynthRecord]:
    """Read official M3Dsynth CSV metadata without requiring pandas.

    Args:
        metadata_dir: Directory containing ``data.csv`` and optional ``sets.csv``.
        splits: Optional split names to keep, for example ``("train",)``.

    Returns:
        List of typed records with split copied from ``sets.csv`` when present.
    """

    metadata_dir = Path(metadata_dir)
    data_csv = metadata_dir / "data.csv"
    sets_csv = metadata_dir / "sets.csv"
    split_by_orig = _read_split_table(sets_csv) if sets_csv.exists() else {}
    wanted_splits = set(splits) if splits is not None else None

    records: list[M3DSynthRecord] = []
    with data_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            split = split_by_orig.get(row["orig_id"])
            if wanted_splits is not None and split not in wanted_splits:
                continue
            records.append(
                M3DSynthRecord(
                    img_id=row["img_id"],
                    ty=row["ty"],
                    mod=row["mod"],
                    orig_id=row["orig_id"],
                    sdir_id=row["sdir_id"],
                    coord_z=int(row["coord_z"]),
                    coord_y=int(row["coord_y"]),
                    coord_x=int(row["coord_x"]),
                    split=split,
                )
            )
    return records


def filter_records(
    records: Iterable[M3DSynthRecord],
    mods: Sequence[str],
    splits: Sequence[str] | None = None,
    include_real: bool = True,
) -> list[M3DSynthRecord]:
    """Filter records by generator, split, and optional real CT volumes."""

    wanted_mods = set(mods)
    if include_real:
        wanted_mods.add("real")
    wanted_splits = set(splits) if splits is not None else None
    return [
        record
        for record in records
        if record.mod in wanted_mods and (wanted_splits is None or record.split in wanted_splits)
    ]


def cross_generator_records(
    records: Iterable[M3DSynthRecord],
    train_mods: Sequence[str],
    test_mods: Sequence[str],
    valid_mods: Sequence[str] | None = None,
    train_splits: Sequence[str] = ("train",),
    valid_splits: Sequence[str] = ("valid",),
    test_splits: Sequence[str] = ("test",),
    include_real: bool = True,
) -> dict[str, list[M3DSynthRecord]]:
    """Build train/valid/test groups for cross-generator evaluation.

    Args:
        records: M3Dsynth metadata records.
        train_mods: Generators used for model fitting, often one generator.
        test_mods: Generators used only for final cross-generator evaluation.
        valid_mods: Generators used for validation; defaults to ``train_mods``.
        train_splits: Dataset split names used for training.
        valid_splits: Dataset split names used for validation.
        test_splits: Dataset split names used for testing.
        include_real: Include real CT records as negative examples.

    Returns:
        Dictionary with ``train``, ``valid``, and ``test`` record lists.
    """

    records = list(records)
    valid_mods = tuple(train_mods) if valid_mods is None else tuple(valid_mods)
    groups = {
        "train": filter_records(records, train_mods, train_splits, include_real=include_real),
        "valid": filter_records(records, valid_mods, valid_splits, include_real=include_real),
        "test": filter_records(records, test_mods, test_splits, include_real=include_real),
    }
    assert_no_orig_id_leakage(groups)
    return groups


def leave_generator_out_records(
    records: Iterable[M3DSynthRecord],
    heldout_mod: str,
    train_mods: Sequence[str],
    train_splits: Sequence[str] = ("train",),
    valid_splits: Sequence[str] = ("valid",),
    test_splits: Sequence[str] = ("test",),
    include_real: bool = True,
) -> dict[str, list[M3DSynthRecord]]:
    """Backward-compatible leave-generator-out wrapper.

    This baseline trains on two generators and tests on one held-out generator.
    The thesis protocol uses ``cross_generator_records`` with one train
    generator and two test generators.
    """

    return cross_generator_records(
        records,
        train_mods=train_mods,
        valid_mods=train_mods,
        test_mods=[heldout_mod],
        train_splits=train_splits,
        valid_splits=valid_splits,
        test_splits=test_splits,
        include_real=include_real,
    )


def assert_no_orig_id_leakage(groups: dict[str, Sequence[M3DSynthRecord]]) -> None:
    """Raise if one LIDC ``orig_id`` appears in more than one split group."""

    seen: dict[str, str] = {}
    for group_name, group_records in groups.items():
        for record in group_records:
            previous = seen.setdefault(record.orig_id, group_name)
            # Same orig_id across train/test would leak patient anatomy into evaluation.
            if previous != group_name:
                raise ValueError(
                    f"orig_id leakage: {record.orig_id} appears in both {previous} and {group_name}"
                )


def scan_dir(
    data_root: str | Path,
    record: M3DSynthRecord,
    real_scan_root: str | Path | None = None,
) -> Path:
    """Return the TIFF directory for a manipulated or real CT record.

    Official M3Dsynth data use ``real/scan/<img_id>`` and symbolic links for
    real records that share one source series.  The portable Windows converter
    stores each real series once as ``real/scan/<orig_id>__<sdir_id>`` because
    creating directory symlinks can require administrator privileges.  The
    official layout remains the first choice when it is present.
    """

    root = Path(real_scan_root) if record.is_real and real_scan_root is not None else Path(data_root) / record.mod / "scan"
    official_path = root / record.img_id
    if record.is_real and not official_path.exists():
        return root / f"{record.orig_id}__{record.sdir_id}"
    return official_path


def label_dir(data_root: str | Path, record: M3DSynthRecord) -> Path:
    """Return directory containing TIFF mask slices for one manipulated record."""

    return Path(data_root) / record.mod / "label" / record.img_id


def _load_tiff_stack_impl(dirname_str: str, dtype_name: str) -> np.ndarray:
    """Load TIFF stack from disk (shared implementation)."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("Pillow is required to load M3Dsynth TIFF stacks") from exc

    target_dtype = np.dtype(dtype_name)
    dirname = Path(dirname_str)
    slices = []
    index = 0
    while True:
        filename = dirname / f"slide{index:04d}.tiff"
        if not filename.exists():
            break
        with Image.open(filename) as image:
            slices.append(np.asarray(image))
        index += 1
    if not slices:
        raise FileNotFoundError(f"no TIFF slices found in {dirname}")
    stack = np.stack(slices, axis=0)
    return stack.astype(target_dtype) if stack.dtype != target_dtype else stack


def load_tiff_stack(
    dirname: str | Path,
    dtype: np.dtype | type | None = None,
    use_cache: bool | None = None,
) -> np.ndarray:
    """Load a TIFF stack saved as ``slide0000.tiff``, ``slide0001.tiff``, ...

    dtype: target dtype. If None, returns native uint16 from TIFF files.
    use_cache: deprecated and ignored. An ``lru_cache`` used to sit here, but
        caching raw stacks keyed by directory could not bound memory (1000
        entries exhausted 32 GB) and never hit anyway, because the sampler drew
        patches in fully random order. Caching now lives in
        :class:`~tesi_m3d.volume_io.VolumeCache`, which caches *normalized*
        volumes and is paired with a volume-grouped sampler that makes hits the
        common case.
    """

    if use_cache is not None:
        warnings.warn(
            "load_tiff_stack(use_cache=...) is deprecated and ignored; "
            "caching moved to tesi_m3d.volume_io.VolumeCache",
            DeprecationWarning,
            stacklevel=2,
        )
    dtype_str = "uint16" if dtype is None else np.dtype(dtype).name
    dirname_str = str(Path(dirname).resolve())
    return _load_tiff_stack_impl(dirname_str, dtype_str)


def normalize_percentile(scan: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Normalize a CT stack to ``[0, 1]`` using robust percentiles."""

    non_zero = scan[scan > 0]
    source = non_zero if non_zero.size else scan.reshape(-1)
    lo, hi = np.percentile(source, [low, high])
    if hi <= lo:
        return np.zeros_like(scan, dtype=np.float32)
    # Preallocate float32 and copy in-place to avoid temporary copy from astype
    result = np.empty(scan.shape, dtype=np.float32)
    result[:] = scan
    lo = np.float32(lo)
    hi = np.float32(hi)
    result -= lo
    result /= (hi - lo)
    return result


def load_scan_and_mask(
    data_root: str | Path,
    record: M3DSynthRecord,
    use_cache: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load normalized scan and boolean manipulation mask for one record.

    Mask z-misalignment is resolved by :func:`~tesi_m3d.volume_io.align_mask_to_scan`,
    which is the single alignment rule shared with the patch index builder.

    use_cache: deprecated and ignored, see :func:`load_tiff_stack`.
    """

    if use_cache is not None:
        warnings.warn(
            "load_scan_and_mask(use_cache=...) is deprecated and ignored",
            DeprecationWarning,
            stacklevel=2,
        )
    scan = load_normalized_scan(data_root, record)
    if record.is_real:
        return scan, np.zeros(scan.shape, dtype=bool)

    mask = align_mask_to_scan(
        load_label_mask(label_dir(data_root, record)),
        scan.shape,
        img_id=record.img_id,
    )
    if scan.shape != mask.shape:
        raise ValueError(f"scan/mask shape mismatch for {record.img_id}: {scan.shape} != {mask.shape}")
    return scan, mask


def build_patch_examples(
    records: Sequence[M3DSynthRecord],
    data_root: str | Path,
    patch_shape: tuple[int, int, int] = (32, 32, 32),
    stride: tuple[int, int, int] = (32, 32, 32),
    positive_overlap_fraction: float = 0.05,
) -> list[PatchExample]:
    """Create patch-level index entries by reading masks once.

    Boundary patches with a non-zero but sub-threshold overlap are ambiguous
    supervision and are skipped.

    Backward-compatible wrapper over :func:`~tesi_m3d.patch_index.build_patch_index`.
    Prefer the ``PatchIndex`` form for large record sets: materializing four
    million ``PatchExample`` objects costs hundreds of MB where the arrays cost
    tens.
    """

    from .patch_index import build_patch_index  # local import breaks the import cycle

    return list(
        build_patch_index(
            records,
            data_root,
            patch_shape=patch_shape,
            stride=stride,
            positive_overlap_fraction=positive_overlap_fraction,
            progress=False,
        )
    )


class M3DSynthPatchDataset:
    """Patch-level dataset returning tensors for classifier training.

    Each item is ``{"image": Tensor[1,D,H,W], "label": Tensor[1], "soft_score": Tensor[1]}``.
    The class does not subclass ``torch.utils.data.Dataset`` directly so this
    module remains importable without PyTorch; DataLoader only needs ``__len__``
    and ``__getitem__``.
    """

    def __init__(
        self,
        records: Sequence[M3DSynthRecord],
        data_root: str | Path,
        examples: "Sequence[PatchExample] | PatchIndex | None" = None,
        patch_shape: tuple[int, int, int] = (32, 32, 32),
        stride: tuple[int, int, int] = (32, 32, 32),
        positive_overlap_fraction: float = 0.05,
        cache_dir: str | Path | None = None,
        volume_cache_size: int = 2,
    ) -> None:
        """Store records and build or accept a patch index.

        cache_dir: where to persist the built patch index; ``None`` rebuilds it
            every time, which costs minutes on the full training set.
        volume_cache_size: how many normalized volumes to keep resident. Only
            useful together with :class:`~tesi_m3d.sampling.VolumeGroupedBatchSampler`,
            which keeps consecutive accesses on the same volume.
        """

        from .patch_index import PatchIndex, load_or_build_patch_index

        self.records = list(records)
        self.data_root = Path(data_root)
        self.patch_shape = tuple(int(v) for v in patch_shape)
        self.stride = tuple(int(v) for v in stride)

        if examples is None:
            self._index = load_or_build_patch_index(
                self.records,
                self.data_root,
                cache_dir,
                patch_shape=self.patch_shape,
                stride=self.stride,
                positive_overlap_fraction=positive_overlap_fraction,
            )
        elif isinstance(examples, PatchIndex):
            self._index = examples
        else:
            self._index = _index_from_examples(examples)

        self._volume_cache = VolumeCache(maxsize=volume_cache_size)
        # Resolving a path costs a syscall; the sampler asks for these keys once
        # per example at construction time, so they are memoized per record.
        self._volume_key_by_record: dict[int, str] = {}

    @property
    def examples(self) -> "PatchIndex":
        """Return the patch index; iterating it yields ``PatchExample``."""

        return self._index

    @property
    def labels(self) -> np.ndarray:
        """Return all patch labels as a ``uint8`` array without materializing objects."""

        return self._index.label

    def volume_key(self, index: int) -> str:
        """Return the resolved scan directory backing patch ``index``.

        This is the grouping key for the sampler and the volume cache. It is the
        *resolved* directory, so the 1787 real training records that point at 489
        directories collapse onto 489 keys instead of being reloaded per record.
        """

        record_index = int(self._index.record_index[index])
        key = self._volume_key_by_record.get(record_index)
        if key is None:
            key = str(scan_dir(self.data_root, self.records[record_index]).resolve())
            self._volume_key_by_record[record_index] = key
        return key

    def __len__(self) -> int:
        """Return number of supervised patch examples."""

        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, object]:
        """Return one patch as torch tensors, reusing the cached volume."""

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on train env
            raise RuntimeError("PyTorch is required to use M3DSynthPatchDataset") from exc

        record_index = int(self._index.record_index[index])
        record = self.records[record_index]
        # The mask is not needed here: labels come from the prebuilt index, and
        # loading it was doubling the IO for every patch.
        scan = self._volume_cache.get(
            self.volume_key(index),
            lambda: load_normalized_scan(self.data_root, record),
        )
        z, y, x = (int(v) for v in self._index.coord[index])
        dz, dy, dx = self.patch_shape
        patch = scan[z : z + dz, y : y + dy, x : x + dx]
        if patch.shape != self.patch_shape:
            # Unreachable for a well-formed index, since the grid is built from
            # the scan shape. Kept as a loud guard rather than a silent pad.
            warnings.warn(
                f"patch at {(z, y, x)} of {record.img_id} is {patch.shape}, "
                f"expected {self.patch_shape}; zero-padding",
                RuntimeWarning,
                stacklevel=2,
            )
            patch = np.pad(
                patch,
                [(0, d - s) for d, s in zip(self.patch_shape, patch.shape)],
                mode="constant",
                constant_values=0,
            )
        # Channel dimension is added here so model input becomes (B, 1, D, H, W).
        image = torch.from_numpy(np.ascontiguousarray(patch[None], dtype=np.float32))
        label = torch.tensor([float(self._index.label[index])], dtype=torch.float32)
        soft_score = torch.tensor([float(self._index.soft_score[index])], dtype=torch.float32)
        return {"image": image, "label": label, "soft_score": soft_score}

    def __getstate__(self) -> dict:
        """Pickle without resident volumes so spawned workers start empty."""

        state = self.__dict__.copy()
        state["_volume_cache"] = VolumeCache(maxsize=self._volume_cache.maxsize)
        return state


def _index_from_examples(examples: "Sequence[PatchExample]") -> "PatchIndex":
    """Convert a legacy ``PatchExample`` sequence into a :class:`PatchIndex`."""

    from .patch_index import PatchIndex

    examples = list(examples)
    if not examples:
        return PatchIndex(
            record_index=np.empty(0, dtype=np.int32),
            coord=np.empty((0, 3), dtype=np.int32),
            label=np.empty(0, dtype=np.uint8),
            soft_score=np.empty(0, dtype=np.float32),
        )
    return PatchIndex(
        record_index=np.asarray([e.record_index for e in examples], dtype=np.int32),
        coord=np.asarray([e.coord for e in examples], dtype=np.int32),
        label=np.asarray([e.label for e in examples], dtype=np.uint8),
        soft_score=np.asarray([e.soft_score for e in examples], dtype=np.float32),
    )


def _read_split_table(sets_csv: Path) -> dict[str, str]:
    """Read ``sets.csv`` into ``orig_id -> split`` mapping."""

    with sets_csv.open(newline="") as handle:
        return {row["orig_id"]: row["set"] for row in csv.DictReader(handle)}
