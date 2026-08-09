"""Dataset helpers for M3Dsynth metadata, TIFF stacks, and patch samples.

M3Dsynth data are expected under ``data_root/<mod>/scan/<img_id>/`` and
``data_root/<mod>/label/<img_id>/``. The project keeps this convention isolated
here so training, inference, and tests can share the same path logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import csv
from typing import Iterable, Sequence

import numpy as np

from .patches import PatchGrid, labels_from_mask, patch_overlap_fraction


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


def scan_dir(data_root: str | Path, record: M3DSynthRecord) -> Path:
    """Return the TIFF directory for a manipulated or real CT record.

    Official M3Dsynth data use ``real/scan/<img_id>`` and symbolic links for
    real records that share one source series.  The portable Windows converter
    stores each real series once as ``real/scan/<orig_id>__<sdir_id>`` because
    creating directory symlinks can require administrator privileges.  The
    official layout remains the first choice when it is present.
    """

    root = Path(data_root) / record.mod / "scan"
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


@lru_cache(maxsize=1000)
def _load_tiff_stack_cached(dirname_str: str, dtype_name: str) -> np.ndarray:
    """Load TIFF stack with LRU caching to avoid redundant disk reads.

    maxsize=1000 allows ~75 GB in-memory cache (assuming 75 MB per series).
    """
    return _load_tiff_stack_impl(dirname_str, dtype_name)


def load_tiff_stack(dirname: str | Path, dtype: np.dtype | type | None = None, use_cache: bool = True) -> np.ndarray:
    """Load a TIFF stack saved as ``slide0000.tiff``, ``slide0001.tiff``, ...

    dtype: target dtype. If None, returns native uint16 from TIFF files.
    use_cache: if False, bypass LRU cache (useful when each series is read once).
    """

    dtype_str = "uint16" if dtype is None else np.dtype(dtype).name
    dirname_str = str(Path(dirname).resolve())
    if use_cache:
        return _load_tiff_stack_cached(dirname_str, dtype_str)
    else:
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


def load_scan_and_mask(data_root: str | Path, record: M3DSynthRecord, use_cache: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Load normalized scan and boolean manipulation mask for one record.

    Handles shape misalignment in the z-dimension:
    - pix2pix padding: mask often has +1 z-slice (trimmed);
    - General mismatch: if mask z differs from scan z, trim (if longer) or pad with
      zeros (if shorter). Y and X must match exactly.

    use_cache: if False, bypass LRU cache (use during training where each series is read once).
    """

    scan = normalize_percentile(load_tiff_stack(scan_dir(data_root, record), use_cache=use_cache))
    if record.is_real:
        mask = np.zeros(scan.shape, dtype=bool)
    else:
        mask = load_tiff_stack(label_dir(data_root, record), use_cache=use_cache) > 0

        # Align mask to scan shape if z-dimension differs
        if scan.shape != mask.shape:
            scan_z, scan_y, scan_x = scan.shape
            mask_z, mask_y, mask_x = mask.shape

            # Y and X must match exactly — this is a critical error if they don't
            if scan_y != mask_y or scan_x != mask_x:
                raise ValueError(
                    f"scan/mask spatial (y, x) mismatch for {record.img_id}: "
                    f"scan {scan.shape} != mask {mask.shape}"
                )

            # Handle z-dimension mismatch
            if mask_z > scan_z:
                # Mask is longer: trim from the end (pix2pix padding convention)
                mask = mask[:scan_z]
            elif mask_z < scan_z:
                # Mask is shorter: pad with zeros (background) at the end
                padding = scan_z - mask_z
                mask = np.pad(mask, ((0, padding), (0, 0), (0, 0)), mode="constant", constant_values=False)

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

    Boundary patches with tiny non-zero overlap receive label ``None`` from
    ``labels_from_mask`` and are skipped to avoid noisy supervision.
    Only loads mask (not scan) to minimize memory during indexing.
    """

    examples: list[PatchExample] = []
    for record_index, record in enumerate(records):
        # Load mask only (much smaller) to get shape and labels, without loading scan
        if record.is_real:
            # Real data: infer shape from scan alone without loading all pixels
            scan = load_tiff_stack(scan_dir(data_root, record), use_cache=False)
            mask = np.zeros(scan.shape, dtype=bool)
            del scan  # Free memory immediately
        else:
            # Pix2pix: load mask as bool to save memory
            mask = (load_tiff_stack(label_dir(data_root, record), use_cache=False) > 0).astype(bool)
            # Get shape from mask, align to scan shape if needed
            scan_shape_z = mask.shape[0]  # Assume mask z is correct after alignment
            # For now, use mask shape as proxy (will be corrected in __getitem__)

        grid = PatchGrid(mask.shape, patch_shape=patch_shape, stride=stride)
        labels = labels_from_mask(mask, grid, positive_overlap_fraction=positive_overlap_fraction)
        for coord, slc, label in zip(grid.iter_coords(), grid.iter_slices(), labels):
            if label is None:
                continue
            examples.append(
                PatchExample(
                    record_index=record_index,
                    coord=coord,
                    label=int(label),
                    soft_score=patch_overlap_fraction(mask[slc]),
                )
            )
    return examples


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
        examples: Sequence[PatchExample] | None = None,
        patch_shape: tuple[int, int, int] = (32, 32, 32),
        stride: tuple[int, int, int] = (32, 32, 32),
        positive_overlap_fraction: float = 0.05,
    ) -> None:
        """Store records and optionally build patch index from masks."""

        self.records = list(records)
        self.data_root = Path(data_root)
        self.patch_shape = patch_shape
        self.stride = stride
        self.examples = list(examples) if examples is not None else build_patch_examples(
            self.records,
            self.data_root,
            patch_shape=patch_shape,
            stride=stride,
            positive_overlap_fraction=positive_overlap_fraction,
        )

    def __len__(self) -> int:
        """Return number of supervised patch examples."""

        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        """Load one patch and return torch tensors plus metadata."""

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on train env
            raise RuntimeError("PyTorch is required to use M3DSynthPatchDataset") from exc

        example = self.examples[index]
        record = self.records[example.record_index]
        scan, _ = load_scan_and_mask(self.data_root, record, use_cache=False)
        z, y, x = example.coord
        dz, dy, dx = self.patch_shape
        patch = scan[z : z + dz, y : y + dy, x : x + dx]
        # Pad patch if it's smaller than patch_shape (happens at boundaries)
        if patch.shape != self.patch_shape:
            pad_z = dz - patch.shape[0]
            pad_y = dy - patch.shape[1]
            pad_x = dx - patch.shape[2]
            patch = np.pad(patch, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant", constant_values=0)
        # Channel dimension is added here so model input becomes (B, 1, D, H, W).
        image = torch.from_numpy(patch[None].astype(np.float32, copy=False))
        label = torch.tensor([float(example.label)], dtype=torch.float32)
        soft_score = torch.tensor([float(example.soft_score)], dtype=torch.float32)
        return {"image": image, "label": label, "soft_score": soft_score}


def _read_split_table(sets_csv: Path) -> dict[str, str]:
    """Read ``sets.csv`` into ``orig_id -> split`` mapping."""

    with sets_csv.open(newline="") as handle:
        return {row["orig_id"]: row["set"] for row in csv.DictReader(handle)}
