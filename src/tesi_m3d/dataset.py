"""Dataset helpers for M3Dsynth metadata, TIFF stacks, and patch samples.

M3Dsynth data are expected under ``data_root/<mod>/scan/<img_id>/`` and
``data_root/<mod>/label/<img_id>/``. The project keeps this convention isolated
here so training, inference, and tests can share the same path logic.
"""

from __future__ import annotations

from dataclasses import dataclass
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


def load_tiff_stack(dirname: str | Path, dtype: np.dtype | type = np.float32) -> np.ndarray:
    """Load a TIFF stack saved as ``slide0000.tiff``, ``slide0001.tiff``, ..."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("Pillow is required to load M3Dsynth TIFF stacks") from exc

    dirname = Path(dirname)
    slices = []
    index = 0
    while True:
        filename = dirname / f"slide{index:04d}.tiff"
        if not filename.exists():
            break
        with Image.open(filename) as image:
            slices.append(np.asarray(image, dtype=dtype))
        index += 1
    if not slices:
        raise FileNotFoundError(f"no TIFF slices found in {dirname}")
    return np.stack(slices, axis=0)


def normalize_percentile(scan: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Normalize a CT stack to ``[0, 1]`` using robust percentiles."""

    non_zero = scan[scan > 0]
    source = non_zero if non_zero.size else scan.reshape(-1)
    lo, hi = np.percentile(source, [low, high])
    if hi <= lo:
        return np.zeros_like(scan, dtype=np.float32)
    scan = np.clip(scan.astype(np.float32), lo, hi)
    return ((scan - lo) / (hi - lo)).astype(np.float32)


def load_scan_and_mask(data_root: str | Path, record: M3DSynthRecord) -> tuple[np.ndarray, np.ndarray]:
    """Load normalized scan and boolean manipulation mask for one record."""

    scan = normalize_percentile(load_tiff_stack(scan_dir(data_root, record), dtype=np.float32))
    if record.is_real:
        mask = np.zeros(scan.shape, dtype=bool)
    else:
        mask = load_tiff_stack(label_dir(data_root, record), dtype=np.float32) > 0
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
    """

    examples: list[PatchExample] = []
    for record_index, record in enumerate(records):
        scan, mask = load_scan_and_mask(data_root, record)
        grid = PatchGrid(scan.shape, patch_shape=patch_shape, stride=stride)
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
        scan, _ = load_scan_and_mask(self.data_root, record)
        z, y, x = example.coord
        dz, dy, dx = self.patch_shape
        patch = scan[z : z + dz, y : y + dy, x : x + dx]
        # Channel dimension is added here so model input becomes (B, 1, D, H, W).
        image = torch.from_numpy(patch[None].astype(np.float32, copy=False))
        label = torch.tensor([float(example.label)], dtype=torch.float32)
        soft_score = torch.tensor([float(example.soft_score)], dtype=torch.float32)
        return {"image": image, "label": label, "soft_score": soft_score, "record": record, "coord": example.coord}


def _read_split_table(sets_csv: Path) -> dict[str, str]:
    """Read ``sets.csv`` into ``orig_id -> split`` mapping."""

    with sets_csv.open(newline="") as handle:
        return {row["orig_id"]: row["set"] for row in csv.DictReader(handle)}
