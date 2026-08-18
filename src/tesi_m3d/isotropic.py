"""Materialize an isotropic M3Dsynth TIFF corpus and matching metadata."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

from .dataset import label_dir, load_tiff_stack, read_records, scan_dir
from .lidc_conversion import save_tiff_stack
from .volume_io import align_mask_to_scan, load_label_mask


def resample_volume(volume: np.ndarray, spacing: tuple[float, float, float], target_mm: float, order: int) -> np.ndarray:
    """Resample a z-y-x volume to cubic ``target_mm`` voxels."""

    if volume.ndim != 3 or target_mm <= 0 or any(value <= 0 for value in spacing):
        raise ValueError("volume must be 3D and spacings must be positive")
    try:
        from scipy.ndimage import zoom
    except ImportError as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError("scipy is required; install the project train extra") from exc
    target_shape = np.maximum(1, np.rint(np.asarray(volume.shape) * np.asarray(spacing) / target_mm)).astype(int)
    result = zoom(volume, target_shape / np.asarray(volume.shape), order=order, prefilter=order > 1)
    return result.astype(bool if order == 0 else volume.dtype, copy=False)


def scale_coordinate(value: str, spacing: float, target_mm: float) -> str:
    """Map one voxel coordinate to the isotropic grid."""

    return str(int(round(float(value) * spacing / target_mm)))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def convert_corpus(source_root: Path, output_root: Path, metadata_dir: Path, output_metadata_dir: Path, target_mm: float = 1.0, limit: int | None = None) -> None:
    """Resample unique scans, labels, and all voxel-coordinate metadata."""

    if source_root.resolve() == output_root.resolve():
        raise ValueError("source and output roots must differ")
    lidc_rows = _read_rows(metadata_dir / "LIDC.csv")
    spacing = {
        (row["orig_id"], row["sdir_id"]): tuple(float(row[f"spacing_{axis}"]) for axis in "zyx")
        for row in lidc_rows
    }
    records = read_records(metadata_dir)
    if limit is not None:
        records = records[:limit]

    completed: set[Path] = set()
    for index, record in enumerate(records, 1):
        current = spacing.get((record.orig_id, record.sdir_id))
        if current is None:
            raise ValueError(f"missing LIDC spacing for {record.orig_id}/{record.sdir_id}")
        source_scan = scan_dir(source_root, record)
        destination_scan = scan_dir(output_root, record)
        source_volume = None
        if destination_scan not in completed:
            if destination_scan.exists() and not (destination_scan / ".complete").exists():
                raise FileExistsError(f"incomplete output exists: {destination_scan}")
            if not (destination_scan / ".complete").exists():
                source_volume = load_tiff_stack(source_scan)
                save_tiff_stack(destination_scan, resample_volume(source_volume, current, target_mm, 1))
            completed.add(destination_scan)

        if record.is_manipulated:
            destination_label = label_dir(output_root, record)
            if destination_label.exists() and not (destination_label / ".complete").exists():
                raise FileExistsError(f"incomplete output exists: {destination_label}")
            if not (destination_label / ".complete").exists():
                scan_shape = (source_volume if source_volume is not None else load_tiff_stack(source_scan)).shape
                mask = align_mask_to_scan(load_label_mask(label_dir(source_root, record)), scan_shape, record.img_id)
                save_tiff_stack(destination_label, resample_volume(mask, current, target_mm, 0))
        print(f"[{index}/{len(records)}] {record.mod}/{record.img_id}")

    data_rows = _read_rows(metadata_dir / "data.csv")
    for row in data_rows:
        current = spacing[(row["orig_id"], row["sdir_id"])]
        for axis, value in zip("zyx", current):
            row[f"coord_{axis}"] = scale_coordinate(row[f"coord_{axis}"], value, target_mm)
    _write_rows(output_metadata_dir / "data.csv", data_rows)

    centers = _read_rows(metadata_dir / "centers.csv")
    by_img = {row["img_id"]: spacing[(row["orig_id"], row["sdir_id"])] for row in data_rows}
    for row in centers:
        for axis, value in zip("zyx", by_img[row["img_id"]]):
            row[f"center_test_{axis}"] = scale_coordinate(row[f"center_test_{axis}"], value, target_mm)
    _write_rows(output_metadata_dir / "centers.csv", centers)

    for row in lidc_rows:
        for axis in "zyx":
            row[f"spacing_{axis}"] = str(float(target_mm))
    _write_rows(output_metadata_dir / "LIDC.csv", lidc_rows)
    shutil.copy2(metadata_dir / "sets.csv", output_metadata_dir / "sets.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--metadata-dir", default="metadata/m3dsynth")
    parser.add_argument("--output-metadata-dir", required=True)
    parser.add_argument("--target-mm", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    convert_corpus(Path(args.source_root), Path(args.output_root), Path(args.metadata_dir), Path(args.output_metadata_dir), args.target_mm, args.limit)


if __name__ == "__main__":
    main()
