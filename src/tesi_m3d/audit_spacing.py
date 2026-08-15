"""Audit physical patch dimensions using the official M3Dsynth spacing table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def physical_patch_geometry(
    spacing: tuple[float, float, float],
    patch_shape: tuple[int, int, int],
    target_mm: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return patch size in mm and voxel dimensions corresponding to target_mm."""

    spacing_array = np.asarray(spacing, dtype=np.float64)
    patch_array = np.asarray(patch_shape, dtype=np.float64)
    if spacing_array.shape != (3,) or patch_array.shape != (3,) or np.any(spacing_array <= 0):
        raise ValueError("spacing and patch_shape must contain three positive values")
    if target_mm <= 0:
        raise ValueError("target_mm must be positive")
    return tuple(spacing_array * patch_array), tuple(float(target_mm) / spacing_array)


def _stats(values) -> dict[str, float]:
    values = np.asarray(list(values), dtype=np.float64)
    return {
        "min": float(np.min(values)), "median": float(np.median(values)),
        "mean": float(np.mean(values)), "max": float(np.max(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def summarize_spacing_rows(rows: list[dict]) -> dict:
    """Summarize spacing and physical patch dimensions overall and by split/generator."""

    def summary(items: list[dict]) -> dict:
        return {
            "n_records": len(items),
            **{
                name: _stats(item[name] for item in items)
                for name in (
                    "spacing_z", "spacing_y", "spacing_x",
                    "patch_mm_z", "patch_mm_y", "patch_mm_x",
                    "target_vox_z", "target_vox_y", "target_vox_x",
                )
            },
        }

    if not rows:
        raise ValueError("cannot summarize an empty spacing audit")
    groups = sorted({(str(row["split"]), str(row["generator"])) for row in rows})
    return {
        "overall": summary(rows),
        "by_split_generator": {
            f"{split}:{generator}": summary([
                row for row in rows if row["split"] == split and row["generator"] == generator
            ])
            for split, generator in groups
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default="metadata/m3dsynth")
    parser.add_argument("--patch-shape", nargs=3, type=int, default=(32, 32, 32), metavar=("Z", "Y", "X"))
    parser.add_argument("--target-mm", type=float, default=32.0)
    parser.add_argument("--out-dir", default="outputs/spacing_audit")
    return parser.parse_args()


def main() -> None:
    from .dataset import read_records

    args = parse_args()
    metadata_dir = Path(args.metadata_dir)
    spacing = {}
    with (metadata_dir / "LIDC.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            spacing[(row["orig_id"], row["sdir_id"])] = tuple(
                float(row[name]) for name in ("spacing_z", "spacing_y", "spacing_x")
            )
    rows, missing = [], []
    for record in read_records(metadata_dir):
        current = spacing.get((record.orig_id, record.sdir_id))
        if current is None:
            missing.append(f"{record.orig_id}:{record.sdir_id}")
            continue
        patch_mm, target_vox = physical_patch_geometry(current, tuple(args.patch_shape), args.target_mm)
        rows.append({
            "img_id": record.img_id, "orig_id": record.orig_id, "sdir_id": record.sdir_id,
            "split": record.split or "unknown", "generator": record.mod,
            **{f"spacing_{axis}": current[index] for index, axis in enumerate("zyx")},
            **{f"patch_mm_{axis}": patch_mm[index] for index, axis in enumerate("zyx")},
            **{f"target_vox_{axis}": target_vox[index] for index, axis in enumerate("zyx")},
        })
    if missing:
        raise ValueError(f"missing spacing for {len(missing)} records; first entries: {missing[:5]}")

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "patch_shape_voxels": list(args.patch_shape),
        "target_physical_side_mm": args.target_mm,
        "unique_source_series": summarize_spacing_rows(list({
            (row["orig_id"], row["sdir_id"]): row for row in rows
        }.values()))["overall"],
        **summarize_spacing_rows(rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))
    print(f"Spacing audit written to {output_dir}")


if __name__ == "__main__":
    main()
