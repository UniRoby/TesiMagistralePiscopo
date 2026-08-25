"""Audit forensic traces between paired pristine and CT-GAN volumes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .dataset import label_dir, load_tiff_stack, read_records, scan_dir
from .volume_io import align_mask_to_scan, load_label_mask


def select_audit_records(records, mod: str, max_records: int, seed: int = 21):
    """Select a deterministic, manipulation-type-balanced audit subset."""

    if max_records < 1:
        raise ValueError("max_records must be positive")
    rng = np.random.default_rng(seed)
    groups = {
        ty: [record for record in records if record.mod == mod and record.ty == ty]
        for ty in ("inj", "rem")
    }
    selected = []
    for ty, count in (("inj", (max_records + 1) // 2), ("rem", max_records // 2)):
        candidates = groups[ty]
        if not candidates:
            continue
        indices = rng.permutation(len(candidates))[:count]
        selected.extend(candidates[int(index)] for index in indices)
    return sorted(selected, key=lambda record: (record.ty, record.img_id))


def _region_stats(residual: np.ndarray, region: np.ndarray) -> dict[str, float]:
    voxels = int(np.count_nonzero(region))
    if voxels == 0:
        return {"voxels": 0, "changed_fraction": 0.0, "mean_abs": 0.0, "rms": 0.0}
    changed = 0
    absolute_sum = squared_sum = 0.0
    for residual_slice, region_slice in zip(residual, region):
        values = residual_slice[region_slice].astype(np.float64, copy=False)
        changed += int(np.count_nonzero(values))
        absolute_sum += float(np.abs(values).sum())
        squared_sum += float(np.square(values).sum())
    return {
        "voxels": voxels,
        "changed_fraction": changed / voxels,
        "mean_abs": absolute_sum / voxels,
        "rms": float(np.sqrt(squared_sum / voxels)),
    }


def residual_region_stats(pristine: np.ndarray, manipulated: np.ndarray, mask: np.ndarray) -> dict[str, dict[str, float]]:
    """Measure residual energy inside the target, two outer shells and background."""

    if pristine.shape != manipulated.shape or pristine.shape != mask.shape:
        raise ValueError("pristine, manipulated and mask must have the same shape")
    residual = manipulated.astype(np.float32) - pristine.astype(np.float32)
    dilated_one = _binary_dilation(mask, iterations=1)
    dilated_five = _binary_dilation(mask, iterations=5)
    regions = {
        "inside": mask,
        "ring_1": np.logical_and(dilated_one, ~mask),
        "ring_2_5": np.logical_and(dilated_five, ~dilated_one),
        "background": ~dilated_five,
    }
    return {name: _region_stats(residual, region) for name, region in regions.items()}


def highpass_residual(volume: np.ndarray) -> np.ndarray:
    """Return a NumPy equivalent of the model's fixed 3x3x3 high-pass channel."""

    volume = np.asarray(volume, dtype=np.float32)
    padded = np.pad(volume, 1, mode="reflect")
    local_sum = np.zeros_like(volume, dtype=np.float32)
    for z in range(3):
        for y in range(3):
            for x in range(3):
                local_sum += padded[z : z + volume.shape[0], y : y + volume.shape[1], x : x + volume.shape[2]]
    return volume - local_sum / 27.0


def _binary_dilation(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Dilate a 3D mask with six-connectivity using NumPy only."""

    result = np.asarray(mask, dtype=bool)
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant")
        z, y, x = result.shape
        result = (
            padded[1 : z + 1, 1 : y + 1, 1 : x + 1]
            | padded[0:z, 1 : y + 1, 1 : x + 1] | padded[2 : z + 2, 1 : y + 1, 1 : x + 1]
            | padded[1 : z + 1, 0:y, 1 : x + 1] | padded[1 : z + 1, 2 : y + 2, 1 : x + 1]
            | padded[1 : z + 1, 1 : y + 1, 0:x] | padded[1 : z + 1, 1 : y + 1, 2 : x + 2]
        )
    return result


def _crop_around_mask(mask: np.ndarray, margin: int = 16) -> tuple[slice, slice, slice]:
    points = np.argwhere(mask)
    if not len(points):
        raise ValueError("manipulation mask is empty")
    lower = np.maximum(points.min(axis=0) - margin, 0)
    upper = np.minimum(points.max(axis=0) + margin + 1, mask.shape)
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))  # type: ignore[return-value]


def _normalize8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(image, (1, 99))
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    return (np.clip((image - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def _save_panel(path: Path, pristine: np.ndarray, manipulated: np.ndarray, residual: np.ndarray, center) -> None:
    from PIL import Image, ImageDraw

    z, y, x = (int(value) for value in center)
    views = (
        (pristine[z], manipulated[z], np.abs(residual[z])),
        (pristine[:, y, :], manipulated[:, y, :], np.abs(residual[:, y, :])),
        (pristine[:, :, x], manipulated[:, :, x], np.abs(residual[:, :, x])),
    )
    rows = []
    for view in views:
        images = [Image.fromarray(_normalize8(item)).resize((256, 256)) for item in view]
        row = Image.new("L", (768, 256))
        for index, image in enumerate(images):
            row.paste(image, (index * 256, 0))
        rows.append(row)
    panel = Image.new("L", (768, 768))
    for index, row in enumerate(rows):
        panel.paste(row, (0, index * 256))
    draw = ImageDraw.Draw(panel)
    draw.text((5, 5), "columns: pristine | manipulated | absolute residual", fill=255)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)


def _spacing_table(metadata_dir: Path) -> dict[tuple[str, str], tuple[float, float, float]]:
    with (metadata_dir / "LIDC.csv").open(newline="", encoding="utf-8") as handle:
        return {
            (row["orig_id"], row["sdir_id"]): tuple(float(row[f"spacing_{axis}"]) for axis in "zyx")
            for row in csv.DictReader(handle)
        }


def _aggregate(rows: list[dict], key: str | None = None) -> dict:
    groups = {"overall": rows} if key is None else {
        str(value): [row for row in rows if row[key] == value]
        for value in sorted({row[key] for row in rows})
    }
    return {
        name: {
            "n_records": len(items),
            **{
                metric: float(np.mean([float(item[metric]) for item in items]))
                for metric in (
                    "inside_rms", "ring_1_rms", "ring_2_5_rms", "background_rms",
                    "inside_changed_fraction", "outside_changed_fraction", "highpass_difference_rms",
                )
            },
        }
        for name, items in groups.items()
        if items
    }


def audit_record(record, data_root: Path, spacing, panel_path: Path | None = None) -> dict:
    """Audit one manipulated record against its paired real source scan."""

    manipulated = load_tiff_stack(scan_dir(data_root, record))
    pristine_record = replace(record, mod="real")
    pristine = load_tiff_stack(scan_dir(data_root, pristine_record))
    raw_mask = load_label_mask(label_dir(data_root, record))
    base = {
        "img_id": record.img_id, "type": record.ty, "generator": record.mod,
        "orig_id": record.orig_id, "sdir_id": record.sdir_id,
        "scan_shape": "x".join(map(str, manipulated.shape)),
        "pristine_shape": "x".join(map(str, pristine.shape)),
        "raw_mask_shape": "x".join(map(str, raw_mask.shape)),
        "mask_z_delta": int(raw_mask.shape[0] - manipulated.shape[0]),
    }
    current_spacing = spacing.get((record.orig_id, record.sdir_id))
    if current_spacing is None:
        return {**base, "status": "missing_spacing"}
    base.update({f"spacing_{axis}": current_spacing[index] for index, axis in enumerate("zyx")})
    base["spacing_group"] = f"z={current_spacing[0]:g}"
    if pristine.shape != manipulated.shape or raw_mask.shape[1:] != manipulated.shape[1:]:
        return {**base, "status": "geometry_mismatch"}
    mask = align_mask_to_scan(raw_mask, manipulated.shape, record.img_id)
    if not np.any(mask):
        return {**base, "status": "empty_mask"}

    residual = manipulated.astype(np.float32) - pristine.astype(np.float32)
    region = residual_region_stats(pristine, manipulated, mask)
    crop = _crop_around_mask(mask)
    hp_difference = highpass_residual(manipulated[crop]) - highpass_residual(pristine[crop])
    points = np.argwhere(mask)
    mask_center = points.mean(axis=0)
    coord_distance = float(np.linalg.norm(mask_center - np.asarray(record.coord)))
    total_changed = int(np.count_nonzero(residual))
    changed_inside = int(np.count_nonzero(residual[mask]))
    outside = (total_changed - changed_inside) / max(total_changed, 1)
    row = {
        **base, "status": "ok", "coord_mask_center_distance_vox": coord_distance,
        "outside_changed_fraction": float(outside),
        "highpass_difference_rms": float(np.sqrt(np.mean(hp_difference.astype(np.float64) ** 2))),
    }
    for name, stats in region.items():
        for metric, value in stats.items():
            row[f"{name}_{metric}"] = value
    if panel_path is not None:
        _save_panel(panel_path, pristine, manipulated, residual, np.rint(mask_center))
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--metadata-dir", default="metadata/m3dsynth")
    parser.add_argument("--mod", default="pix2pix")
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--output-dir", default="outputs/ctgan_trace_audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root, metadata_dir, output_dir = Path(args.data_root), Path(args.metadata_dir), Path(args.output_dir)
    selected = select_audit_records(read_records(metadata_dir), args.mod, args.max_records, args.seed)
    spacing = _spacing_table(metadata_dir)
    try:
        from tqdm import tqdm
        iterator = tqdm(selected, desc="auditing CT-GAN pairs", unit="vol")
    except ImportError:  # pragma: no cover - optional progress only
        iterator = selected
    rows = []
    for index, record in enumerate(iterator):
        panel = output_dir / "examples" / f"{record.img_id}.png" if index < 6 else None
        try:
            rows.append(audit_record(record, data_root, spacing, panel))
        except (FileNotFoundError, ValueError) as exc:
            rows.append({"img_id": record.img_id, "type": record.ty, "generator": record.mod, "status": f"error: {exc}"})
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    valid = [row for row in rows if row.get("status") == "ok"]
    report = {
        "requested_records": args.max_records,
        "selected_records": len(selected),
        "valid_records": len(valid),
        "errors": len(rows) - len(valid),
        "by_type": _aggregate(valid, "type") if valid else {},
        "by_spacing_z": _aggregate(valid, "spacing_group") if valid else {},
        "overall": _aggregate(valid).get("overall", {}) if valid else {},
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
