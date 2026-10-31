"""Compare paired real/pix2pix CTs inside, near and far from the edit mask."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .audit_intensity import IntensityAuditor, _resolve_path
from .dataset import label_dir, scan_dir
from .volume_io import SLICE_TEMPLATE, align_mask_to_scan, count_contiguous_slices, load_label_mask


def _near_mask(mask: np.ndarray, radius: int = 4) -> np.ndarray:
    """Return a cube-shaped voxel neighborhood, excluding the edit itself."""

    expanded = mask.copy()
    for axis in range(3):
        grown = expanded.copy()
        for shift in range(1, radius + 1):
            before = [slice(None)] * 3
            after = [slice(None)] * 3
            before[axis] = slice(shift, None)
            after[axis] = slice(None, -shift)
            grown[tuple(before)] |= expanded[tuple(after)]
            grown[tuple(after)] |= expanded[tuple(before)]
        expanded = grown
    return expanded & ~mask


def _add_region(stats: dict, delta: np.ndarray, region: np.ndarray) -> None:
    absolute = np.abs(delta)
    values = absolute[region]
    stats["voxels"] += int(values.size)
    stats["changed_voxels"] += int(np.count_nonzero(values))
    stats["absolute_difference_sum"] += int(values.sum(dtype=np.int64))
    stats["max_absolute_difference"] = max(stats["max_absolute_difference"], int(values.max()) if values.size else 0)
    for axis in (0, 1):
        first = [slice(None)] * 2
        second = [slice(None)] * 2
        first[axis] = slice(1, None)
        second[axis] = slice(None, -1)
        first, second = tuple(first), tuple(second)
        edges = region[first] & region[second]
        stats["gradient_edges"] += int(np.count_nonzero(edges))
        stats["gradient_difference_sum"] += int(np.abs(delta[first] - delta[second])[edges].sum(dtype=np.int64))


def _result(stats: dict) -> dict:
    count = stats["voxels"]
    edges = stats["gradient_edges"]
    return {
        "voxels": count,
        "changed_voxels": stats["changed_voxels"],
        "changed_fraction": stats["changed_voxels"] / count if count else None,
        "mean_absolute_difference": stats["absolute_difference_sum"] / count if count else None,
        "mean_absolute_difference_on_changed": stats["absolute_difference_sum"] / stats["changed_voxels"] if stats["changed_voxels"] else None,
        "max_absolute_difference": stats["max_absolute_difference"],
        "mean_gradient_difference_xy": stats["gradient_difference_sum"] / edges if edges else None,
    }


def _save_preview(path: Path, real: np.ndarray, fake: np.ndarray, mask: np.ndarray, near: np.ndarray) -> None:
    from PIL import Image, ImageDraw

    source = real[real > 0]
    low, high = np.percentile(source, (1, 99)) if source.size else (0, 1)
    high = max(high, low + 1)
    def gray(plane):
        return np.clip((plane.astype(np.float32) - low) * (255 / (high - low)), 0, 255).astype(np.uint8)

    difference = np.abs(fake.astype(np.int32) - real.astype(np.int32))
    nonzero = difference[difference > 0]
    scale = max(1, float(np.percentile(nonzero, 99))) if nonzero.size else 1
    difference = np.clip(difference * (255 / scale), 0, 255).astype(np.uint8)
    regions = np.zeros(mask.shape, dtype=np.uint8)
    regions[near] = 100
    regions[mask] = 255
    height, width = real.shape
    canvas = Image.new("L", (4 * width, height + 20))
    for index, plane in enumerate((gray(real), gray(fake), difference, regions)):
        canvas.paste(Image.fromarray(plane), (index * width, 20))
    draw = ImageDraw.Draw(canvas)
    for index, title in enumerate(("real", "pix2pix", "absolute difference (scaled)", "mask white / near gray")):
        draw.text((index * width + 4, 3), title, fill=255)
    canvas.save(path)


class LocalArtifactAuditor(IntensityAuditor):
    """Screen paired TIFF stacks for changes inside and outside the annotation."""

    def run(self, output_dir: str | Path) -> dict:
        try:
            import tifffile
        except ImportError as exc:
            raise RuntimeError("tifffile is required to audit TIFF stacks") from exc

        output_dir = _resolve_path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        preview_dir = output_dir / "previews"
        preview_dir.mkdir(exist_ok=True)
        pairs = self._pairs()
        rows = []
        fields = ["img_id", "region", "voxels", "changed_voxels", "changed_fraction",
                  "mean_absolute_difference", "mean_absolute_difference_on_changed",
                  "max_absolute_difference", "mean_gradient_difference_xy"]
        with (output_dir / "regions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for number, (real_record, fake_record) in enumerate(pairs, 1):
                real_dir = scan_dir(self.data_root, real_record, self.real_scan_root)
                fake_dir = scan_dir(self.data_root, fake_record)
                real_depth = count_contiguous_slices(real_dir)
                fake_depth = count_contiguous_slices(fake_dir)
                if not real_depth or real_depth != fake_depth:
                    raise ValueError(f"scan depth mismatch for {fake_record.img_id}: {real_depth} != {fake_depth}")
                first_real = tifffile.imread(real_dir / SLICE_TEMPLATE.format(0))
                first_fake = tifffile.imread(fake_dir / SLICE_TEMPLATE.format(0))
                if first_real.ndim != 2 or first_fake.shape != first_real.shape:
                    raise ValueError(f"scan shape mismatch for {fake_record.img_id}")
                shape = (real_depth, *first_real.shape)
                mask = align_mask_to_scan(load_label_mask(label_dir(self.data_root, fake_record)), shape, fake_record.img_id)
                if not mask.any():
                    raise ValueError(f"empty manipulation mask for {fake_record.img_id}")
                near = _near_mask(mask)
                preview_index = int(np.argmax(np.count_nonzero(mask, axis=(1, 2))))
                stats = {name: {key: 0 for key in ("voxels", "changed_voxels", "absolute_difference_sum",
                                                   "max_absolute_difference", "gradient_edges", "gradient_difference_sum")}
                         for name in ("mask", "near", "far")}
                outside_preview = None
                largest_outside_difference = -1
                for index in range(real_depth):
                    real = first_real if index == 0 else tifffile.imread(real_dir / SLICE_TEMPLATE.format(index))
                    fake = first_fake if index == 0 else tifffile.imread(fake_dir / SLICE_TEMPLATE.format(index))
                    if real.shape != first_real.shape or fake.shape != real.shape or real.dtype != np.uint16 or fake.dtype != np.uint16:
                        raise ValueError(f"invalid TIFF slice shape/dtype for {fake_record.img_id}, z={index}")
                    delta = fake.astype(np.int32) - real.astype(np.int32)
                    tissue = (real > 0) & (fake > 0)
                    regions = {"mask": mask[index], "near": near[index] & tissue,
                               "far": ~(mask[index] | near[index]) & tissue}
                    for name, region in regions.items():
                        _add_region(stats[name], delta, region)
                    outside_difference = int(np.abs(delta[regions["near"] | regions["far"]]).sum(dtype=np.int64))
                    if outside_difference > largest_outside_difference:
                        largest_outside_difference = outside_difference
                        outside_preview = (index, real.copy(), fake.copy(), mask[index].copy(), near[index].copy())
                    if index == preview_index:
                        _save_preview(preview_dir / f"{fake_record.img_id}.png", real, fake, mask[index], near[index])
                if outside_preview is not None and outside_preview[0] != preview_index:
                    _, real, fake, mask_plane, near_plane = outside_preview
                    _save_preview(preview_dir / f"{fake_record.img_id}_outside.png", real, fake, mask_plane, near_plane)
                for name, values in stats.items():
                    row = {"img_id": fake_record.img_id, "region": name, **_result(values)}
                    writer.writerow(row)
                    rows.append(row)
                handle.flush()
                print(f"[{number}/{len(pairs)}] completed {fake_record.img_id}", flush=True)

        summary = {
            "n_pairs": len(pairs), "split": self.split, "seed": self.seed,
            "near_radius_voxels": 4,
            "regions": {
                name: {
                    metric: float(np.mean([row[metric] for row in rows if row["region"] == name and row[metric] is not None]))
                    if any(row["region"] == name and row[metric] is not None for row in rows) else None
                    for metric in ("changed_fraction", "mean_absolute_difference",
                                   "mean_absolute_difference_on_changed", "max_absolute_difference",
                                   "mean_gradient_difference_xy")
                }
                for name in ("mask", "near", "far")
            },
            "definitions": {
                "mask": "aligned manipulation annotation",
                "near": "outside mask, within four voxels on each axis; both TIFF values nonzero",
                "far": "outside near and mask; both TIFF values nonzero",
                "mean_gradient_difference_xy": "mean absolute difference of adjacent in-plane intensity differences in stored uint16 units",
            },
            "interpretation": "Screening only: differences outside the mask do not by themselves prove a generator artifact or a model shortcut.",
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--metadata-dir", default="metadata/m3dsynth")
    parser.add_argument("--split", choices=("train", "valid", "test"))
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--real-scan-root")
    parser.add_argument("--out-dir", default="outputs/local_artifact_audit")
    args = parser.parse_args()
    report = LocalArtifactAuditor(args.data_root, args.metadata_dir, args.split, args.max_pairs,
                                  args.seed, args.real_scan_root).run(args.out_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
