"""Inspect dense windows before fusion on the successful overfit's four manipulated CTs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from monai.data.utils import compute_importance_map, dense_patch_slices

from .compare_segmentation_batches import tensor_digest
from .diagnose_localization import write_csv, write_json
from .segmentation import file_digest, load_predictor
from .segmentation_data import crop_plan, load_bundle, manifest_records
from .segmentation_metrics import ScoreHistogram, dice_counts, preview_planes, save_preview
from .train_segmentation import read_json


def slices(start, shape):
    return tuple(slice(int(a), int(a + n)) for a, n in zip(start, shape))


def intersection(first, second):
    lo = np.maximum([s.start for s in first], [s.start for s in second])
    hi = np.minimum([s.stop for s in first], [s.stop for s in second])
    if np.any(hi <= lo):
        return None
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def relative(region, parent):
    return tuple(slice(s.start - p.start, s.stop - p.start) for s, p in zip(region, parent))


def metrics(mask, probability):
    hist = ScoreHistogram()
    hist.update(mask, probability)
    counts = hist.counts(0.5)
    return {**counts, "dice": dice_counts(counts), "ap": hist.ap()}


def differences(first, second, mask):
    delta = np.abs(first - second)
    return {"mean_abs_difference": float(delta.mean()), "max_abs_difference": float(delta.max()),
            "binary_disagreement": float(np.mean((first >= 0.5) != (second >= 0.5))),
            "foreground_mean_abs_difference": float(delta[mask].mean()) if mask.any() else None,
            "background_mean_abs_difference": float(delta[~mask].mean()) if (~mask).any() else None}


def inspect_windows(model, image, mask, coords, patch_shape, *, device="cpu"):
    """Fuse every MONAI-grid window contributing to the fixed-crop ROI, in FP32.

    Only ROI accumulators are allocated; this is not a full-volume evaluation.
    """
    patch = tuple(int(n) for n in patch_shape)
    if (image.shape != mask.shape or len(image.shape) != 3 or len(patch) != 3 or min(patch) < 8 or
            any(p % 8 for p in patch) or any(n < p for n, p in zip(image.shape, patch)) or not coords):
        raise ValueError("expected matching 3D arrays, positive crop coordinates and an unpadded multiple-of-eight patch")
    if not np.isfinite(image).all() or not np.isin(mask, (0, 1)).all():
        raise ValueError("expected finite image and binary mask")
    mask = mask.astype(bool, copy=False)
    fixed_regions = [slices(c, patch) for c in coords]
    if any(any(s.start < 0 or s.stop > n for s, n in zip(region, image.shape)) for region in fixed_regions):
        raise ValueError("fixed crop outside scan")
    lo = np.min(np.asarray(coords), axis=0)
    hi = np.max(np.asarray(coords) + patch, axis=0)
    points = np.argwhere(mask)
    if not points.size:
        raise ValueError("window diagnostic requires a positive mask")
    lo, hi = np.minimum(lo, points.min(0)), np.maximum(hi, points.max(0) + 1)
    roi = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    shape = tuple(int(n) for n in hi - lo)
    numerator, denominator = [np.zeros(shape, np.float32) for _ in range(2)]
    mean, m2 = [np.zeros(shape, np.float64) for _ in range(2)]
    low, high = np.full(shape, np.inf, np.float32), np.full(shape, -np.inf, np.float32)
    count = np.zeros(shape, np.int32)
    importance = compute_importance_map(patch, mode="gaussian", sigma_scale=0.125).numpy()
    model.to(device).eval()

    def predict(region):
        batch = torch.from_numpy(np.ascontiguousarray(image[region], dtype=np.float32))[None, None].to(device)
        probability = model(batch).float().sigmoid()[0, 0].cpu().numpy()
        if probability.shape != patch or not np.isfinite(probability).all():
            raise ValueError("invalid window prediction")
        return probability

    window_rows, common_rows, fixed_rows = [], [], []
    with torch.inference_mode():
        fixed = [predict(region) for region in fixed_regions]
        for i, (region, probability) in enumerate(zip(fixed_regions, fixed)):
            fixed_rows.append({"crop_index": i, **metrics(mask[region], probability)})
        interval = tuple(p if p == n else max(int(p * 0.5), 1) for p, n in zip(patch, image.shape))
        for grid_index, region in enumerate(dense_patch_slices(image.shape, patch, interval)):
            common = intersection(region, roi)
            if common is None:
                continue
            probability = predict(region)
            truth = mask[region]
            points = np.argwhere(truth)
            edge = np.minimum(points, np.asarray(patch) - 1 - points).min(1) if points.size else None
            window_rows.append({"grid_index": grid_index, "z": region[0].start, "y": region[1].start,
                                "x": region[2].start, "positive_voxels": int(truth.sum()),
                                "target_mean_edge_distance": float(edge.mean()) if edge is not None else None,
                                **metrics(truth, probability)})
            a, b = relative(common, region), relative(common, roi)
            values, weights = probability[a], importance[a]
            numerator[b] += values * weights
            denominator[b] += weights
            count[b] += 1
            delta = values - mean[b]
            mean[b] += delta / count[b]
            m2[b] += delta * (values - mean[b])
            low[b] = np.minimum(low[b], values)
            high[b] = np.maximum(high[b], values)
            for i, fixed_region in enumerate(fixed_regions):
                shared = intersection(region, fixed_region)
                if shared is None:
                    continue
                p, ref, y = probability[relative(shared, region)], fixed[i][relative(shared, fixed_region)], mask[shared]
                common_rows.append({"grid_index": grid_index, "crop_index": i, "voxels": int(y.size),
                                    "positive_voxels": int(y.sum()), **differences(p, ref, y),
                                    "window_dice": metrics(y, p)["dice"], "fixed_dice": metrics(y, ref)["dice"]})
    if np.any(count == 0) or np.any(denominator <= 0):
        raise ValueError("incomplete ROI coverage")
    fused = numerator / denominator
    std = np.sqrt(np.maximum(m2 / count, 0)).astype(np.float32)
    for i, region in enumerate(fixed_regions):
        p = fused[relative(region, roi)]
        fixed_rows[i].update({"fused_" + k: v for k, v in metrics(mask[region], p).items()})
        fixed_rows[i].update(differences(fixed[i], p, mask[region]))
    roi_mask = mask[roi]
    summary = {"roi_origin_zyx": lo.tolist(), "roi_shape": list(shape), "windows": len(window_rows),
               "windows_intersecting_mask": sum(r["positive_voxels"] > 0 for r in window_rows),
               "roi_metrics": metrics(roi_mask, fused),
               "foreground_mean_window_std": float(std[roi_mask].mean()) if roi_mask.any() else None,
               "foreground_mean_window_range": float((high - low)[roi_mask].mean()) if roi_mask.any() else None,
               "background_mean_window_std": float(std[~roi_mask].mean()) if (~roi_mask).any() else None}
    arrays = {"probability": fused, "mask": roi_mask.astype(np.uint8), "window_count": count,
              "window_min": low, "window_max": high, "window_std": std, "roi_origin_zyx": lo}
    return summary, window_rows, common_rows, fixed_rows, arrays, image[roi]


def run_diagnostic(data_root, *, real_scan_root=None, device="cpu", cache_dir=None,
                   source_dir="outputs/segmentation_overfit_recovery",
                   output_dir="outputs/segmentation_window_diagnostics"):
    source, output = Path(source_dir), Path(output_dir)
    protected = [source, Path(data_root)] + ([Path(real_scan_root)] if real_scan_root else [])
    if any(output.resolve() == p.resolve() or output.resolve() in p.resolve().parents or p.resolve() in output.resolve().parents for p in protected):
        raise ValueError("output must be separate from source runs and data")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output already contains files; choose a new directory")
    checkpoint = source / "full_pool/best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"trained checkpoint required on laboratory PC: {checkpoint}")
    manifest, original = read_json(source / "manifest.json"), read_json(source / "comparison.json")
    expected = read_json(source / "full_pool/probabilities/best/report.json")
    digest = file_digest(checkpoint)
    if (not original.get("complete") or not original.get("passed") or original.get("manifest_id") != manifest["manifest_id"] or
            digest != expected["checkpoint_sha256"]):
        raise ValueError("source comparison/checkpoint mismatch")
    model, payload = load_predictor(checkpoint, device)
    config = payload["config"]
    if (payload.get("task") != "segmentation" or payload.get("manifest", {}).get("manifest_id") != manifest["manifest_id"] or
            config.get("overfit_protocol") != "full_pool"):
        raise ValueError("expected the successful full-pool segmenter")
    with (source / "crop_pool.csv").open(newline="") as handle:
        ledger = list(csv.DictReader(handle))
    records = manifest_records(manifest, "overfit")
    if len(records) != 6 or sum(not r.is_real for r in records) != 4 or len(ledger) != 48:
        raise ValueError("expected six scans and 48 source crops")
    output.mkdir(parents=True, exist_ok=True)
    report = {"task": "segmentation_windows", "complete": False, "evaluation_only": True,
              "manifest_id": manifest["manifest_id"], "checkpoint_sha256": digest,
              "precision": "FP32 for both fixed and dense windows; no training or threshold calibration",
              "geometry": "production MONAI grid, overlap 0.5, gaussian sigma_scale 0.125",
              "scope": "ROI encloses the full mask and four fixed positive crops; all contributing windows included",
              "interpretation": "shared-voxel differences isolate context/position effects before fusion; ROI metrics are not whole-volume metrics",
              "cases": []}
    write_json(output / "report.json", report)
    try:
        for index, record in enumerate(records):
            if record.is_real:
                continue
            report["active_case"] = record.img_id
            write_json(output / "report.json", report)
            image, mask, lung, _ = load_bundle(data_root, record, cache_dir, config["preprocessing"], real_scan_root)
            patch = tuple(config["patches"]["patch_shape"])
            if any(n < p for n, p in zip(image.shape, patch)):
                raise ValueError("diagnostic expects native scans larger than the crop; padding would change coordinates")
            coords = crop_plan(mask, lung, patch, np.random.default_rng([manifest["seed"], 0, index]))[:4]
            for i, coord in enumerate(coords):
                region, entry = slices(coord, patch), ledger[index * 8 + i]
                if (entry["img_id"] != record.img_id or entry["mod"] != record.mod or int(entry["crop_index"]) != i or
                        entry["image_sha256"] != tensor_digest(torch.from_numpy(np.array(image[region], dtype=np.float32))[None]) or
                        entry["mask_sha256"] != tensor_digest(torch.from_numpy(np.array(mask[region], dtype=np.float32))[None])):
                    raise ValueError("fixed crop differs from the source ledger")
            result, windows, common, fixed, arrays, roi_image = inspect_windows(model, image, mask, coords, patch, device=device)
            folder = output / f"{record.mod}_{record.img_id}"
            folder.mkdir()
            write_csv(folder / "windows.csv", windows)
            write_csv(folder / "shared_voxels.csv", common)
            write_csv(folder / "fixed_vs_fused.csv", fixed)
            np.savez_compressed(folder / "roi.npz", **arrays)
            save_preview(folder / "fused.png", preview_planes(roi_image, arrays["mask"].astype(bool), arrays["probability"]), 0.5)
            report["cases"].append({"img_id": record.img_id, "mod": record.mod, **result})
            write_json(folder / "summary.json", report["cases"][-1])
            print(f"windows: {record.img_id}, {len(windows)} contributing windows", flush=True)
        report["complete"] = True
        report.pop("active_case", None)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--real-scan-root")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir")
    parser.add_argument("--source-dir", default="outputs/segmentation_overfit_recovery")
    parser.add_argument("--output-dir", default="outputs/segmentation_window_diagnostics")
    run_diagnostic(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
