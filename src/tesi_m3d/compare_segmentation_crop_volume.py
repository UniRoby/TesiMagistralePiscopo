"""Read-only comparison of fixed train crops and dense maps on the same six scans."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .compare_segmentation_batches import FixedCropPool
from .dataset import load_tiff_stack, scan_dir
from .diagnose_localization import write_csv, write_json
from .segmentation import file_digest, load_predictor, predict_checkpoint, read_calibration
from .segmentation_data import SegmentationDataset, checked_mask, manifest_records, validate_stack
from .segmentation_metrics import ScoreHistogram, dice_counts, preview_planes, save_preview
from .train import _load_torch_payload, resolve_cache_dir


def _read(path):
    return json.loads(Path(path).read_text())


def _summarize(rows, threshold):
    positive = [row for row in rows if row["positive_voxels"]]
    real = [row for row in rows if row["mod"] == "real"]
    total = {key: sum(row[key] for row in rows) for key in ("tp", "fp", "fn", "tn")}
    real_voxels = sum(row["fp"] + row["tn"] for row in real)
    return {"threshold": threshold, "micro_dice": dice_counts(total),
            "mean_positive_dice": float(np.mean([row["dice"] for row in positive])) if positive else None,
            "mean_positive_ap": float(np.mean([row["ap"] for row in positive])) if positive else None,
            "peak_hits": sum(bool(row["peak_inside_mask"]) for row in positive if "peak_inside_mask" in row),
            "real_false_positive_fraction": sum(row["fp"] for row in real) / real_voxels if real_voxels else None}


def _metrics(hist, threshold):
    counts = hist.counts(threshold)
    return {**counts, "dice": dice_counts(counts), "ap": hist.ap(), "auc": hist.auc()}


def evaluate_crops(model, pool, *, device="cpu", batch_size=2, validation_threshold=None):
    """Evaluate all 48 source crops once, without training or per-crop calibration."""
    model.eval()
    rows, validation_rows = [], []
    with torch.inference_mode():
        for start in range(0, len(pool.image), batch_size):
            logits = model(pool.image[start:start + batch_size].to(device)).float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("non-finite crop logits")
            probabilities = logits.sigmoid().cpu().numpy()[:, 0]
            for offset, probability in enumerate(probabilities):
                identity = pool.rows[start + offset]
                mask = pool.mask[start + offset, 0].numpy().astype(bool)
                hist = ScoreHistogram()
                hist.update(mask, probability)
                row = {"mod": identity["mod"], "img_id": identity["img_id"],
                       "crop_index": identity["crop_index"], "positive_voxels": int(mask.sum()),
                       "max_probability": float(probability.max()), **_metrics(hist, 0.5)}
                if validation_threshold is not None:
                    calibrated = _metrics(hist, validation_threshold)
                    validation_rows.append({**row, **calibrated})
                    row.update({f"validation_{key}": value for key, value in calibrated.items()
                                if key in ("tp", "fp", "fn", "tn", "dice")})
                rows.append(row)
    positive = [r for r in rows if r["positive_voxels"]]
    negative = [r for r in rows if not r["positive_voxels"]]
    negative_voxels = sum(r["fp"] + r["tn"] for r in negative)
    summary = _summarize(rows, 0.5)
    summary.update(n_crops=len(rows), n_positive_crops=len(positive),
                   negative_crop_fp_fraction=sum(r["fp"] for r in negative) / negative_voxels,
                   max_probability=max(r["max_probability"] for r in rows))
    return summary, rows, _summarize(validation_rows, validation_threshold) if validation_rows else None


def evaluate_dense(model, payload, records, data_root, *, output, device="cpu", batch_size=2,
                   cache_dir=None, real_scan_root=None, validation_threshold=None):
    """Evaluate original voxels with the production sliding-window predictor."""
    rows, validation_rows = [], []
    histograms = []
    (output / "images").mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records):
        dirname = scan_dir(data_root, record, real_scan_root)
        validate_stack(dirname)
        raw = load_tiff_stack(dirname)
        mask = checked_mask(data_root, record, raw.shape)
        probability, image = predict_checkpoint(model, payload, raw, dirname, cache_dir,
                                                device=device, batch_size=batch_size)
        hist = ScoreHistogram()
        hist.update(mask, probability)
        peak = tuple(int(v) for v in np.unravel_index(np.argmax(probability), probability.shape))
        row = {"mod": record.mod, "img_id": record.img_id, "positive_voxels": int(mask.sum()),
               "peak_inside_mask": bool(mask[peak]), "peak_zyx": list(peak),
               "max_probability": float(probability.max()), **_metrics(hist, 0.5)}
        if validation_threshold is not None:
            calibrated = _metrics(hist, validation_threshold)
            validation_rows.append({**row, **calibrated})
            row.update({f"validation_{key}": value for key, value in calibrated.items()
                        if key in ("tp", "fp", "fn", "tn", "dice")})
        rows.append(row)
        histograms.append(hist)
        save_preview(output / "images" / f"{index:02d}_{record.mod}_{record.img_id}.png",
                     preview_planes(image, mask, probability),
                     validation_threshold if validation_threshold is not None else 0.5)
        print(f"dense train {output.name}: {index + 1}/{len(records)} {record.mod}/{record.img_id}", flush=True)
    pooled = ScoreHistogram()
    for hist in histograms:
        pooled.add(hist)
    summary = _summarize(rows, 0.5)
    summary.update(n_volumes=len(rows), n_positive_volumes=sum(r["positive_voxels"] > 0 for r in rows),
                   dense_ap=pooled.ap(), dense_auc=pooled.auc())
    return summary, rows, _summarize(validation_rows, validation_threshold) if validation_rows else None


def run_comparison(data_root, *, real_scan_root=None, device="cpu", batch_size=2,
                   output_dir="outputs/segmentation_crop_volume_comparison", cache_dir=None,
                   source_dir="outputs/segmentation_overfit_recovery",
                   pilot_dir="outputs/segmentation_pilot_recovery",
                   clip_dir="outputs/segmentation_pilot_recovery_clip"):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output = Path(output_dir)
    sources = {"overfit": Path(source_dir) / "full_pool", "pilot": Path(pilot_dir),
               "pilot_clip": Path(clip_dir)}
    source_roots = (Path(source_dir), Path(pilot_dir), Path(clip_dir))
    if any(output.resolve() == folder.resolve() or output.resolve() in folder.resolve().parents
           or folder.resolve() in output.resolve().parents for folder in source_roots):
        raise ValueError("output directory must be outside source runs")
    if output.exists() and any(output.iterdir()):
        raise ValueError("comparison output already contains files; choose a new --output-dir")
    checkpoints = {name: folder / "best.pt" for name, folder in sources.items()}
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("trained checkpoints are required on the laboratory PC: " + ", ".join(missing))
    manifest = _read(Path(source_dir) / "manifest.json")
    source_report = _read(Path(source_dir) / "comparison.json")
    if (not source_report.get("complete") or not source_report.get("passed") or
            source_report.get("manifest_id") != manifest["manifest_id"] or
            "full_pool" not in source_report.get("arms", {})):
        raise ValueError("source comparison is incomplete or belongs to another manifest")
    records = manifest_records(manifest, "overfit")
    if len(records) != 6 or sum(r.is_real for r in records) != 2 or any(
            r not in manifest_records(manifest, "pilot_train") for r in records):
        raise ValueError("the same six overfit scans must belong to pilot_train")
    ledger_path = Path(source_dir) / "crop_pool.csv"
    with ledger_path.open(newline="", encoding="utf-8") as handle:
        ledger = list(csv.DictReader(handle))
    if len(ledger) != 48:
        raise ValueError("source crop ledger must contain 48 crops")
    baseline = _read(sources["overfit"] / "probabilities/best/report.json")
    gate = _read(sources["overfit"] / "gate.json")
    if not gate.get("passed") or gate.get("manifest_id") != manifest["manifest_id"]:
        raise ValueError("source overfit gate or manifest differs")
    payloads, calibrations = {}, {}
    for name, path in checkpoints.items():
        digest = file_digest(path)
        expected = baseline if name == "overfit" else _read(sources[name] / "evaluation/metrics.json")
        if digest != expected["checkpoint_sha256"]:
            raise ValueError(f"checkpoint differs from recorded {name} report: {path}")
        payload = _load_torch_payload(path)
        if payload.get("task") != "segmentation" or payload.get("manifest", {}).get("manifest_id") != manifest["manifest_id"]:
            raise ValueError(f"checkpoint task/manifest differs: {path}")
        if name == "overfit":
            if payload["config"]["stage"] != "overfit" or payload["config"].get("overfit_protocol") != "full_pool":
                raise ValueError("expected the full-pool overfit checkpoint")
            calibrations[name] = None
        else:
            if payload["config"]["stage"] != "pilot" or expected.get("cohort_id") != _read(Path(pilot_dir) / "evaluation/metrics.json")["cohort_id"]:
                raise ValueError("pilot checkpoint/evaluation cohort differs")
            calibrations[name] = read_calibration(sources[name] / "calibration.json", path)
        payloads[name] = {"checkpoint_sha256": digest, "config": payload["config"]}
    del payload
    for record in records:
        validate_stack(scan_dir(data_root, record, real_scan_root))
    output.mkdir(parents=True, exist_ok=True)
    report = {"task": "segmentation_crop_volume_comparison", "complete": False,
              "evaluation_only": True, "manifest_id": manifest["manifest_id"],
              "scope": "six train scans; 48 fixed source crops with repeated voxels versus dense original volumes",
              "thresholds": "0.5 for all checkpoints; pilot validation threshold additionally recorded",
              "ap_definition": "65,536-bin floor-quantized probabilities; no voxel sampling",
              "runs": {}}
    write_json(output / "report.json", report)
    for name, path in checkpoints.items():
        config = payloads[name]["config"]
        model, payload = load_predictor(path, device)
        dataset = SegmentationDataset(records, data_root, patch_shape=config["patches"]["patch_shape"],
                                      preprocessing=config["preprocessing"], seed=manifest["seed"],
                                      augment=False, fixed=True, volume_cache_size=1, real_scan_root=real_scan_root,
                                      cache_dir=resolve_cache_dir(config, cache_dir, output))
        pool = FixedCropPool(dataset)
        if name != "pilot_clip" and pool.digest != source_report["crop_pool_sha256"]:
            raise ValueError("fixed crop pool differs from the successful source comparison")
        for source, current in zip(ledger, pool.rows):
            keys = ("mod", "img_id", "crop_index", "target_voxels", "mask_sha256")
            if name != "pilot_clip":
                keys += ("image_sha256",)
            if any(str(source[key]) != str(current[key]) for key in keys):
                raise ValueError(f"regenerated {name} crop differs from source ledger")
        threshold = calibrations[name]["localization_threshold"] if calibrations[name] else None
        crop_summary, crop_rows, crop_valid = evaluate_crops(
            model, pool, device=device, batch_size=batch_size, validation_threshold=threshold)
        if name == "overfit" and (abs(crop_summary["mean_positive_dice"] - gate["positive_crop_dice"]) > 1e-4 or
                                  abs(crop_summary["negative_crop_fp_fraction"] - gate["negative_crop_fp_fraction"]) > 1e-4):
            raise ValueError("overfit crop gate was not reproduced; do not interpret dense comparison")
        del pool, dataset
        folder = output / name
        folder.mkdir(parents=True, exist_ok=True)
        write_csv(folder / "crops.csv", crop_rows)
        dense_summary, dense_rows, dense_valid = evaluate_dense(model, payload, records, data_root, output=folder,
                                                  device=device, batch_size=batch_size, cache_dir=resolve_cache_dir(config, cache_dir, output),
                                                  real_scan_root=real_scan_root, validation_threshold=threshold)
        write_csv(folder / "volumes.csv", dense_rows)
        result = {"checkpoint": str(path), "checkpoint_sha256": payloads[name]["checkpoint_sha256"],
                  "preprocessing": config["preprocessing"], "crop_fixed_0_5": crop_summary,
                  "dense_fixed_0_5": dense_summary,
                  "validation_pilot_report": str(sources[name] / "evaluation/metrics.json") if name != "overfit" else None,
                  "validation_threshold": threshold,
                  "crop_validation_threshold": crop_valid, "dense_validation_threshold": dense_valid}
        report["runs"][name] = result
        write_json(output / "report.json", report)
        del model, payload
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    report["complete"] = True
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--real-scan-root")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-dir", default="outputs/segmentation_crop_volume_comparison")
    parser.add_argument("--cache-dir")
    parser.add_argument("--source-dir", default="outputs/segmentation_overfit_recovery")
    parser.add_argument("--pilot-dir", default="outputs/segmentation_pilot_recovery")
    parser.add_argument("--clip-dir", default="outputs/segmentation_pilot_recovery_clip")
    args = parser.parse_args()
    print(json.dumps(run_comparison(args.data_root, real_scan_root=args.real_scan_root, device=args.device,
                                    batch_size=args.batch_size, output_dir=args.output_dir, cache_dir=args.cache_dir,
                                    source_dir=args.source_dir, pilot_dir=args.pilot_dir, clip_dir=args.clip_dir), indent=2))


if __name__ == "__main__":
    main()
