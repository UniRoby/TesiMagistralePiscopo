"""Streaming voxel metrics, common classifier comparisons and spatial reports."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .diagnose_localization import write_csv, write_json

BINS = 65536


class ScoreHistogram:
    """AP/AUC on floor-quantized probabilities; fixed memory, including empty masks."""
    def __init__(self):
        self.positive = np.zeros(BINS, dtype=np.int64)
        self.negative = np.zeros(BINS, dtype=np.int64)

    def update(self, truth, scores, roi=None):
        truth, scores = np.asarray(truth, dtype=bool), np.asarray(scores)
        if truth.shape != scores.shape or (roi is not None and np.shape(roi) != truth.shape):
            raise ValueError("truth, scores and ROI shapes must match")
        truth, scores = truth.reshape(-1), scores.reshape(-1)
        region = None if roi is None else np.asarray(roi, dtype=bool).reshape(-1)
        for start in range(0, truth.size, 1_000_000):
            y, p = truth[start:start + 1_000_000], scores[start:start + 1_000_000]
            if region is not None:
                selected = region[start:start + 1_000_000]
                y, p = y[selected], p[selected]
            if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
                raise ValueError("expected finite probabilities in [0,1]")
            index = np.minimum((p.astype(np.float64) * BINS).astype(np.int64), BINS - 1)
            self.positive += np.bincount(index[y], minlength=BINS)
            self.negative += np.bincount(index[~y], minlength=BINS)

    def add(self, other):
        self.positive += other.positive
        self.negative += other.negative

    def curves(self):
        return self.positive[::-1].cumsum()[::-1], self.negative[::-1].cumsum()[::-1]

    def ap(self):
        n = int(self.positive.sum())
        if not n or not self.negative.any():
            return None
        tp, fp = self.curves()
        return float(np.sum((self.positive / n) * np.divide(tp, tp + fp, out=np.zeros(BINS), where=(tp + fp) > 0)))

    def auc(self):
        pos, neg = int(self.positive.sum()), int(self.negative.sum())
        if not pos or not neg:
            return None
        lower_neg = self.negative.cumsum() - self.negative
        return float(np.sum(self.positive.astype(np.float64) * (lower_neg + 0.5 * self.negative)) / (pos * neg))

    def best_threshold(self):
        if not self.positive.any():
            raise ValueError("calibration requires manipulated voxels")
        tp, fp = self.curves()
        dice = 2 * tp / np.maximum(1, self.positive.sum() + tp + fp)
        return float(np.flatnonzero(dice == dice.max())[-1] / BINS)

    def counts(self, threshold):
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("invalid probability threshold")
        index = min(BINS, int(np.ceil(threshold * BINS)))
        tp, fp = int(self.positive[index:].sum()), int(self.negative[index:].sum())
        fn = int(self.positive.sum()) - tp
        tn = int(self.negative.sum()) - fp
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def dice_counts(counts):
    tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0


def cohort_id(records):
    return hashlib.sha256(json.dumps([asdict(r) for r in records], sort_keys=True).encode()).hexdigest()


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def summarize(entries, calibration=None):
    """Calibrate only when explicitly requested by the validation caller."""
    from .evaluation import best_threshold_by_balanced_accuracy, volume_auc_ba
    total = ScoreHistogram()
    for _, hist, _ in entries:
        total.add(hist)
    labels = np.asarray([r["is_manipulated"] for r, _, _ in entries])
    scores = np.asarray([r["detection_score"] for r, _, _ in entries])
    if calibration is None:
        if np.unique(labels).size != 2:
            raise ValueError("validation calibration requires real and manipulated scans")
        detection_threshold, _ = best_threshold_by_balanced_accuracy(labels, scores)
        calibration = {"localization_threshold": total.best_threshold(), "detection_threshold": detection_threshold,
                       "detection_score": "max", "histogram_bins": BINS, "source_split": "valid"}
    if calibration.get("source_split") != "valid" or calibration.get("histogram_bins") != BINS:
        raise ValueError("use a validation calibration with matching histogram resolution")
    threshold = float(calibration["localization_threshold"])
    rows = []
    for row, hist, lung in entries:
        counts = hist.counts(threshold)
        rows.append({**row, **counts, "dice": dice_counts(counts), "ap": hist.ap(), "auc": hist.auc(),
                     "lung_ap": lung.ap() if lung is not None else None,
                     "lung_auc": lung.auc() if lung is not None else None})
    real = [r for r in rows if not r["is_manipulated"]]
    positive = [r for r in rows if r["is_manipulated"]]
    false_positive = sum(r["fp"] for r in real)
    real_voxels = sum(r["fp"] + r["tn"] for r in real)
    auc, ba = volume_auc_ba(labels, scores, calibration["detection_threshold"])
    summary = {"n_volumes": len(rows), "n_positive_volumes": len(positive), "histogram_bins": BINS,
               "ap_definition": "probabilities floor-quantized to 65536 bins; all voxels, no negative sampling; single-class scopes null",
               "dense_ap": total.ap(), "dense_auc": total.auc(), "micro_dice": dice_counts(total.counts(threshold)),
               "mean_positive_dice": _mean([r["dice"] for r in positive]),
               "mean_positive_ap": _mean([r["ap"] for r in positive]),
               "mean_positive_auc": _mean([r["auc"] for r in positive]),
               "mean_lung_ap": _mean([r["lung_ap"] for r in positive]),
               "mean_lung_auc": _mean([r["lung_auc"] for r in positive]),
               "peak_hits": sum(int(r["peak_inside_mask"]) for r in positive),
               "real_false_positive_fraction": false_positive / real_voxels if real_voxels else None,
               "detection_auc": float(auc) if np.isfinite(auc) else None, "detection_balanced_accuracy": float(ba),
               "calibration": calibration, "by_spacing_z": {}}
    for band in ("<=1mm", "1-2mm", ">2mm", "unknown"):
        selected = [r for r in positive if r["spacing_band"] == band]
        summary["by_spacing_z"][band] = {"n": len(selected), "mean_ap": _mean([r["ap"] for r in selected]),
                                               "mean_dice": _mean([r["dice"] for r in selected])}
    return summary, rows


def preview_planes(image, mask, heatmap):
    peak = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    points = np.argwhere(mask)
    center = np.rint(points.mean(0)).astype(int) if points.size else np.asarray(image.shape) // 2
    planes = []
    for location, coord in (("mask", center), ("peak", peak)):
        for axis, name in enumerate("zyx"):
            section = int(coord[axis])
            planes.append((f"{location} {name}={section}", *(np.take(a, section, axis=axis).copy() for a in (image, mask, heatmap))))
    return planes


def save_preview(path, planes, threshold):
    from PIL import Image, ImageDraw
    canvas = Image.new("RGB", (3 * 320, 6 * 344))
    draw = ImageDraw.Draw(canvas)
    for index, (label, image, mask, probability) in enumerate(planes):
        base = np.repeat((np.clip(image, 0, 1) * 255).astype(np.uint8)[..., None], 3, -1)
        truth, heat, errors = base.copy(), np.zeros_like(base), base.copy()
        truth[mask] = (0, 255, 0)
        heat[..., 0] = (probability * 255).astype(np.uint8)
        heat[..., 2] = ((1 - probability) * 255).astype(np.uint8)
        pred = probability >= threshold
        errors[pred & ~mask] = (255, 0, 0)
        errors[mask & ~pred] = (0, 0, 255)
        errors[mask & pred] = (0, 255, 0)
        for col, (title, rgb) in enumerate((("CT / mask green", truth), ("P: blue=0 red=1", heat), (f"FP red FN blue t={threshold:.5f}", errors))):
            canvas.paste(Image.fromarray(rgb).resize((320, 320), resample=Image.Resampling.NEAREST), (col * 320, index * 344 + 24))
            draw.text((col * 320 + 2, index * 344 + 3), f"{label} | {title}", fill="white")
    canvas.save(path)


def evaluate_records(model, payload, records, data_root, *, cache_dir=None, device="cpu", batch_size=2,
                     calibration=None, output_dir=None, spacing=None, real_scan_root=None):
    from .dataset import load_tiff_stack, scan_dir
    from .lung import lung_mask_from_ct
    from .segmentation import predict_checkpoint
    from .segmentation_data import checked_mask, validate_stack
    from tqdm import tqdm
    started = perf_counter()
    if not records:
        raise ValueError("empty evaluation cohort")
    entries, previews = [], []
    chosen = set(np.linspace(0, len(records) - 1, min(8, len(records)), dtype=int))
    for index, record in enumerate(tqdm(records, desc="dense localization", unit="vol")):
        tick = perf_counter()
        dirname = scan_dir(data_root, record, real_scan_root)
        validate_stack(dirname)
        raw = load_tiff_stack(dirname)
        mask = checked_mask(data_root, record, raw.shape)
        heatmap, image = predict_checkpoint(model, payload, raw, dirname, cache_dir, device=device, batch_size=batch_size)
        hist, lung_hist = ScoreHistogram(), None
        hist.update(mask, heatmap)
        coverage = None
        if record.is_manipulated:
            lung = lung_mask_from_ct(raw)
            coverage = float(lung[mask].mean())
            lung_hist = ScoreHistogram()
            lung_hist.update(mask, heatmap, lung | mask)
        current_spacing = (spacing or {}).get(f"{record.orig_id}/{record.sdir_id}")
        z = current_spacing[0] if current_spacing else None
        band = "unknown" if z is None else "<=1mm" if z <= 1 else "1-2mm" if z <= 2 else ">2mm"
        peak = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        row = {"img_id": record.img_id, "mod": record.mod, "is_manipulated": record.is_manipulated,
               "detection_score": float(heatmap.max()), "peak_inside_mask": bool(mask[peak]),
               "positive_voxels": int(mask.sum()), "lung_mask_coverage": coverage,
               "spacing_z": z, "spacing_band": band, "seconds": perf_counter() - tick}
        entries.append((row, hist, lung_hist))
        if output_dir is not None and index in chosen:
            previews.append((f"{record.mod}_{record.img_id}.png", preview_planes(image, mask, heatmap)))
    summary, rows = summarize(entries, calibration)
    summary.update(cohort_id=cohort_id(records), seconds=perf_counter() - started)
    if output_dir is not None:
        output = Path(output_dir)
        (output / "images").mkdir(parents=True, exist_ok=True)
        for name, planes in previews:
            save_preview(output / "images" / name, planes, summary["calibration"]["localization_threshold"])
        write_csv(output / "volumes.csv", rows)
        write_json(output / "metrics.json", summary)
    return summary


def compare_baseline(report, baseline, *, full=False):
    if report["cohort_id"] != baseline["cohort_id"] or report["histogram_bins"] != baseline["histogram_bins"]:
        raise ValueError("candidate and baseline must use exactly the same evaluation cohort and metrics")
    positive = report["n_positive_volumes"]
    ap, base_ap = report["mean_positive_ap"], baseline["mean_positive_ap"]
    fpr, base_fpr = report["real_false_positive_fraction"], baseline["real_false_positive_fraction"]
    checks = {"dice": report["mean_positive_dice"] >= 0.30 if full else report["micro_dice"] >= 0.10,
              "peak": report["peak_hits"] >= int(np.ceil(positive / 2)),
              "ap": ap is not None and base_ap is not None and ap >= 3 * base_ap and ap > base_ap,
              "real_fp": fpr is not None and base_fpr is not None and fpr <= base_fpr + 1e-12}
    return {"passed": bool(positive and all(checks.values())), "checks": checks,
            "baseline_cohort_id": baseline["cohort_id"]}
