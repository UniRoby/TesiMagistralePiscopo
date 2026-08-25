"""Full-volume validation for the voxel-level 3D U-Net baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .dataset import cross_generator_records, label_dir, read_records, scan_dir
from .inference import infer_segmentation_heatmap
from .model import UNet3DModelConfig, build_unet3d
from .volume_io import align_mask_to_scan, load_label_mask, load_normalized_scan


def score_histograms(mask: np.ndarray, heatmap: np.ndarray, bins: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Return positive and negative score histograms on ``[0, 1]``."""

    if mask.shape != heatmap.shape or bins < 2:
        raise ValueError("mask and heatmap must share a shape and bins must be >= 2")
    truth = np.asarray(mask, dtype=bool)
    scores = np.asarray(heatmap, dtype=np.float32)
    positive = np.zeros(bins, dtype=np.int64)
    negative = np.zeros(bins, dtype=np.int64)
    for truth_slice, score_slice in zip(truth, scores):
        positive += np.histogram(score_slice[truth_slice], bins=bins, range=(0.0, 1.0))[0]
        negative += np.histogram(score_slice[~truth_slice], bins=bins, range=(0.0, 1.0))[0]
    return positive, negative


def histogram_metrics(positive: np.ndarray, negative: np.ndarray, index: int) -> dict[str, float]:
    """Compute threshold metrics from ascending score histograms."""

    tp = int(positive[index:].sum())
    fp = int(negative[index:].sum())
    fn = int(positive[:index].sum())
    return {
        "dice": 2.0 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
    }


def auc_ap_from_histograms(positive: np.ndarray, negative: np.ndarray) -> tuple[float, float]:
    """Approximate voxel ROC-AUC and AP from fixed-width score histograms."""

    n_positive, n_negative = int(positive.sum()), int(negative.sum())
    if not n_positive or not n_negative:
        return float("nan"), float("nan")
    tp = np.cumsum(positive[::-1], dtype=np.float64)
    fp = np.cumsum(negative[::-1], dtype=np.float64)
    recall = tp / n_positive
    false_positive_rate = fp / n_negative
    precision = tp / np.maximum(tp + fp, 1)
    integrate = getattr(np, "trapezoid", np.trapz)
    auc = float(integrate(np.r_[0.0, recall], np.r_[0.0, false_positive_rate]))
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return auc, ap


def best_histogram_threshold(positive: np.ndarray, negative: np.ndarray) -> tuple[int, float, float]:
    """Choose the highest histogram threshold tied for best micro Dice."""

    scores = np.asarray([
        histogram_metrics(positive, negative, index)["dice"] for index in range(len(positive))
    ])
    best_value = float(np.max(scores))
    index = int(np.flatnonzero(np.isclose(scores, best_value))[-1])
    return index, index / len(positive), best_value


def _unique_scan_records(records, data_root: Path):
    seen, unique = set(), []
    for record in records:
        key = str(scan_dir(data_root, record).resolve())
        if key not in seen:
            seen.add(key)
            unique.append(record)
    return unique


def _macro(rows: list[dict]) -> dict:
    metrics = ("dice", "iou", "precision", "recall", "voxel_auc", "voxel_ap")
    return {
        metric: float(np.nanmean([float(row[metric]) for row in rows]))
        for metric in metrics
        if rows
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-volumes", type=int)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - train dependency
        raise RuntimeError("PyTorch is required for U-Net evaluation") from exc
    from .train import subset_records
    args = parse_args()
    try:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        payload = torch.load(args.checkpoint, map_location="cpu")
    config = payload["config"]
    model_cfg = config.get("model", {})
    model = build_unet3d(UNet3DModelConfig(
        in_channels=int(model_cfg.get("in_channels", 1)),
        out_channels=int(model_cfg.get("out_channels", 1)),
        base_channels=int(model_cfg.get("base_channels", 16)),
        input_mode=str(model_cfg.get("input_mode", "ct")),
    ))
    model.load_state_dict(payload["model_state_dict"])
    model.to(args.device)

    metadata_dir = Path(config.get("metadata_dir", "metadata/m3dsynth"))
    split_cfg = config["split"]
    groups = cross_generator_records(
        read_records(metadata_dir), train_mods=["pix2pix"], valid_mods=["pix2pix"],
        test_mods=split_cfg.get("test_mods", ["cycle", "diffusion"]),
        train_splits=(split_cfg.get("train_set", "train"),),
        valid_splits=(split_cfg.get("valid_set", "valid"),),
        test_splits=(split_cfg.get("test_set", "test"),), include_real=True,
    )
    seed = int(config.get("training", {}).get("seed", 21))
    configured_limit = config.get("data", {}).get("max_valid_records")
    records = subset_records(groups["valid"], args.max_volumes or configured_limit, seed)
    data_root = Path(args.data_root)
    records = _unique_scan_records(records, data_root)
    patch_shape = tuple(config.get("patches", {}).get("patch_shape", (64, 64, 64)))
    stride = tuple(size // 2 for size in patch_shape)

    try:
        from tqdm import tqdm
        iterator = tqdm(records, desc="full-volume validation", unit="vol")
    except ImportError:  # pragma: no cover - optional progress only
        iterator = records
    evaluated = []
    total_positive = np.zeros(1000, dtype=np.int64)
    total_negative = np.zeros(1000, dtype=np.int64)
    for record in iterator:
        volume = load_normalized_scan(data_root, record)
        truth = (
            np.zeros(volume.shape, dtype=bool) if record.is_real else
            align_mask_to_scan(load_label_mask(label_dir(data_root, record)), volume.shape, record.img_id)
        )
        heatmap = infer_segmentation_heatmap(
            model, volume, patch_shape=patch_shape, stride=stride,
            batch_size=args.batch_size, device=args.device,
        )
        positive, negative = score_histograms(truth, heatmap)
        total_positive += positive
        total_negative += negative
        auc, ap = auc_ap_from_histograms(positive, negative)
        evaluated.append({
            "img_id": record.img_id, "type": record.ty, "generator": record.mod,
            "orig_id": record.orig_id, "is_real": record.is_real,
            "max_score": float(np.max(heatmap)), "voxel_auc": auc, "voxel_ap": ap,
            "positive_histogram": positive, "negative_histogram": negative,
        })

    if not evaluated or not total_positive.sum():
        raise SystemExit("Validation requires at least one manipulated volume with positive mask voxels.")
    threshold_index, threshold, validation_dice = best_histogram_threshold(total_positive, total_negative)
    for row in evaluated:
        row.update(histogram_metrics(row.pop("positive_histogram"), row.pop("negative_histogram"), threshold_index))
        row["predicted_manipulated"] = bool(row["max_score"] >= threshold)
    manipulated = [row for row in evaluated if not row["is_real"]]
    real = [row for row in evaluated if row["is_real"]]
    summary = {
        "experiment_name": config.get("experiment_name", Path(args.checkpoint).parent.name),
        "checkpoint": str(Path(args.checkpoint)),
        "seed": seed,
        "input_mode": model_cfg.get("input_mode", "ct"),
        "loss": config.get("training", {}).get("loss", "bce_dice"),
        "threshold": threshold,
        "threshold_bins": 1000,
        "voxel_auc_ap": "histogram approximation with 1000 fixed bins",
        "validation_micro_dice": validation_dice,
        "n_manipulated": len(manipulated), "n_real": len(real),
        "real_volume_false_positive_rate": float(np.mean([row["predicted_manipulated"] for row in real])) if real else None,
        "macro_manipulated": _macro(manipulated),
        "by_type": {ty: _macro([row for row in manipulated if row["type"] == ty]) for ty in ("inj", "rem")},
    }
    output_dir = Path(args.output_dir or Path(args.checkpoint).with_suffix("").with_name("full_volume_validation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "volumes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(evaluated[0]))
        writer.writeheader()
        writer.writerows(evaluated)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "calibration.json").write_text(json.dumps({"localization": {"threshold": threshold}}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
