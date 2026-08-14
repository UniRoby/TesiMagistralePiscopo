"""Evaluate a trained patch classifier without reconstructing voxel heatmaps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .patches import PatchGrid


def binary_counts(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, int]:
    """Return the confusion counts for one score threshold."""

    truth = np.asarray(y_true, dtype=bool)
    prediction = np.asarray(y_score) >= float(threshold)
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_score must have the same shape")
    return {
        "tp": int(np.logical_and(truth, prediction).sum()),
        "fp": int(np.logical_and(~truth, prediction).sum()),
        "tn": int(np.logical_and(~truth, ~prediction).sum()),
        "fn": int(np.logical_and(truth, ~prediction).sum()),
    }


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    """Compute precision, recall, F1, accuracy, sensitivity and specificity."""

    tp, fp, tn, fn = (counts[key] for key in ("tp", "fp", "tn", "fn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "accuracy": float((tp + tn) / (tp + fp + tn + fn)),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float((recall + specificity) / 2),
    }


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, dict[str, float]]:
    """Find the exact observed score threshold maximizing F1 in O(n log n)."""

    truth = np.asarray(y_true, dtype=bool).reshape(-1)
    scores = np.asarray(y_score, dtype=np.float32).reshape(-1)
    if truth.size == 0 or truth.shape != scores.shape or not np.any(truth):
        raise ValueError("threshold calibration requires aligned scores and positive examples")
    order = np.argsort(scores, kind="stable")[::-1]
    sorted_truth = truth[order]
    sorted_scores = scores[order]
    tp = np.cumsum(sorted_truth)
    fp = np.cumsum(~sorted_truth)
    total_positive = int(truth.sum())
    boundaries = np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    candidate_indices = np.flatnonzero(boundaries)
    f1 = 2 * tp[candidate_indices] / (2 * tp[candidate_indices] + fp[candidate_indices] + total_positive - tp[candidate_indices])
    best_value = float(np.max(f1))
    # Equal F1 prefers the highest threshold, hence the first sorted candidate.
    index = int(candidate_indices[np.flatnonzero(np.isclose(f1, best_value))[0]])
    threshold = float(sorted_scores[index])
    return threshold, metrics_from_counts(binary_counts(truth, scores, threshold))


def topk_hits(overlap: np.ndarray, scores: np.ndarray, ks: tuple[int, ...] = (1, 3, 5)) -> dict[int, bool]:
    """Say whether any of the top-k scoring patches intersects the mask."""

    overlap = np.asarray(overlap).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    if overlap.size == 0 or overlap.shape != scores.shape:
        raise ValueError("overlap and scores must be non-empty arrays with the same shape")
    order = np.argsort(scores, kind="stable")[::-1]
    return {k: bool(np.any(overlap[order[: min(k, order.size)]] > 0)) for k in ks}


def patch_center(slc: tuple[slice, slice, slice]) -> np.ndarray:
    """Return a patch center in z, y, x voxel coordinates."""

    return np.asarray([(axis.start + axis.stop - 1) / 2 for axis in slc], dtype=np.float32)


def patch_overlap_fractions(mask: np.ndarray, grid: PatchGrid) -> np.ndarray:
    """Compute every patch overlap using a summed-volume table."""

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != grid.volume_shape:
        raise ValueError("mask shape does not match grid.volume_shape")
    if not np.any(mask):
        return np.zeros(len(grid), dtype=np.float32)
    integral = np.pad(mask.astype(np.uint32), ((1, 0), (1, 0), (1, 0)))
    integral = integral.cumsum(0, dtype=np.uint32).cumsum(1, dtype=np.uint32).cumsum(2, dtype=np.uint32)
    counts = []
    for z, y, x in grid.iter_coords():
        z1, y1, x1 = z + grid.patch_shape[0], y + grid.patch_shape[1], x + grid.patch_shape[2]
        count = (
            int(integral[z1, y1, x1]) - int(integral[z, y1, x1])
            - int(integral[z1, y, x1]) - int(integral[z1, y1, x])
            + int(integral[z, y, x1]) + int(integral[z, y1, x])
            + int(integral[z1, y, x]) - int(integral[z, y, x])
        )
        counts.append(count)
    return np.asarray(counts, dtype=np.float32) / float(np.prod(grid.patch_shape))


def _unique_records(records, data_root: Path):
    from .dataset import scan_dir

    unique = {}
    for record in records:
        unique.setdefault(str(scan_dir(data_root, record).resolve()), record)
    return list(unique.values())


def _load_validation_records(config: dict, data_root: Path):
    from .dataset import cross_generator_records, read_records
    from .train import subset_records

    metadata_dir = Path(config.get("metadata_dir", "metadata/m3dsynth"))
    if not metadata_dir.is_absolute():
        metadata_dir = Path.cwd() / metadata_dir
    split = config.get("split", {})
    data = config.get("data", {}) or {}
    training = config.get("training", {})
    groups = cross_generator_records(
        read_records(metadata_dir),
        train_mods=split["train_mods"],
        valid_mods=split.get("valid_mods", split["train_mods"]),
        test_mods=split.get("test_mods", [split.get("heldout_mod")]),
        train_splits=(split.get("train_set", "train"),),
        valid_splits=(split.get("valid_set", "valid"),),
        test_splits=(split.get("test_set", "test"),),
        include_real=bool(split.get("include_real", True)),
    )
    seed = int(data.get("record_subset_seed", training.get("seed", 21)))
    records = subset_records(groups["valid"], data.get("max_valid_records"), seed)
    return _unique_records(records, data_root)


def _load_model(checkpoint: Path, device: str):
    from .model import Patch3DModelConfig, build_patch3d_classifier
    from .train import _load_torch_payload

    payload = _load_torch_payload(checkpoint)
    cfg = payload.get("config", {}).get("model", {})
    model = build_patch3d_classifier(Patch3DModelConfig(
        in_channels=int(cfg.get("in_channels", 1)),
        num_classes=int(cfg.get("num_classes", 1)),
        base_channels=int(cfg.get("base_channels", 16)),
        dropout=float(cfg.get("dropout", 0.2)),
    ))
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device)


def _draw_report(scan, mask, slices, scores, overlaps, destination: Path) -> None:
    from PIL import Image, ImageDraw

    ranked = np.argsort(scores, kind="stable")[::-1][:5]
    mask_z = int(np.mean(np.argwhere(mask)[:, 0]))
    panels = [("mask center", mask_z, None)] + [
        (f"top {rank + 1}", int(round(patch_center(slices[index])[0])), int(index))
        for rank, index in enumerate(ranked)
    ]
    rendered = []
    colors = [(255, 40, 40), (255, 150, 20), (255, 150, 20), (255, 220, 30), (255, 220, 30)]
    for title, z, patch_index in panels:
        base = np.uint8(np.clip(scan[z], 0, 1) * 255)
        rgb = np.repeat(base[:, :, None], 3, axis=2)
        rgb[mask[z]] = (40, 220, 40)
        image = Image.fromarray(rgb).resize((512, 512))
        draw = ImageDraw.Draw(image)
        subtitle = title
        if patch_index is not None:
            slc = slices[patch_index]
            intersection = mask[z].copy()
            intersection[: slc[1].start] = False
            intersection[slc[1].stop :] = False
            intersection[:, : slc[2].start] = False
            intersection[:, slc[2].stop :] = False
            if slc[0].start <= z < slc[0].stop:
                rgb[intersection] = (0, 240, 255)
                image = Image.fromarray(rgb).resize((512, 512))
                draw = ImageDraw.Draw(image)
            scale_y, scale_x = 512 / scan.shape[1], 512 / scan.shape[2]
            rank = list(ranked).index(patch_index)
            color = (0, 240, 255) if overlaps[patch_index] > 0 else colors[rank]
            draw.rectangle(
                (slc[2].start * scale_x, slc[1].start * scale_y,
                 (slc[2].stop - 1) * scale_x, (slc[1].stop - 1) * scale_y),
                outline=color, width=4,
            )
            subtitle += f" score={scores[patch_index]:.3f} hit={'yes' if overlaps[patch_index] > 0 else 'no'}"
        draw.rectangle((0, 0, 512, 24), fill=(0, 0, 0))
        draw.text((6, 5), subtitle, fill=(255, 255, 255))
        rendered.append(image)
    canvas = Image.new("RGB", (1536, 1024), "black")
    for position, image in enumerate(rendered):
        canvas.paste(image, ((position % 3) * 512, (position // 3) * 512))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def _volume_score_candidates(scores: np.ndarray) -> dict[str, float]:
    ordered = np.sort(np.asarray(scores, dtype=np.float32))[::-1]
    return {"max": float(ordered[0]), **{
        f"top{k}_mean": float(np.mean(ordered[: min(k, ordered.size)])) for k in (3, 5)
    }}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--out-dir")
    return parser.parse_args()


def main() -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from tqdm import tqdm

    from .dataset import label_dir
    from .inference import predict_patch_scores
    from .evaluation import best_threshold_by_balanced_accuracy
    from .train import load_yaml_config
    from .volume_io import align_mask_to_scan, load_label_mask, load_normalized_scan

    args = parse_args()
    config = load_yaml_config(args.config)
    checkpoint = Path(args.checkpoint)
    data_root = Path(args.data_root or config.get("data_root", "data/M3Dsynth"))
    output_dir = Path(args.out_dir) if args.out_dir else checkpoint.parent / "patch_level_report"
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_cfg = config.get("patches", {})
    patch_shape = tuple(patch_cfg.get("patch_shape", (32, 32, 32)))
    stride = tuple(patch_cfg.get("inference_stride", (16, 16, 16)))
    positive_fraction = float(patch_cfg.get("positive_overlap_fraction", 0.05))
    batch_size = int(args.batch_size or config.get("evaluation", {}).get("calibration_batch_size", 32))
    records = _load_validation_records(config, data_root)
    model = _load_model(checkpoint, args.device)

    patch_truth, patch_scores = [], []
    volume_rows, candidate_scores = [], {name: [] for name in ("max", "top3_mean", "top5_mean")}
    for record in tqdm(records, desc="patch-level validation", unit="vol"):
        scan = load_normalized_scan(data_root, record)
        mask = np.zeros(scan.shape, dtype=bool) if record.is_real else align_mask_to_scan(
            load_label_mask(label_dir(data_root, record)), scan.shape, record.img_id
        )
        grid = PatchGrid(tuple(scan.shape), patch_shape, stride)
        slices = list(grid.iter_slices())
        scores = predict_patch_scores(model, scan, grid, batch_size=batch_size, device=args.device)
        overlaps = patch_overlap_fractions(mask, grid)
        supervised = np.logical_or(overlaps == 0, overlaps >= positive_fraction)
        patch_truth.append(overlaps[supervised] >= positive_fraction)
        patch_scores.append(scores[supervised])
        candidates = _volume_score_candidates(scores)
        for name, value in candidates.items():
            candidate_scores[name].append(value)

        if record.is_manipulated:
            hits = topk_hits(overlaps, scores)
            top_index = int(np.argmax(scores))
            mask_center = np.mean(np.argwhere(mask), axis=0)
            distance = float(np.linalg.norm(patch_center(slices[top_index]) - mask_center))
            image_name = f"{record.img_id}.png"
            _draw_report(scan, mask, slices, scores, overlaps, output_dir / "images" / image_name)
            volume_rows.append({
                "img_id": record.img_id,
                "top1_hit": int(hits[1]), "top3_hit": int(hits[3]), "top5_hit": int(hits[5]),
                "top1_center_distance_voxels": distance,
                "top1_score": float(scores[top_index]), "report_image": f"images/{image_name}",
            })

    truth = np.concatenate(patch_truth)
    scores = np.concatenate(patch_scores)
    patch_threshold, patch_binary = best_f1_threshold(truth, scores)
    patch_counts = binary_counts(truth, scores, patch_threshold)
    volume_truth = np.asarray([record.is_manipulated for record in records], dtype=bool)
    volume_results = {}
    for name, values in candidate_scores.items():
        values_array = np.asarray(values, dtype=np.float32)
        threshold, _ = best_threshold_by_balanced_accuracy(volume_truth, values_array)
        counts = binary_counts(volume_truth, values_array, threshold)
        volume_results[name] = {
            "auc": float(roc_auc_score(volume_truth, values_array)), "threshold": threshold,
            **metrics_from_counts(counts), "confusion": counts,
        }
    selected_volume_score = max(volume_results, key=lambda name: (volume_results[name]["auc"], volume_results[name]["balanced_accuracy"]))
    distances = np.asarray([row["top1_center_distance_voxels"] for row in volume_rows])
    report = {
        "checkpoint": str(checkpoint), "n_validation_volumes": len(records),
        "patch_definition": {"positive_overlap_fraction": positive_fraction, "ambiguous_patches_excluded": True},
        "patch_classification": {
            "n_patches": int(truth.size), "n_positive": int(truth.sum()),
            "positive_fraction": float(np.mean(truth)), "auc": float(roc_auc_score(truth, scores)),
            "average_precision": float(average_precision_score(truth, scores)),
            "threshold": patch_threshold, **patch_binary, "confusion": patch_counts,
        },
        "localization": {
            "definition": "A top-k hit means at least one selected patch has any non-zero mask intersection.",
            **{f"top{k}_hit_rate": float(np.mean([row[f'top{k}_hit'] for row in volume_rows])) for k in (1, 3, 5)},
            "positive_patch_recall": patch_binary["recall"],
            "top1_center_distance_voxels": {
                "mean": float(np.mean(distances)), "median": float(np.median(distances)),
                "p90": float(np.percentile(distances, 90)),
            },
        },
        "volume_classification": {"selected_score": selected_volume_score, "candidates": volume_results},
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "volumes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=volume_rows[0].keys())
        writer.writeheader()
        writer.writerows(volume_rows)
    print(json.dumps(report, indent=2))
    print(f"Patch-level report written to {output_dir}")


if __name__ == "__main__":
    main()
