"""Evaluate 3D heatmaps with per-volume AUC and maximum balanced accuracy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def summarize_rows(rows: list[dict]) -> dict:
    """Return macro mean and sample standard deviation overall and by generator."""

    def metrics(items: list[dict]) -> dict:
        result = {"n_volumes": len(items)}
        for name in ("voxel_auc", "max_balanced_accuracy"):
            values = np.asarray([item[name] for item in items], dtype=np.float64)
            result[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        return result

    if not rows:
        raise ValueError("cannot summarize an empty evaluation")
    generators = sorted({str(row["generator"]) for row in rows})
    return {
        "overall": metrics(rows),
        "by_generator": {
            generator: metrics([row for row in rows if row["generator"] == generator])
            for generator in generators
        },
    }


def _records_for_evaluation(config: dict, split_name: str, mods: list[str] | None, limit: int | None):
    from .dataset import read_records
    from .train import subset_records

    metadata_dir = Path(config.get("metadata_dir", "metadata/m3dsynth"))
    if not metadata_dir.is_absolute():
        metadata_dir = Path.cwd() / metadata_dir
    split_cfg = config.get("split", {})
    split_value = split_cfg.get(f"{split_name}_set", split_name)
    if mods is None:
        mods = list(split_cfg.get(f"{split_name}_mods", split_cfg.get("valid_mods", [])))
    records = [
        record for record in read_records(metadata_dir)
        if record.is_manipulated and record.split == split_value and record.mod in set(mods)
    ]
    if not records:
        raise ValueError(f"no manipulated records found for split={split_value!r}, mods={mods}")
    seed = int(config.get("training", {}).get("seed", 21))
    return subset_records(records, limit, seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--mods", nargs="+", help="Generator names; defaults to the selected config split.")
    parser.add_argument("--max-records", type=int, help="Optional deterministic debug limit.")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir")
    return parser.parse_args()


def main() -> None:
    from tqdm import tqdm

    from .dataset import label_dir
    from .evaluate_patch_level import _load_model
    from .evaluation import voxel_auc_max_balanced_accuracy
    from .inference import infer_heatmap
    from .train import load_yaml_config
    from .volume_io import align_mask_to_scan, load_label_mask, load_normalized_scan

    args = parse_args()
    config = load_yaml_config(args.config)
    checkpoint = Path(args.checkpoint)
    data_root = Path(args.data_root or config.get("data_root", "data/M3Dsynth"))
    records = _records_for_evaluation(config, args.split, args.mods, args.max_records)
    patch_cfg = config.get("patches", {})
    evaluation_cfg = config.get("evaluation", {})
    patch_shape = tuple(patch_cfg.get("patch_shape", (32, 32, 32)))
    stride = tuple(patch_cfg.get("inference_stride", (16, 16, 16)))
    aggregation = str(evaluation_cfg.get("heatmap_aggregation", "average"))
    batch_size = int(args.batch_size or evaluation_cfg.get("calibration_batch_size", 32))
    output_dir = Path(args.out_dir) if args.out_dir else checkpoint.parent / f"paper_protocol_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _load_model(checkpoint, args.device)

    rows = []
    for record in tqdm(records, desc="paper-protocol evaluation", unit="vol"):
        scan = load_normalized_scan(data_root, record)
        mask = align_mask_to_scan(
            load_label_mask(label_dir(data_root, record)), scan.shape, record.img_id
        )
        heatmap = infer_heatmap(
            model, scan, patch_shape=patch_shape, stride=stride, batch_size=batch_size,
            aggregation=aggregation, device=args.device,
        )
        auc, max_ba, threshold = voxel_auc_max_balanced_accuracy(mask, heatmap)
        rows.append({
            "img_id": record.img_id, "orig_id": record.orig_id,
            "generator": record.mod, "manipulation": record.ty,
            "voxel_auc": auc, "max_balanced_accuracy": max_ba,
            "max_ba_threshold": threshold,
        })

    summary = {
        "protocol": {
            "unit": "3D manipulated volume",
            "aggregation": "macro mean of per-volume metrics",
            "std": "sample standard deviation (ddof=1)",
            "split": args.split, "patch_shape": list(patch_shape),
            "stride": list(stride), "heatmap_aggregation": aggregation,
        },
        **summarize_rows(rows),
    }
    with (output_dir / "per_volume.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Paper-compatible report written to {output_dir}")


if __name__ == "__main__":
    main()
