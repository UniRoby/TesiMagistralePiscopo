"""Mine high-scoring clean patches for a hard-negative fine-tuning run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate_patch_level import patch_overlap_fractions
from .patch_index import PatchIndex
from .patches import PatchGrid


def select_hard_negative_indices(scores: np.ndarray, overlaps: np.ndarray, count: int) -> np.ndarray:
    """Return up to ``count`` highest-scoring patches with zero mask overlap."""

    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    overlaps = np.asarray(overlaps, dtype=np.float32).reshape(-1)
    if scores.shape != overlaps.shape or count < 1:
        raise ValueError("scores and overlaps must match and count must be positive")
    clean = np.flatnonzero(overlaps == 0.0)
    ordered = clean[np.argsort(scores[clean], kind="stable")[::-1]]
    return ordered[:count]


def _scores_for_coords(model, scan: np.ndarray, coords: np.ndarray, patch_shape, batch_size: int, device: str) -> np.ndarray:
    import torch

    result = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(coords), batch_size):
            batch_coords = coords[start : start + batch_size]
            batch = np.stack([
                scan[z : z + patch_shape[0], y : y + patch_shape[1], x : x + patch_shape[2]]
                for z, y, x in batch_coords
            ])
            logits = model(torch.from_numpy(batch[:, None]).to(device))
            result.append(torch.sigmoid(logits).reshape(-1).cpu().numpy())
    return np.concatenate(result).astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--out", required=True, help="Destination .npz PatchIndex.")
    parser.add_argument("--negatives-per-volume", type=int, default=64)
    parser.add_argument("--random-negatives-per-volume", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    from tqdm import tqdm

    from .dataset import label_dir
    from .train import build_loaders, load_yaml_config
    from .volume_io import align_mask_to_scan, load_label_mask, load_normalized_scan
    from .evaluate_patch_level import _load_model

    args = parse_args()
    config = load_yaml_config(args.config)
    data_root = Path(args.data_root or config.get("data_root", "data/M3Dsynth"))
    objects = build_loaders(config, data_root_override=str(data_root), device=args.device)
    source = objects.train_dataset.examples
    records = objects.train_dataset.records
    patch_shape = tuple(config.get("patches", {}).get("patch_shape", (32, 32, 32)))
    stride = tuple(config.get("hard_negative_mining", {}).get("mining_stride", (16, 16, 16)))
    model = _load_model(Path(args.checkpoint), args.device)

    record_parts, coord_parts, label_parts, soft_parts, flag_parts, score_parts = [], [], [], [], [], []
    rng = np.random.default_rng(int(config.get("training", {}).get("seed", 21)))
    for record_index, record in tqdm(list(enumerate(records)), desc="mining hard negatives", unit="vol"):
        positives = np.flatnonzero((source.record_index == record_index) & (source.label == 1))
        if positives.size:
            record_parts.append(source.record_index[positives])
            coord_parts.append(source.coord[positives])
            label_parts.append(source.label[positives])
            soft_parts.append(source.soft_score[positives])
            flag_parts.append(np.zeros(positives.size, dtype=bool))

        scan = load_normalized_scan(data_root, record)
        mask = np.zeros(scan.shape, dtype=bool) if record.is_real else align_mask_to_scan(
            load_label_mask(label_dir(data_root, record)), scan.shape, record.img_id
        )
        grid = PatchGrid(tuple(scan.shape), patch_shape, stride)
        coords = np.asarray(list(grid.iter_coords()), dtype=np.int32)
        scores = _scores_for_coords(model, scan, coords, patch_shape, args.batch_size, args.device)
        overlaps = patch_overlap_fractions(mask, grid)
        selected = select_hard_negative_indices(scores, overlaps, args.negatives_per_volume)
        clean = np.flatnonzero(overlaps == 0.0)
        remaining = np.setdiff1d(clean, selected, assume_unique=True)
        random_selected = rng.choice(
            remaining,
            size=min(args.random_negatives_per_volume, remaining.size),
            replace=False,
        )
        negatives = np.concatenate([selected, random_selected])
        record_parts.append(np.full(negatives.size, record_index, dtype=np.int32))
        coord_parts.append(coords[negatives])
        label_parts.append(np.zeros(negatives.size, dtype=np.uint8))
        soft_parts.append(np.zeros(negatives.size, dtype=np.float32))
        flag_parts.append(np.r_[np.ones(selected.size, dtype=bool), np.zeros(random_selected.size, dtype=bool)])
        score_parts.append(scores[selected])

    mined = PatchIndex(
        record_index=np.concatenate(record_parts), coord=np.concatenate(coord_parts),
        label=np.concatenate(label_parts), soft_score=np.concatenate(soft_parts),
    )
    out = Path(args.out)
    mined.save(out)
    flags_path = out.with_name(f"{out.stem}_flags.npy")
    np.save(flags_path, np.concatenate(flag_parts))
    report = {
        "checkpoint": str(args.checkpoint), "mining_stride": list(stride),
        "hard_negatives_per_volume": args.negatives_per_volume,
        "random_negatives_per_volume": args.random_negatives_per_volume,
        "n_patches": len(mined), "n_positive": mined.n_positive,
        "n_hard_negative": int(np.count_nonzero(np.concatenate(flag_parts))),
        "n_random_negative": int(len(mined) - mined.n_positive - np.count_nonzero(np.concatenate(flag_parts))),
        "flags_path": str(flags_path),
        "hard_negative_score": {
            "min": float(np.min(np.concatenate(score_parts))),
            "median": float(np.median(np.concatenate(score_parts))),
            "max": float(np.max(np.concatenate(score_parts))),
        },
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Hard-negative index written to {out}")


if __name__ == "__main__":
    main()
