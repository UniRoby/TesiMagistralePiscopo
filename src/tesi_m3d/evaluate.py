"""CLI wrapper for heatmap localization metrics."""

from __future__ import annotations

import argparse

import numpy as np

from .evaluation import binary_localization_metrics, threshold_heatmap, voxel_auc_ap


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for heatmap evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate a heatmap against a binary 3D mask.")
    parser.add_argument("--heatmap", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    """Load arrays, compute metrics, and print one compact report line."""

    args = parse_args()
    heatmap = np.load(args.heatmap)
    mask = np.load(args.mask).astype(bool)
    pred = threshold_heatmap(heatmap, args.threshold)
    metrics = binary_localization_metrics(mask, pred)
    auc, ap = voxel_auc_ap(mask, heatmap)
    print(
        f"precision={metrics.precision:.6f}, recall={metrics.recall:.6f}, "
        f"f1={metrics.f1:.6f}, dice={metrics.dice:.6f}, iou={metrics.iou:.6f}, "
        f"voxel_auc={auc:.6f}, ap={ap:.6f}"
    )


if __name__ == "__main__":
    main()
