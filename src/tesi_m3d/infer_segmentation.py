"""Predict from a CT only, using checkpoint preprocessing and frozen calibration."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .dataset import load_tiff_stack
from .diagnose_localization import write_json
from .segmentation import SEGMENTER_NAME, load_predictor, predict_checkpoint, read_calibration
from .segmentation_data import validate_stack
from .train import resolve_cache_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--volume-dir", required=True)
    parser.add_argument("--calibration")
    parser.add_argument("--out-dir", "--output-dir", dest="out_dir", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    model, payload = load_predictor(Path(args.checkpoint), args.device)
    if payload["config"]["model"]["name"] != SEGMENTER_NAME:
        parser.error("use tesi_m3d.inference for patch-classifier checkpoints")
    calibration_path = args.calibration or str(Path(args.checkpoint).with_name("calibration.json"))
    calibration = read_calibration(calibration_path, args.checkpoint)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    validate_stack(args.volume_dir)
    heatmap, _ = predict_checkpoint(model, payload, load_tiff_stack(args.volume_dir), args.volume_dir,
                                     resolve_cache_dir({}, args.cache_dir, output),
                                     device=args.device, batch_size=args.batch_size)
    np.save(output / "probability.npy", heatmap)
    np.save(output / "mask.npy", heatmap >= calibration["localization_threshold"])
    write_json(output / "prediction.json", {"shape": list(heatmap.shape), "probability_dtype": "float32",
               "detection_score": float(heatmap.max()),
               "is_manipulated": bool(heatmap.max() >= calibration["detection_threshold"]),
               "calibration": calibration, "checkpoint": args.checkpoint})


if __name__ == "__main__":
    main()
