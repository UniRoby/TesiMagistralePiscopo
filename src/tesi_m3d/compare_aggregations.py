"""Compare average and Gaussian heatmap aggregation without retraining."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .train import build_loaders, calibrate_best_checkpoint, load_yaml_config, resolve_cache_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", help="Output JSON; defaults beside the checkpoint.")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    config = load_yaml_config(args.config)
    output_dir = checkpoint.parent
    cache_dir = resolve_cache_dir(config, None, output_dir)
    objects = build_loaders(
        config, data_root_override=args.data_root, cache_dir=cache_dir, device=args.device
    )
    if objects.valid_dataset is None:
        raise SystemExit("The configuration produced no validation dataset.")

    results = {}
    for aggregation in ("average", "gaussian"):
        candidate = copy.deepcopy(config)
        candidate.setdefault("evaluation", {})["heatmap_aggregation"] = aggregation
        print(f"Evaluating aggregation={aggregation}...")
        calibration = calibrate_best_checkpoint(
            checkpoint, objects.valid_dataset, candidate, device=args.device,
            batch_size=int(candidate["evaluation"].get("calibration_batch_size", 32)),
        )
        results[aggregation] = calibration
        (output_dir / f"calibration_{aggregation}.json").write_text(
            json.dumps(calibration, indent=2) + "\n"
        )

    selected = max(
        results,
        key=lambda name: (
            results[name]["classification"]["auc"],
            results[name]["localization"]["f1"],
        ),
    )
    comparison = {"selected_aggregation": selected, "results": results}
    out = Path(args.out) if args.out else output_dir / "aggregation_comparison.json"
    out.write_text(json.dumps(comparison, indent=2) + "\n")
    print(f"Selected aggregation={selected}; comparison written to {out}")


if __name__ == "__main__":
    main()
