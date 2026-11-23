"""Paired linear/squared Dice training on the four fixed hard-crop probes."""
from __future__ import annotations

import argparse
import copy
import json
from importlib.metadata import version
from pathlib import Path

import torch

from .debug_segmentation import run_fixed_stage
from .debug_segmentation_hard_crops import prepare_cases
from .diagnose_localization import write_json
from .diagnose_segmentation_probabilities import evaluate_checkpoint
from .train import load_yaml_config, resolve_cache_dir
from .train_segmentation import check_prerequisites, read_json, validate_config


def run_comparison(config, data_root, *, device="cpu", reference_dir="outputs/segmentation_batch_comparison",
                   output_dir="outputs/segmentation_loss_comparison", cache_dir=None, real_scan_root=None):
    config = copy.deepcopy(config)
    validate_config(config)
    if config["stage"] != "overfit":
        raise ValueError("loss comparison requires the original overfit config")
    manifest = read_json(config["manifest"])
    check_prerequisites(config, manifest, data_root, real_scan_root)
    output, reference = Path(output_dir), Path(reference_dir)
    if output.resolve() == reference.resolve() or reference.resolve() in output.resolve().parents:
        raise ValueError("comparison output must be outside the reference directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError("loss comparison output already contains files; choose a new --output-dir")
    cases, previous = prepare_cases(config, manifest, reference, data_root,
                                    cache_dir=resolve_cache_dir(config, cache_dir, output), real_scan_root=real_scan_root)
    initial_hash = previous["arms"]["mixed"]["initial_state_sha256"]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "effective_config.json", config)
    write_json(output / "manifest.json", manifest)
    write_json(output / "selection.json", {"reference_dir": str(reference_dir),
               "reference_crop_pool_sha256": previous["crop_pool_sha256"], "threshold": 0.5,
               "crops": [r for _, _, selected in cases for r in selected]})
    report = {"experiment": "dice_denominator_comparison", "complete": False, "passed": False,
              "manifest_id": manifest["manifest_id"], "initial_state_sha256": initial_hash,
              "reference_crop_pool_sha256": previous["crop_pool_sha256"],
              "max_updates_per_stage": config["training"].get("max_updates", 300),
              "runtime": {"torch": torch.__version__, "monai": version("monai"), "device": str(device)},
              "monitoring": "common linear Soft Dice, hard Dice and negative FP at fixed threshold 0.5",
              "active_stage": None, "active_checkpoint": None, "arms": {}, "deltas_squared_minus_linear": {}}
    try:
        for arm, squared in (("linear", False), ("squared", True)):
            arm_report = report["arms"][arm] = {"squared_pred": squared, "stages": {}}
            for case, pair, _ in cases:
                for name, count in (("positive", 1), ("positive_negative", 2)):
                    key = f"{case}/{name}"
                    label = f"{arm}/{key}"
                    folder = output / label
                    report["active_stage"] = label
                    print(f"starting paired loss comparison: {label}", flush=True)
                    batch = {k: t[:count] for k, t in pair.items()}
                    result, checked = run_fixed_stage(config, manifest, batch, folder, device=device,
                        complete_budget=True, expected_initial_state_sha256=initial_hash, squared_pred=squared)
                    stage = arm_report["stages"][key] = {"last": result, "best": read_json(folder / "best_metrics.json"),
                                                       "update_check": checked, "probability_diagnostics": {}}
                    write_json(output / "comparison.json", report)
                    if not checked["passed"]:
                        raise RuntimeError(f"parameter/BatchNorm preflight failed at {label}")
                    if (result["updates"] != report["max_updates_per_stage"]
                            or result["exposures_per_crop"] != report["max_updates_per_stage"]
                            or result["initial_state_sha256"] != initial_hash):
                        raise ValueError(f"loss comparison budget/exposure/initialization mismatch at {label}")
                    for checkpoint in ("best", "last"):
                        report["active_checkpoint"] = f"{label}/{checkpoint}"
                        stage["probability_diagnostics"][checkpoint] = evaluate_checkpoint(
                            folder / f"{checkpoint}.pt", batch, config, manifest, stage[checkpoint],
                            initial_hash, folder / checkpoint, device)
                    report["active_checkpoint"] = None
                    write_json(output / "comparison.json", report)
                    if str(device).startswith("cuda"):
                        torch.cuda.empty_cache()
        linear_stages, squared_stages = (report["arms"][name]["stages"] for name in ("linear", "squared"))
        for key in linear_stages:
            delta = report["deltas_squared_minus_linear"][key] = {}
            for checkpoint in ("best", "last"):
                first = linear_stages[key]["probability_diagnostics"][checkpoint]
                second = squared_stages[key]["probability_diagnostics"][checkpoint]
                a, b = first["crops"][0], second["crops"][0]
                delta[checkpoint] = {k: b[k] - a[k] if b[k] is not None and a[k] is not None else None
                                     for k in ("hard_dice", "precision", "recall", "false_positive_voxels",
                                               "false_negative_voxels", "mean_probability_outside")}
                delta[checkpoint]["negative_crop_fp_fraction"] = (second["metrics"]["negative_crop_fp_fraction"]
                    - first["metrics"]["negative_crop_fp_fraction"]) if len(first["crops"]) == 2 else None
        report.update(complete=True, active_stage=None, passed=all(s["last"]["passed"] for s in squared_stages.values()),
            all_reproduced_within_1e_4=all(d["reproduced_within_1e_4"] for arm in report["arms"].values()
                for stage in arm["stages"].values() for d in stage["probability_diagnostics"].values()))
        return report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(output / "comparison.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/segmentation_overfit.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--real-scan-root")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reference-dir", default="outputs/segmentation_batch_comparison")
    parser.add_argument("--output-dir", default="outputs/segmentation_loss_comparison")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    report = run_comparison(load_yaml_config(args.config), args.data_root, device=args.device,
                            reference_dir=args.reference_dir, output_dir=args.output_dir,
                            cache_dir=args.cache_dir, real_scan_root=args.real_scan_root)
    print(json.dumps({k: report[k] for k in ("complete", "passed", "all_reproduced_within_1e_4",
                                            "deltas_squared_minus_linear")}, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
