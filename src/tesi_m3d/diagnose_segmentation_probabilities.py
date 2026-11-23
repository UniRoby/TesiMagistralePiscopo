"""Read-only full-crop probability diagnosis of the eight hard-crop checkpoints."""
from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from .compare_segmentation_batches import tensor_digest
from .debug_segmentation import crop_regions, spatial_errors
from .debug_segmentation_hard_crops import HARD_CROPS, prepare_cases
from .diagnose_localization import write_csv, write_json
from .segmentation import build_segmenter, file_digest, segmentation_loss_components
from .segmentation_metrics import preview_planes, save_preview
from .train import _load_torch_payload, resolve_cache_dir
from .train_segmentation import check_prerequisites, read_json, validate_config


def probability_statistics(probability, target):
    """Exact quantiles and float64 sums on all voxels, without histogram binning."""
    probability, target = np.asarray(probability), np.asarray(target)
    if (probability.ndim != 3 or not probability.size or probability.shape != target.shape or not np.isfinite(probability).all()
            or (probability < 0).any() or (probability > 1).any() or not np.isin(target, [0, 1]).all()):
        raise ValueError("expected matching 3D probability in [0,1] and binary mask")
    target = target.astype(bool)
    distributions = []
    for name, region in crop_regions(probability.shape):
        region = region.numpy()
        for compartment, selected in (("all", region), ("foreground", region & target), ("background", region & ~target)):
            values = probability[selected].astype(np.float64)
            quantiles = np.quantile(values, [0, .01, .05, .5, .95, .99, 1]) if values.size else [None] * 7
            distributions.append({"region": name, "compartment": compartment, "n_voxels": int(values.size),
                                  "probability_mass": float(values.sum()),
                                  "mean": float(values.mean()) if values.size else None,
                                  **dict(zip(("min", "p01", "p05", "p50", "p95", "p99", "max"),
                                             [float(v) if v is not None else None for v in quantiles])),
                                  **{f"fraction_ge_{level}": float((values >= level).mean()) if values.size else None
                                     for level in (.1, .5, .9)}})
    n_truth = int(target.sum())
    detected = probability >= .5
    tp = int((target & detected).sum())
    n_pred = int(detected.sum())
    mass_fg = float(probability[target].sum(dtype=np.float64))
    mass_bg = float(probability[~target].sum(dtype=np.float64))
    mass = mass_fg + mass_bg
    summary = {"target_voxels": n_truth, "predicted_voxels": n_pred, "true_positive_voxels": tp,
               "false_positive_voxels": n_pred - tp, "false_negative_voxels": n_truth - tp,
               "hard_dice": 2 * tp / (n_truth + n_pred) if n_truth else None,
               "soft_dice_float64": (2 * mass_fg + 1e-5) / (mass + n_truth + 1e-5) if n_truth else None,
               "probability_mass_inside": mass_fg, "probability_mass_outside": mass_bg,
               "mean_probability_inside": mass_fg / n_truth if n_truth else None,
               "mean_probability_outside": mass_bg / (target.size - n_truth) if target.size > n_truth else None,
               "background_mass_fraction": mass_bg / mass if mass else None,
               "precision": tp / n_pred if n_pred else None, "recall": tp / n_truth if n_truth else None,
               "negative_crop_fp_fraction": n_pred / target.size if not n_truth else None}
    return summary, distributions


def evaluate_checkpoint(path, batch, config, manifest, expected, initial_hash, output, device):
    payload = _load_torch_payload(path)
    fingerprint = {key + "_sha256": tensor_digest(batch[key]) for key in ("image", "mask")}
    keys = ("positive_crop_dice", "negative_crop_fp_fraction", "bce", "soft_dice_loss", "passed")
    if (payload.get("task") != "segmentation_debug" or payload.get("config") != config
            or payload.get("manifest") != manifest or payload.get("fixed_batch") != fingerprint
            or payload.get("initial_state_sha256") != initial_hash or payload.get("updates") != expected["updates"]
            or any(payload.get("metrics", {}).get(k) != expected[k] for k in keys)):
        raise ValueError(f"checkpoint identity/config/crop hashes/metrics mismatch: {path}")
    model = build_segmenter(config["model"], state_dict=payload["model_state_dict"]).to(device).eval()
    microbatch = min(config["training"]["batch_size"], len(batch["image"]))
    with torch.inference_mode():
        logits = torch.cat([model(batch["image"][start:start + microbatch].to(device)).float().cpu()
                            for start in range(0, len(batch["image"]), microbatch)])
    if logits.shape != batch["mask"].shape or not torch.isfinite(logits).all():
        raise ValueError(f"non-finite or incorrectly shaped checkpoint logits: {path}")
    probability = logits.sigmoid()
    bce, dice_loss = segmentation_loss_components(logits, batch["mask"])
    del model, payload
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "probabilities.npz", probability=probability.numpy(), mask=batch["mask"].numpy().astype(np.uint8))
    write_csv(output / "spatial_errors.csv", spatial_errors(batch["mask"], probability))
    summaries, distributions = [], []
    for i in range(len(probability)):
        kind = "positive" if i == 0 else "negative"
        p, target = probability[i, 0].numpy(), batch["mask"][i, 0].numpy().astype(bool)
        summary, rows = probability_statistics(p, target)
        summaries.append({"crop": kind, **summary})
        distributions.extend({"crop": kind, **r} for r in rows)
        save_preview(output / f"{kind}.png", preview_planes(batch["image"][i, 0].numpy(), target, p), .5)
    metrics = {"positive_crop_dice": summaries[0]["hard_dice"],
               "negative_crop_fp_fraction": summaries[1]["negative_crop_fp_fraction"] if len(summaries) == 2 else None,
               "bce": float(bce), "soft_dice_loss": float(dice_loss)}
    delta = {k: metrics[k] - expected[k] if metrics[k] is not None else None for k in metrics}
    result = {"checkpoint_sha256": file_digest(path), "updates": expected["updates"], "evaluation": "float32_no_autocast",
              "threshold": .5, "metrics": metrics, "metric_delta_from_saved": delta,
              "reproduced_within_1e_4": all(v is None or abs(v) <= 1e-4 for v in delta.values()), "crops": summaries}
    write_csv(output / "distributions.csv", distributions)
    write_json(output / "report.json", result)
    return result


def run_diagnosis(data_root, *, run_dir="outputs/segmentation_hard_crops", device="cpu",
                  reference_dir="outputs/segmentation_batch_comparison", output_dir="outputs/segmentation_probability_diagnostics",
                  cache_dir=None, real_scan_root=None):
    source, reference, output = Path(run_dir), Path(reference_dir), Path(output_dir)
    if any(output.resolve() == p.resolve() or p.resolve() in output.resolve().parents for p in (source, reference)):
        raise ValueError("diagnosis output must be outside the source run and reference directories")
    if output.exists() and any(output.iterdir()):
        raise ValueError("diagnosis output already contains files; choose a new --output-dir")
    config, manifest, saved = (read_json(source / name) for name in ("effective_config.json", "manifest.json", "report.json"))
    validate_config(config)
    stages = [f"{case}/{stage}" for case, _ in HARD_CROPS for stage in ("positive", "positive_negative")]
    if (config["stage"] != "overfit" or saved.get("task") != "segmentation_hard_crops" or saved.get("error")
            or saved.get("active_stage") is not None or set(saved.get("stages", {})) != set(stages)
            or saved.get("manifest_id") != manifest["manifest_id"]):
        raise ValueError("diagnosis requires a completed hard-crop run, including scientifically failed stages")
    paths = [source / stage / f"{name}.pt" for stage in stages for name in ("best", "last")]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing trained checkpoints; run this command on the laboratory PC: " + ", ".join(missing))
    check_prerequisites(config, manifest, data_root, real_scan_root)
    cases, comparison = prepare_cases(config, manifest, reference, data_root,
                                      cache_dir=resolve_cache_dir(config, cache_dir, output), real_scan_root=real_scan_root)
    selection = read_json(source / "selection.json")
    if (selection.get("crops") != [row for _, _, selected in cases for row in selected]
            or selection.get("threshold") != .5 or saved.get("initial_state_sha256") != comparison["arms"]["mixed"]["initial_state_sha256"]
            or saved.get("reference_crop_pool_sha256") != comparison["crop_pool_sha256"]
            or selection.get("reference_crop_pool_sha256") != comparison["crop_pool_sha256"]):
        raise ValueError("hard-crop run selection or reference identity mismatch")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "selection.json", selection)
    write_json(output / "effective_config.json", config)
    write_json(output / "manifest.json", manifest)
    report = {"task": "segmentation_probability_diagnostics", "complete": False,
              "manifest_id": manifest["manifest_id"], "source_run": str(source),
              "runtime": {"torch": torch.__version__, "monai": version("monai"), "device": str(device)},
              "quantiles": "exact numpy linear quantiles on all float32 probability voxels; masses summed in float64",
              "active_checkpoint": None, "checkpoints": {}}
    try:
        for case, pair, _ in cases:
            for stage, count in (("positive", 1), ("positive_negative", 2)):
                key = f"{case}/{stage}"
                batch = {k: t[:count] for k, t in pair.items()}
                for name in ("best", "last"):
                    checkpoint = source / key / f"{name}.pt"
                    label = f"{key}/{name}"
                    report["active_checkpoint"] = label
                    print(f"measuring full-crop probabilities: {label}", flush=True)
                    report["checkpoints"][label] = evaluate_checkpoint(checkpoint, batch, config, manifest,
                        saved["stages"][key][name], saved["initial_state_sha256"], output / label, device)
                    write_json(output / "report.json", report)
                    if str(device).startswith("cuda"):
                        torch.cuda.empty_cache()
        report.update(complete=True, active_checkpoint=None,
                      all_reproduced_within_1e_4=all(r["reproduced_within_1e_4"] for r in report["checkpoints"].values()))
        return report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--real-scan-root")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-dir", default="outputs/segmentation_hard_crops")
    parser.add_argument("--reference-dir", default="outputs/segmentation_batch_comparison")
    parser.add_argument("--output-dir", default="outputs/segmentation_probability_diagnostics")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    result = run_diagnosis(args.data_root, run_dir=args.run_dir, device=args.device, reference_dir=args.reference_dir,
                           output_dir=args.output_dir, cache_dir=args.cache_dir, real_scan_root=args.real_scan_root)
    print(f"Probability diagnosis complete: {len(result['checkpoints'])} checkpoints, {args.output_dir}")
    print(f"Saved metrics reproduced within 1e-4: {result['all_reproduced_within_1e_4']}")


if __name__ == "__main__":
    main()
