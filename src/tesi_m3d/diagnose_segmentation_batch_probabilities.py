"""Read-only probability/threshold audit of grouped/mixed best/last on 48 fixed crops."""
from __future__ import annotations

import argparse
import csv
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .compare_segmentation_batches import FixedCropPool
from .diagnose_localization import write_csv, write_json
from .diagnose_segmentation_probabilities import probability_statistics
from .segmentation import build_segmenter, file_digest
from .segmentation_data import SegmentationDataset, manifest_records
from .segmentation_metrics import BINS, ScoreHistogram, preview_planes, save_preview
from .train import _load_torch_payload, resolve_cache_dir, set_global_seed
from .train_segmentation import check_prerequisites, read_json, validate_config

THRESHOLDS = np.arange(1001, dtype=np.float64) / 1000
FIXED_INDEX = 500


def threshold_counts(probability, target):
    """Exact >= counts at common thresholds, on already validated crop arrays."""
    scores, truth = np.asarray(probability, dtype=np.float64), np.asarray(target, dtype=bool)
    foreground, background = np.sort(scores[truth]), np.sort(scores[~truth])
    tp = foreground.size - np.searchsorted(foreground, THRESHOLDS, side="left")
    fp = background.size - np.searchsorted(background, THRESHOLDS, side="left")
    return np.stack((tp, fp, foreground.size - tp, background.size - fp), axis=1)


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def profiles(counts, positive, real):
    """Keep the gate's macro positive Dice separate from voxel-weighted metrics."""
    pos, neg, reals = counts[positive].sum(0), counts[~positive].sum(0), counts[real].sum(0)
    crop_dice = 2 * counts[positive, :, 0] / np.maximum(1, 2 * counts[positive, :, 0]
                                                     + counts[positive, :, 1] + counts[positive, :, 2])
    rows = []
    for i, threshold in enumerate(THRESHOLDS):
        tp, fp, fn, tn = map(int, pos[i])
        nfp, ntn = int(neg[i, 1]), int(neg[i, 3])
        rows.append({"threshold": float(threshold), "mean_positive_crop_dice": float(crop_dice[:, i].mean()),
                     "positive_crop_micro_dice": ratio(2 * tp, 2 * tp + fp + fn),
                     "positive_crop_precision": ratio(tp, tp + fp), "positive_crop_recall": ratio(tp, tp + fn),
                     "positive_crop_tp": tp, "positive_crop_fp": fp, "positive_crop_fn": fn,
                     "negative_crop_fp": nfp, "negative_crop_fp_fraction": ratio(nfp, nfp + ntn),
                     "real_crop_fp_fraction": ratio(int(reals[i, 1]), int(reals[i, 1] + reals[i, 3])),
                     "all_crop_micro_dice": ratio(2 * tp, 2 * tp + fp + nfp + fn)})
    best = max(rows, key=lambda r: (r["mean_positive_crop_dice"], r["threshold"]))
    feasible = [r for r in rows if r["negative_crop_fp_fraction"] < .01]
    return rows, {"best_common_mean_dice": best,
                  "best_common_with_negative_fp_below_1pct": max(feasible,
                      key=lambda r: (r["mean_positive_crop_dice"], r["threshold"])) if feasible else None,
                  "mean_per_positive_crop_oracle_dice": float(crop_dice.max(1).mean()),
                  "grid_contains_numeric_gate_solution": any(r["mean_positive_crop_dice"] >= .9 for r in feasible)}


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evaluate_checkpoint(path, pool, config, manifest, arm, checkpoint, source_rows, output, device, *,
                        experiment="batch_composition_comparison"):
    started = perf_counter()
    expected = arm[checkpoint]
    payload = _load_torch_payload(path)
    sampling = {key: arm[key] for key in ("layout", "fixed_foreground_normalizer", "crop_pool_sha256")}
    if arm["layout"] == "full_pool":
        sampling.update({key: arm[key] for key in ("effective_batch_size", "accumulation_steps", "objective")})
    if (payload.get("task") != "segmentation" or payload.get("experiment") != experiment
            or payload.get("config") != config or payload.get("manifest") != manifest
            or payload.get("sampling") != sampling or payload.get("training_loss") != arm["training_loss"]
            or payload.get("lr_schedule") != arm.get("lr_schedule")
            or payload.get("updates") != expected["updates"] or payload.get("best_overfit_metrics") != arm["best"]):
        raise ValueError(f"checkpoint identity/config/sampling/loss/updates mismatch: {path}")
    if len(source_rows) != 48:
        raise ValueError("source metrics must identify all 48 crops")
    for row, identity in zip(source_rows, pool.rows):
        if any(str(row[key]) != str(identity[key]) for key in ("mod", "img_id", "crop_index", "target_voxels")):
            raise ValueError("source crop identity or target count changed")
    # Match run_arm's seed/cuDNN setup as well as overfit_metrics' inference path.
    set_global_seed(config["training"].get("seed", 21))
    model = build_segmenter(config["model"], state_dict=payload["model_state_dict"]).to(device).eval()
    del payload
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    output.mkdir(parents=True, exist_ok=True)
    positive_indices = np.flatnonzero(pool.positive)
    negative_indices = np.flatnonzero(~pool.positive)
    selected = {min(positive_indices, key=lambda i: float(source_rows[i]["dice"])),
                max(negative_indices, key=lambda i: float(source_rows[i]["negative_crop_fp_fraction"]))}
    rows, distributions, counts, histograms, previews = [], [], [], [], {}
    microbatch = config["training"]["batch_size"]
    with torch.no_grad():
        for start in range(0, 48, microbatch):
            logits = model(pool.image[start:start + microbatch].to(device)).float()
            if logits.shape != pool.mask[start:start + microbatch].shape or not torch.isfinite(logits).all():
                raise ValueError("non-finite or incorrectly shaped checkpoint logits")
            probabilities = logits.sigmoid().cpu().numpy()
            for offset, probability in enumerate(probabilities[:, 0]):
                i = start + offset
                target = pool.mask[i, 0].numpy().astype(bool)
                summary, distribution = probability_statistics(probability, target)
                current = threshold_counts(probability, target)
                hist = ScoreHistogram()
                hist.update(target, probability)
                histograms.append((hist.positive, hist.negative))
                counts.append(current)
                crop_dice = 2 * current[:, 0] / np.maximum(1, 2 * current[:, 0] + current[:, 1] + current[:, 2])
                oracle = int(np.flatnonzero(crop_dice == crop_dice.max())[-1]) if target.any() else None
                source = source_rows[i]
                rows.append({**pool.rows[i], **summary, "voxel_ap": hist.ap(), "peak_probability": float(probability.max()),
                    "delta_saved_peak_probability": float(probability.max()) - float(source["peak_probability"]),
                    "diagnostic_oracle_threshold": float(THRESHOLDS[oracle]) if oracle is not None else None,
                    "diagnostic_oracle_dice": float(crop_dice[oracle]) if oracle is not None else None,
                    "delta_saved_dice": summary["hard_dice"] - float(source["dice"]) if target.any() else None,
                    "delta_saved_negative_fp_fraction": summary["negative_crop_fp_fraction"]
                        - float(source["negative_crop_fp_fraction"]) if not target.any() else None,
                    **{f"delta_saved_{key}": summary[key] - int(source[key]) for key in
                       ("true_positive_voxels", "false_positive_voxels", "false_negative_voxels")}})
                distributions.extend({"pool_index": i, "img_id": pool.rows[i]["img_id"], **r} for r in distribution)
                if i in selected:
                    previews[i] = probability.copy()
            if (start + microbatch) % 8 == 0:
                print(f"probability audit: {output.parent.name}/{output.name} {start + microbatch}/48 crops", flush=True)
    del model
    counts = np.stack(counts)
    histograms = np.stack(histograms)
    curve, diagnostic = profiles(counts, pool.positive, pool.real)
    fixed = curve[FIXED_INDEX]
    delta = {"positive_crop_dice": fixed["mean_positive_crop_dice"] - expected["positive_crop_dice"],
             "negative_crop_fp_fraction": fixed["negative_crop_fp_fraction"] - expected["negative_crop_fp_fraction"]}
    failures = [f"aggregate {key}: delta={value:+.8g}" for key, value in delta.items() if not abs(value) <= 1e-4]
    if (fixed["mean_positive_crop_dice"] >= .9 and fixed["negative_crop_fp_fraction"] < .01) != expected["passed"]:
        failures.append("fixed .5 gate decision differs from saved decision")
    for row in rows:
        for key in ("delta_saved_dice", "delta_saved_negative_fp_fraction", "delta_saved_peak_probability"):
            if row[key] is not None and not abs(row[key]) <= 1e-4:
                failures.append(f"{key}={row[key]:+.8g} (pool_index={row['pool_index']}, "
                                f"scan={row['img_id']}, crop_index={row['crop_index']})")
    reproduced = not failures
    aps = {}
    for name, chosen in (("all_crops", np.ones(48, dtype=bool)), ("positive_crops", pool.positive), ("real_crops", pool.real)):
        hist = ScoreHistogram()
        hist.positive, hist.negative = histograms[chosen].sum(0)
        aps[name] = hist.ap()
    valid_aps = [r["voxel_ap"] for r in rows if r["voxel_ap"] is not None]
    aps["mean_positive_crop_ap"] = float(np.mean(valid_aps)) if valid_aps else None
    result = {"checkpoint_sha256": file_digest(path), "updates": expected["updates"],
              "evaluation": "float32_no_autocast", "training_loss": arm["training_loss"],
              "cudnn_benchmark": torch.backends.cudnn.benchmark,
              "sigmoid_device": str(device),
              "fixed_threshold": fixed, "voxel_ap": aps, "diagnostic_train_only": diagnostic,
              "metric_delta_from_saved": delta, "reproduced_within_1e_4": reproduced,
              "reproduction_failures": failures,
              "max_abs_saved_count_delta": max(abs(r[key]) for r in rows for key in
                  ("delta_saved_true_positive_voxels", "delta_saved_false_positive_voxels", "delta_saved_false_negative_voxels")),
              "preview_pool_indices": sorted(map(int, selected)), "seconds": perf_counter() - started,
              "max_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20 if str(device).startswith("cuda") else None}
    np.savez_compressed(output / "metrics.npz", thresholds=THRESHOLDS, counts=counts, histograms=histograms)
    write_csv(output / "crops.csv", rows)
    write_csv(output / "distributions.csv", distributions)
    write_csv(output / "thresholds.csv", curve)
    write_json(output / "report.json", result)
    if not reproduced:
        raise ValueError("checkpoint/crop metrics were not reproduced within 1e-4: "
                         + "; ".join(failures[:3]) + f". All {len(failures)} failures: {output / 'report.json'}; "
                         "do not interpret threshold diagnostics")
    diagnostic_rows = []
    for strategy in ("best_common_mean_dice", "best_common_with_negative_fp_below_1pct"):
        chosen = diagnostic[strategy]
        if chosen is None:
            continue
        j = int(round(chosen["threshold"] * 1000))
        for i, identity in enumerate(pool.rows):
            tp, fp, fn, tn = map(int, counts[i, j])
            diagnostic_rows.append({**identity, "strategy": strategy, "threshold": chosen["threshold"],
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "dice": ratio(2 * tp, 2 * tp + fp + fn) if pool.positive[i] else None,
                "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
                "negative_crop_fp_fraction": ratio(fp, fp + tn) if not pool.positive[i] else None})
    write_csv(output / "diagnostic_crops.csv", diagnostic_rows)
    for i, probability in previews.items():
        planes = preview_planes(pool.image[i, 0].numpy(), pool.mask[i, 0].numpy().astype(bool), probability.astype(np.float64))
        save_preview(output / f"pool{i:02d}_fixed.png", planes, .5)
        feasible = diagnostic["best_common_with_negative_fp_below_1pct"]
        if feasible is not None:
            save_preview(output / f"pool{i:02d}_diagnostic.png", planes, feasible["threshold"])
    result["seconds"] = perf_counter() - started
    write_json(output / "report.json", result)
    return result


def run_diagnosis(data_root, *, run_dir="outputs/segmentation_batch_comparison_squared", device="cpu",
                  output_dir="outputs/segmentation_batch_probability_diagnostics", cache_dir=None, real_scan_root=None):
    source, output = Path(run_dir), Path(output_dir)
    if output.resolve() == source.resolve() or source.resolve() in output.resolve().parents:
        raise ValueError("diagnosis output must be outside the source run")
    if output.exists() and any(output.iterdir()):
        raise ValueError("diagnosis output already contains files; choose a new --output-dir")
    config, manifest, saved = (read_json(source / name) for name in ("effective_config.json", "manifest.json", "comparison.json"))
    validate_config(config)
    if (config["stage"] != "overfit" or config["training"]["max_updates"] != 6 * config["training"]["epochs"]
            or saved.get("experiment") != "batch_composition_comparison"
            or not saved.get("complete") or saved.get("error") or saved.get("active_arm") is not None
            or set(saved.get("arms", {})) != {"grouped", "mixed"} or saved.get("manifest_id") != manifest["manifest_id"]):
        raise ValueError("diagnosis requires a complete grouped/mixed comparison, including scientifically failed gates")
    paths = [source / arm / f"{cp}.pt" for arm in ("grouped", "mixed") for cp in ("best", "last")]
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError("missing trained checkpoints; run this audit on the laboratory PC")
    check_prerequisites(config, manifest, data_root, real_scan_root)
    dataset = SegmentationDataset(manifest_records(manifest, "overfit"), data_root,
        patch_shape=config["patches"]["patch_shape"], preprocessing=config["preprocessing"], seed=manifest["seed"],
        augment=False, fixed=True, volume_cache_size=1, real_scan_root=real_scan_root,
        cache_dir=resolve_cache_dir(config, cache_dir, output))
    pool = FixedCropPool(dataset)
    del dataset
    ledger = read_rows(source / "crop_pool.csv")
    for row in ledger:
        for key in ("pool_index", "scan_index", "crop_index", "target_voxels"):
            row[key] = int(row[key])
    if ledger != pool.rows or pool.digest != saved["crop_pool_sha256"]:
        raise ValueError("regenerated crop identities/image/mask hashes differ from the source comparison")
    loss = {"bce_weight": .5, "dice_weight": .5, "squared_pred": config["training"].get("squared_pred", False), "empty_crops": "BCE_only"}
    for name, arm in saved["arms"].items():
        if (arm.get("layout") != name or arm.get("fixed_foreground_normalizer") != 4
                or arm.get("crop_pool_sha256") != pool.digest or arm.get("training_loss") != loss
                or arm.get("initial_state_sha256") != saved["arms"]["grouped"].get("initial_state_sha256")
                or not arm.get("initial_state_sha256") or arm.get("updates") != config["training"]["max_updates"]
                or arm.get("epochs") != config["training"]["epochs"]
                or arm.get("exposures_per_crop") != config["training"]["epochs"]
                or read_json(source / name / "effective_config.json") != config
                or read_json(source / name / "manifest.json") != manifest
                or read_json(source / name / "report.json") != arm):
            raise ValueError("comparison arm config/manifest/pool/initialization/loss/budget mismatch")
    if saved.get("training_loss") != loss:
        raise ValueError("comparison loss metadata mismatch")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "effective_config.json", config)
    write_json(output / "manifest.json", manifest)
    write_csv(output / "crop_pool.csv", pool.rows)
    report = {"task": "segmentation_batch_probability_diagnostics", "complete": False, "evaluation_only": True,
              "source_run": str(source), "source_comparison_sha256": file_digest(source / "comparison.json"),
              "manifest_id": manifest["manifest_id"], "crop_pool_sha256": pool.digest,
              "runtime": {"torch": torch.__version__, "monai": version("monai"), "device": str(device)},
              "ap_definition": f"floor-quantized probabilities, {BINS} bins; no voxel subsampling; single-class AP null",
              "threshold_definition": "1001 common thresholds 0..1 step .001; exact >= counts on float64-promoted float32 probabilities",
              "scope": "48 fixed training crops, including overlapping/repeated physical voxels; not native-volume validation",
              "diagnostic_only": "train/oracle thresholds cannot replace the fixed .5 gate or validation calibration",
              "active_checkpoint": None, "checkpoints": {}}
    try:
        for name, arm in saved["arms"].items():
            for cp in ("best", "last"):
                label = f"{name}/{cp}"
                report["active_checkpoint"] = label
                write_json(output / "report.json", report)
                print(f"starting read-only probability audit: {label}", flush=True)
                folder = source / name / "best" if cp == "best" else source / name
                report["checkpoints"][label] = evaluate_checkpoint(source / name / f"{cp}.pt", pool,
                    config, manifest, arm, cp, read_rows(folder / "crops.csv"), output / label, device)
                write_json(output / "report.json", report)
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
        report.update(complete=True, active_checkpoint=None, all_reproduced_within_1e_4=True)
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
    parser.add_argument("--run-dir", default="outputs/segmentation_batch_comparison_squared")
    parser.add_argument("--output-dir", default="outputs/segmentation_batch_probability_diagnostics")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    result = run_diagnosis(args.data_root, run_dir=args.run_dir, output_dir=args.output_dir, device=args.device,
                           cache_dir=args.cache_dir, real_scan_root=args.real_scan_root)
    print(f"Read-only diagnosis complete: {len(result['checkpoints'])} checkpoints; fixed .5 gate unchanged.")


if __name__ == "__main__":
    main()
