"""Paired mixed-batch overfit: zero versus replicate padding in the decoder only."""
from __future__ import annotations

import argparse
import copy
from importlib.metadata import version
from pathlib import Path

import torch

from .compare_segmentation_batches import FixedCropPool, run_arm
from .diagnose_localization import write_csv, write_json
from .diagnose_segmentation_batch_probabilities import evaluate_checkpoint, read_rows
from .segmentation import file_digest
from .segmentation_data import SegmentationDataset, manifest_records
from .train import load_yaml_config, resolve_cache_dir
from .train_segmentation import check_prerequisites, read_json, validate_config

EXPERIMENT = "decoder_padding_comparison"


def boundary_summary(distributions, crops):
    """Recompose exact .5 counts from full-voxel compartment fractions, not quantiles."""
    identities = {int(row["pool_index"]): row for row in crops}
    totals = {}
    for row in distributions:
        if row["compartment"] == "all":
            continue
        crop = identities[int(row["pool_index"])]
        groups = ["positive_crops" if int(crop["target_voxels"]) else "negative_crops"]
        if crop["mod"] == "real":
            groups.append("real_crops")
        n = int(row["n_voxels"])
        detected = round(n * float(row["fraction_ge_0.5"])) if n else 0
        for group in groups:
            total = totals.setdefault((group, row["region"]), dict(target_voxels=0, background_voxels=0, tp=0, fp=0))
            if row["compartment"] == "foreground":
                total["target_voxels"] += n
                total["tp"] += detected
            else:
                total["background_voxels"] += n
                total["fp"] += detected
    rows = []
    for (group, region), total in totals.items():
        target, background, tp, fp = (total[k] for k in ("target_voxels", "background_voxels", "tp", "fp"))
        rows.append({"group": group, "region": region, **total, "fn": target - tp,
                     "recall": tp / target if target else None,
                     "background_fp_fraction": fp / background if background else None})
    return rows


def run_comparison(config, data_root, *, device="cpu", reference_dir="outputs/segmentation_batch_comparison_squared",
                   output_dir="outputs/segmentation_padding_comparison", cache_dir=None, real_scan_root=None):
    config = copy.deepcopy(config)
    validate_config(config)
    training = config["training"]
    if (config["stage"] != "overfit" or not training.get("squared_pred") or training["epochs"] > 50
            or training["max_updates"] != 6 * training["epochs"]
            or config["model"].get("decoder_padding_mode", "zeros") != "zeros"):
        raise ValueError("padding comparison needs the zero-padding squared overfit, max_updates = 6 * epochs <= 300")
    source, output = Path(reference_dir), Path(output_dir)
    if output.resolve() == source.resolve() or source.resolve() in output.resolve().parents:
        raise ValueError("comparison output must be outside the reference directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError("comparison output already contains files; choose a new --output-dir")
    previous = read_json(source / "comparison.json")
    manifest = read_json(config["manifest"])
    reference_config = read_json(source / "effective_config.json")
    for cfg in (config, reference_config):
        cfg["model"].setdefault("decoder_padding_mode", "zeros")
    if (previous.get("experiment") != "batch_composition_comparison" or not previous.get("complete")
            or previous.get("error") or previous.get("active_arm") is not None
            or set(previous.get("arms", {})) != {"grouped", "mixed"}
            or previous.get("manifest_id") != manifest["manifest_id"]
            or read_json(source / "manifest.json") != manifest or reference_config != config):
        raise ValueError("reference must be the complete squared grouped/mixed comparison with identical config/manifest")
    initial_hash = previous["arms"]["mixed"].get("initial_state_sha256")
    if not initial_hash or previous["arms"]["grouped"].get("initial_state_sha256") != initial_hash:
        raise ValueError("reference initial model states differ or are missing")
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
    if ledger != pool.rows or pool.digest != previous.get("crop_pool_sha256"):
        raise ValueError("regenerated crop identities/image/mask hashes differ from the reference comparison")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "crop_pool.csv", pool.rows)
    write_json(output / "effective_config.json", config)
    write_json(output / "manifest.json", manifest)
    report = {"experiment": EXPERIMENT, "complete": False, "passed": False,
              "manifest_id": manifest["manifest_id"], "crop_pool_sha256": pool.digest,
              "reference_dir": str(source), "reference_comparison_sha256": file_digest(source / "comparison.json"),
              "initial_state_sha256": initial_hash, "active_arm": None, "active_checkpoint": None, "arms": {},
              "runtime": {"torch": torch.__version__, "monai": version("monai"), "device": str(device)},
              "scope": "48 fixed training crops; .5 gate unchanged; no pilot/full promotion or validation calibration"}
    try:
        for padding in ("zeros", "replicate"):
            report["active_arm"] = padding
            write_json(output / "comparison.json", report)
            arm_config = copy.deepcopy(config)
            arm_config["model"]["decoder_padding_mode"] = padding
            print(f"starting decoder padding comparison: {padding}", flush=True)
            folder = output / padding
            arm = run_arm(arm_config, manifest, pool, folder, mixed=True, device=device,
                          expected_initial_state_sha256=initial_hash, experiment=EXPERIMENT)
            report["arms"][padding] = {**arm, "probability_diagnostics": {}}
            for checkpoint in ("best", "last"):
                report["active_checkpoint"] = f"{padding}/{checkpoint}"
                write_json(output / "comparison.json", report)
                metrics_dir = folder / "best" if checkpoint == "best" else folder
                crops = read_rows(metrics_dir / "crops.csv")
                diagnosis_dir = folder / "probabilities" / checkpoint
                diagnosis = evaluate_checkpoint(folder / f"{checkpoint}.pt", pool, arm_config, manifest, arm,
                    checkpoint, crops, diagnosis_dir, device, experiment=EXPERIMENT)
                regions = boundary_summary(read_rows(diagnosis_dir / "distributions.csv"), pool.rows)
                write_csv(diagnosis_dir / "boundary_metrics.csv", regions)
                report["arms"][padding]["probability_diagnostics"][checkpoint] = diagnosis
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
            report["active_checkpoint"] = None
        zeros, replicate = (report["arms"][name] for name in ("zeros", "replicate"))
        report.update(complete=True, active_arm=None, passed=replicate["last"]["passed"],
                      delta_replicate_minus_zeros={checkpoint: {key: replicate[checkpoint][key] - zeros[checkpoint][key]
                          for key in ("positive_crop_dice", "negative_crop_fp_fraction")}
                          for checkpoint in ("best", "last")})
        return report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(output / "comparison.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/segmentation_overfit_squared.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--real-scan-root")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reference-dir", default="outputs/segmentation_batch_comparison_squared")
    parser.add_argument("--output-dir", default="outputs/segmentation_padding_comparison")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    report = run_comparison(load_yaml_config(args.config), args.data_root, device=args.device,
        reference_dir=args.reference_dir, output_dir=args.output_dir,
        cache_dir=args.cache_dir, real_scan_root=args.real_scan_root)
    print(f"Padding comparison complete: replicate last gate passed={report['passed']}")
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
