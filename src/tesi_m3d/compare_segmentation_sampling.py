"""Equal-budget pilots: current sampling versus paired negatives or aligned windows."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

from .dataset import read_records, scan_dir
from .diagnose_localization import write_csv, write_json
from .segmentation import build_segmenter, file_digest
from .segmentation_data import manifest_records, validate_stack, validate_experiment_output
from .train import load_yaml_config, set_global_seed
from .train_segmentation import check_prerequisites, read_json, run_training, validate_config


def resolve_pairs(records, metadata):
    real = {}
    for record in metadata:
        if record.is_real:
            if record.img_id in real and real[record.img_id] != record:
                raise ValueError("ambiguous original CT metadata")
            real[record.img_id] = record
    pairs = []
    for record in records:
        paired = record if record.is_real else real.get(record.img_id)
        if (paired is None or record.split != "train" or paired.split != "train" or
                (record.orig_id, record.sdir_id) != (paired.orig_id, paired.sdir_id)):
            raise ValueError(f"missing or mismatched train original: {record.img_id}")
        pairs.append(paired)
    return pairs


def verify_sampling(output, expected_updates, variant="paired", patch_shape=None):
    traces = []
    for arm in ("standard", variant):
        with (output / arm / "sampling.csv").open(newline="") as handle:
            traces.append(list(csv.DictReader(handle)))
    if any(len(rows) != expected_updates for rows in traces):
        raise ValueError("unequal or incomplete training budgets")
    fields = ("update", "epoch", "img_id", "crops", "positive_crops", "retained_image_sha256", "mask_sha256")
    if variant == "grid":
        fields = ("update", "epoch", "img_id", "crops", "positive_crops", "flips", "volume_shape")
    for index, (a, b) in enumerate(zip(*traces), 1):
        if int(a["update"]) != index or a["crops"] != "8" or any(a[k] != b[k] for k in fields):
            raise ValueError(f"sampling comparison is not controlled at update {index}")
        if variant == "grid":
            from monai.data.utils import dense_patch_slices
            shape = json.loads(b["volume_shape"])
            intervals = tuple(max(1, int(p * 0.5)) if n != p else p for n, p in zip(shape, patch_shape))
            allowed = {tuple(s.start for s in w) for w in dense_patch_slices(shape, tuple(patch_shape), intervals)}
            coords = json.loads(b["crop_starts"])
            if len(coords) != 8 or any(tuple(c) not in allowed for c in coords):
                raise ValueError("training crops are not on the inference grid")
        if variant == "paired" and a["positive_crops"] == "0" and a["image_sha256"] != b["image_sha256"]:
            raise ValueError("real-volume crops changed between arms")
    hashes = [read_json(output / arm / "initialization.json")["state_sha256"] for arm in ("standard", variant)]
    if hashes[0] != hashes[1]:
        raise ValueError("initial weights differ")
    return {"updates_per_arm": expected_updates, "crops_per_arm": expected_updates * 8,
            "initial_state_sha256": hashes[0], **({"retained_crops_and_targets_identical": True} if variant == "paired" else
               {"volume_order_class_counts_and_flips_identical": True, "all_starts_on_inference_grid": True,
                "positive_voxel_presentations": {arm: sum(int(r["positive_voxels"]) for r in rows)
                                                for arm, rows in zip(("standard", variant), traces)},
                "unique_starts_per_update_mean": {arm: sum(len({tuple(c) for c in json.loads(r["crop_starts"])}) for r in rows) / len(rows)
                                                  for arm, rows in zip(("standard", variant), traces)}})}


def run_comparison(config, data_root, *, window_report, output_dir="outputs/segmentation_sampling_comparison",
                   device="cpu", real_scan_root=None, cache_dir=None, variant="paired", sampling_report=None):
    if variant not in ("paired", "grid"):
        raise ValueError("variant must be paired or grid")
    config = copy.deepcopy(config)
    validate_config(config)
    if config["stage"] != "pilot" or config.get("sampling_protocol", "standard") != "standard":
        raise ValueError("use the unmodified recovery pilot config")
    manifest = read_json(config["manifest"])
    diagnostic = read_json(window_report)
    if not diagnostic.get("complete") or diagnostic.get("task") != "segmentation_windows" or diagnostic.get("manifest_id") != manifest["manifest_id"]:
        raise ValueError("complete the window diagnostic on this manifest first")
    if variant == "grid":
        prior = read_json(sampling_report) if sampling_report else {}
        if not prior.get("complete") or prior.get("task") != "segmentation_sampling" or prior.get("manifest_id") != manifest["manifest_id"]:
            raise ValueError("grid experiment requires the completed phase-2 sampling report")
    check_prerequisites(config, manifest, data_root, real_scan_root)
    output = Path(output_dir)
    # Protect sources, including user-specified output ancestors.
    protected = [Path(config["output_dir"]), Path(window_report).parent, Path(config["manifest"]).parent,
                 Path(config["overfit_report"]).parent]
    if sampling_report:
        protected.append(Path(sampling_report).parent)
    validate_experiment_output(output, protected, data_root, real_scan_root)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output is not empty; choose a new directory (no partial-run resume)")
    records = manifest_records(manifest, "pilot_train")
    pairs = resolve_pairs(records, read_records(config.get("metadata_dir", "metadata/m3dsynth"))) if variant == "paired" else []
    pair_rows = []
    for record, paired in zip(records, pairs):
        if record.is_real:
            continue
        fake_path, real_path = scan_dir(data_root, record, real_scan_root), scan_dir(data_root, paired, real_scan_root)
        if validate_stack(fake_path) != validate_stack(real_path):
            raise ValueError(f"paired native shapes differ: {record.img_id}")
        pair_rows.append({"img_id": record.img_id, "orig_id": record.orig_id, "sdir_id": record.sdir_id,
                          "fake_path": str(fake_path), "real_path": str(real_path)})
    output.mkdir(parents=True, exist_ok=True)
    if variant == "paired":
        write_csv(output / "pairs.csv", pair_rows)
    # Disable early stopping symmetrically so both arms spend the same budget.
    config["training"]["early_stopping_validations"] = config["training"]["epochs"] + 1
    set_global_seed(config["training"].get("seed", 21))
    model = build_segmenter(config["model"])
    initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    del model
    report = {"task": "segmentation_sampling" if variant == "paired" else "segmentation_geometry", "complete": False, "manifest_id": manifest["manifest_id"],
              "window_report_sha256": file_digest(window_report), "arms": {},
              "scope": "two fresh pilots; paired negatives replace crop slots 4/5; global slots 6/7 retained",
              "limitations": "single seed; geometry checks do not certify anatomical registration; no full promotion"}
    if variant == "grid":
        report.update(sampling_report_sha256=file_digest(sampling_report),
                      scope="two fresh pilots; nearest same-class MONAI windows, overlap 0.5; unchanged flips and 4 positive/4 negative slots; no pairing or consistency loss",
                      limitations="single seed; aligned starts are not uniform grid sampling; foreground fractions and lung-negative content can change; duplicates allowed; no small-volume padding; no full promotion")
    write_json(output / "comparison.json", report)
    try:
        for arm in ("standard", variant):
            report["active_arm"] = arm
            write_json(output / "comparison.json", report)
            current = copy.deepcopy(config)
            current["sampling_protocol"] = arm
            current["output_dir"] = str(output / arm)
            report["arms"][arm] = run_training(
                current, data_root, device=device, output_dir=output / arm, cache_dir=cache_dir,
                real_scan_root=real_scan_root, initial_state=initial_state,
                paired_records=pairs if arm == "paired" else None, sampling_audit=True)
        report["control"] = verify_sampling(output, len(records) * config["training"]["epochs"], variant, config["patches"]["patch_shape"])
        metrics = ("dense_ap", "mean_positive_ap", "micro_dice", "mean_positive_dice", "peak_hits", "real_false_positive_fraction")
        report[variant + "_minus_standard"] = {k: report["arms"][variant][k] - report["arms"]["standard"][k] for k in metrics}
        report["complete"] = True
        report.pop("active_arm", None)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(output / "comparison.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/segmentation_pilot_recovery.yaml")
    parser.add_argument("--window-report", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--real-scan-root")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir")
    parser.add_argument("--variant", choices=("paired", "grid"), default="paired")
    parser.add_argument("--sampling-report", help="completed phase-2 comparison.json, required for grid")
    parser.add_argument("--output-dir")
    args = vars(parser.parse_args())
    if args["output_dir"] is None:
        args["output_dir"] = "outputs/segmentation_geometry_comparison" if args["variant"] == "grid" else "outputs/segmentation_sampling_comparison"
    config = load_yaml_config(args.pop("config"))
    run_comparison(config, **args)


if __name__ == "__main__":
    main()
