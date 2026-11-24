"""Controlled mixed-batch overfit: padding, deterministic repeats or late LR decay."""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager, nullcontext
from importlib.metadata import version
import os
from pathlib import Path
import platform

import torch

from .compare_segmentation_batches import FixedCropPool, run_arm
from .diagnose_localization import write_csv, write_json
from .diagnose_segmentation_batch_probabilities import evaluate_checkpoint, read_rows
from .segmentation import file_digest
from .segmentation_data import SegmentationDataset, manifest_records
from .train import load_yaml_config, resolve_cache_dir
from .train_segmentation import check_prerequisites, read_json, validate_config

EXPERIMENT = "decoder_padding_comparison"
DETERMINISTIC_EXPERIMENT = "deterministic_overfit_repeats"
LR_EXPERIMENT = "deterministic_late_lr_decay"


@contextmanager
def deterministic_execution():
    previous = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in (None, ":4096:8", ":16:8"):
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8")
    if workspace is None and torch.cuda.is_initialized():
        raise RuntimeError("start a fresh process with CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization")
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = previous[2:]
        if workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace


def compare_repeats(output, updates):
    files = ["update_metrics.csv", "overfit_metrics.csv", "case_history.csv", "crops.csv", "best/crops.csv"]
    files += [f"probabilities/{cp}/{name}.csv" for cp in ("best", "last")
              for name in ("crops", "thresholds", "boundary_metrics")]
    checks, first = {}, None
    for relative in files:
        left, right = (read_rows(output / arm / relative) for arm in ("repeat_1", "repeat_2"))
        if relative == "update_metrics.csv" and (len(left) != updates or len(right) != updates):
            raise ValueError("both repeats must contain the full update trace")
        checks[relative] = left == right
        if left != right and first is None:
            index = next((i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]), min(len(left), len(right)))
            a, b = left[index] if index < len(left) else {}, right[index] if index < len(right) else {}
            first = {"file": relative, "row": index + 1, "updates": a.get("updates", b.get("updates")),
                     "differences": {key: {"repeat_1": a.get(key), "repeat_2": b.get(key)}
                                     for key in a.keys() | b.keys() if a.get(key) != b.get(key)}}
    return {"reproducible": all(checks.values()), "exact_file_matches": checks, "first_divergence": first}


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


def run_comparison(config, data_root, *, device="cpu", reference_dir=None,
                   output_dir=None, cache_dir=None, real_scan_root=None,
                   deterministic_repeats=False, lr_decay=False):
    if deterministic_repeats and lr_decay:
        raise ValueError("choose deterministic repeats or LR decay, not both")
    reference_dir = reference_dir or ("outputs/segmentation_determinism_comparison" if lr_decay
                                     else "outputs/segmentation_batch_comparison_squared")
    output_dir = output_dir or ("outputs/segmentation_lr_decay_comparison" if lr_decay else
                               "outputs/segmentation_determinism_comparison" if deterministic_repeats
                                else "outputs/segmentation_padding_comparison")
    with deterministic_execution() if deterministic_repeats or lr_decay else nullcontext():
        return _run_comparison(config, data_root, device=device, reference_dir=reference_dir, output_dir=output_dir,
                              cache_dir=cache_dir, real_scan_root=real_scan_root,
                              deterministic_repeats=deterministic_repeats, lr_decay=lr_decay)


def _run_comparison(config, data_root, *, device, reference_dir, output_dir, cache_dir, real_scan_root,
                    deterministic_repeats, lr_decay):
    config = copy.deepcopy(config)
    validate_config(config)
    training = config["training"]
    if (config["stage"] != "overfit" or not training.get("squared_pred") or training["epochs"] > 50
            or training["max_updates"] != 6 * training["epochs"]
            or config["model"].get("decoder_padding_mode", "zeros") != "zeros"):
        raise ValueError("padding comparison needs the zero-padding squared overfit, max_updates = 6 * epochs <= 300")
    if lr_decay and training["max_updates"] != 300:
        raise ValueError("LR decay comparison requires exactly 300 updates and 50 epochs")
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
    reference_labels = ("repeat_1", "repeat_2") if lr_decay else ("grouped", "mixed")
    if (previous.get("experiment") != (DETERMINISTIC_EXPERIMENT if lr_decay else "batch_composition_comparison")
            or not previous.get("complete")
            or previous.get("error") or previous.get("active_arm") is not None
            or previous.get("active_checkpoint") is not None
            or set(previous.get("arms", {})) != set(reference_labels)
            or previous.get("manifest_id") != manifest["manifest_id"]
            or read_json(source / "manifest.json") != manifest or reference_config != config):
        raise ValueError("reference must be the complete matching comparison with identical config/manifest")
    if lr_decay and (previous.get("reproducible") is not True
                     or previous.get("runtime", {}).get("deterministic_algorithms") is not True
                     or previous.get("runtime", {}).get("warn_only") is not False
                     or previous.get("runtime", {}).get("cudnn_benchmark") is not False
                     or any(previous["arms"][name].get("deterministic") is not True for name in reference_labels)
                     or not compare_repeats(source, training["max_updates"])["reproducible"]):
        raise ValueError("LR decay requires a reproducible deterministic reference, including exact saved traces")
    initial_hash = previous["arms"][reference_labels[1]].get("initial_state_sha256")
    if not initial_hash or previous["arms"][reference_labels[0]].get("initial_state_sha256") != initial_hash:
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
    strict = deterministic_repeats or lr_decay
    experiment = LR_EXPERIMENT if lr_decay else DETERMINISTIC_EXPERIMENT if deterministic_repeats else EXPERIMENT
    labels = ("decay",) if lr_decay else ("repeat_1", "repeat_2") if deterministic_repeats else ("zeros", "replicate")
    report = {"experiment": experiment, "complete": False, "passed": False,
              "manifest_id": manifest["manifest_id"], "crop_pool_sha256": pool.digest,
              "reference_dir": str(source), "reference_comparison_sha256": file_digest(source / "comparison.json"),
              "initial_state_sha256": initial_hash, "active_arm": None, "active_checkpoint": None, "arms": {},
              "runtime": {"torch": torch.__version__, "monai": version("monai"), "device": str(device)},
              "scope": "48 fixed training crops; .5 gate unchanged; no pilot/full promotion or validation calibration"}
    if strict:
        if deterministic_repeats:
            report["reproducible"] = None
        report["runtime"].update(python=platform.python_version(), cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(), cpu_threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(),
            gpu=torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None,
            gpu_capability=list(torch.cuda.get_device_capability(device)) if str(device).startswith("cuda") else None,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            pythonhashseed=os.environ.get("PYTHONHASHSEED"), trilinear_backward="cpu_first_order",
            maxpool3d_backward="cpu_first_order_native_indices",
            comparison="exact traces/weight and gradient hashes/metrics within this process; not bitwise equivalence to historical runs")
    if lr_decay:
        report.update(reference_control=previous["arms"]["repeat_1"], reference_prefix=None)
        report["runtime"]["comparison"] = "exact first 200 updates against the saved constant-LR control; late decay is the only intended change"
    try:
        for label in labels:
            padding = "zeros" if strict else label
            report["active_arm"] = label
            write_json(output / "comparison.json", report)
            arm_config = copy.deepcopy(config)
            arm_config["model"]["decoder_padding_mode"] = padding
            print(f"starting {experiment}: {label}, padding={padding}", flush=True)
            folder = output / label
            arm = run_arm(arm_config, manifest, pool, folder, mixed=True, device=device,
                          expected_initial_state_sha256=initial_hash, experiment=experiment, deterministic=strict,
                          **({"lr_decay_reference": read_rows(source / "repeat_1/update_metrics.csv")} if lr_decay else {}))
            report["arms"][label] = {**arm, "probability_diagnostics": {}}
            for checkpoint in ("best", "last"):
                report["active_checkpoint"] = f"{label}/{checkpoint}"
                write_json(output / "comparison.json", report)
                metrics_dir = folder / "best" if checkpoint == "best" else folder
                crops = read_rows(metrics_dir / "crops.csv")
                diagnosis_dir = folder / "probabilities" / checkpoint
                diagnosis = evaluate_checkpoint(folder / f"{checkpoint}.pt", pool, arm_config, manifest, arm,
                    checkpoint, crops, diagnosis_dir, device, experiment=experiment)
                regions = boundary_summary(read_rows(diagnosis_dir / "distributions.csv"), pool.rows)
                write_csv(diagnosis_dir / "boundary_metrics.csv", regions)
                report["arms"][label]["probability_diagnostics"][checkpoint] = diagnosis
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
            report["active_checkpoint"] = None
        if lr_decay:
            arm, control = report["arms"]["decay"], report["reference_control"]
            report.update(complete=True, active_arm=None, reference_prefix=arm["reference_prefix"],
                          passed=arm["reference_prefix"]["passed"] and arm["last"]["passed"],
                          delta_decay_minus_constant={checkpoint: {key: arm[checkpoint][key] - control[checkpoint][key]
                              for key in ("positive_crop_dice", "negative_crop_fp_fraction")}
                              for checkpoint in ("best", "last")})
        elif deterministic_repeats:
            report.update(compare_repeats(output, training["max_updates"]))
            report.update(complete=True, active_arm=None,
                          passed=report["reproducible"] and all(a["last"]["passed"] for a in report["arms"].values()))
        else:
            zeros, replicate = (report["arms"][name] for name in labels)
            report.update(complete=True, active_arm=None, passed=replicate["last"]["passed"],
                          delta_replicate_minus_zeros={checkpoint: {key: replicate[checkpoint][key] - zeros[checkpoint][key]
                              for key in ("positive_crop_dice", "negative_crop_fp_fraction")}
                              for checkpoint in ("best", "last")})
        return report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if lr_decay and (output / "decay/prefix_check.json").is_file():
            report["reference_prefix"] = read_json(output / "decay/prefix_check.json")
        write_json(output / "comparison.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/segmentation_overfit_squared.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--real-scan-root")
    parser.add_argument("--device", default="cpu")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--deterministic-repeats", action="store_true",
                      help="two identical zero-padding runs, strict determinism and explicit CPU spatial backward")
    mode.add_argument("--lr-decay", action="store_true",
                      help="one strict run: verify the saved constant-LR prefix, divide both LRs by 10 after update 200")
    parser.add_argument("--reference-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    report = run_comparison(load_yaml_config(args.config), args.data_root, device=args.device,
        reference_dir=args.reference_dir, output_dir=args.output_dir or
            ("outputs/segmentation_lr_decay_comparison" if args.lr_decay else
             "outputs/segmentation_determinism_comparison" if args.deterministic_repeats else "outputs/segmentation_padding_comparison"),
        cache_dir=args.cache_dir, real_scan_root=args.real_scan_root,
        deterministic_repeats=args.deterministic_repeats, lr_decay=args.lr_decay)
    if args.lr_decay:
        print(f"LR decay comparison complete: prefix exact={report['reference_prefix']['passed']}, last gate passed={report['passed']}")
    elif args.deterministic_repeats:
        print(f"Deterministic comparison complete: reproducible={report['reproducible']}, localization gate passed={report['passed']}")
        if not report["reproducible"]:
            raise SystemExit(3)
    else:
        print(f"Padding comparison complete: replicate last gate passed={report['passed']}")
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
