"""Check weight updates, memorize fixed crops, then gate the six-scan overfit."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from .diagnose_localization import write_json
from .segmentation import build_segmenter, segmentation_loss_components
from .segmentation_data import SegmentationDataset, manifest_records
from .segmentation_metrics import preview_planes, save_preview
from .train import _write_metrics_row, load_yaml_config, resolve_cache_dir, set_global_seed
from .train_segmentation import (check_prerequisites, make_optimizer, overfit_rank, read_json, run_training,
                                 save_checkpoint, train_volume, validate_config)


def measure_fixed(model, batch, device, microbatch, output=None):
    """Use the gate's float32 evaluation and fixed 0.5 threshold."""
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(batch["image"][i:i + microbatch].to(device)).float().cpu()
                            for i in range(0, len(batch["image"]), microbatch)])
    if not torch.isfinite(logits).all():
        raise FloatingPointError("non-finite fixed-crop prediction")
    target = batch["mask"].bool()
    probability = logits.sigmoid()
    detected = probability >= 0.5
    truth, pred = target[0], detected[0]
    dice = float(2 * (truth & pred).sum() / (truth.sum() + pred.sum()))
    fpr = float(detected[1:].float().mean()) if len(target) > 1 else None
    bce, soft_dice = segmentation_loss_components(logits, target)
    report = {"positive_crop_dice": dice, "negative_crop_fp_fraction": fpr,
              "bce": float(bce), "soft_dice_loss": float(soft_dice),
              "passed": dice >= 0.9 and (fpr is None or fpr < 0.01)}
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        for index, kind in enumerate(("positive", "negative")[:len(target)]):
            planes = preview_planes(batch["image"][index, 0].numpy(), target[index, 0].numpy(), probability[index, 0].numpy())
            save_preview(output / f"{kind}.png", planes, 0.5)
    return report


def run_fixed_stage(config, manifest, batch, output, *, device="cpu"):
    """Fresh MedicalNet initialization per stage; diagnostic weights never unlock pilots."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    set_global_seed(training.get("seed", 21))
    model = build_segmenter(config["model"]).to(device)
    optimizer = make_optimizer(model, training)
    microbatch = min(training["batch_size"], len(batch["image"]))
    accumulation = len(batch["image"]) // microbatch
    running = {name: value.clone() for name, value in model.named_buffers()
               if name.endswith(("running_mean", "running_var", "num_batches_tracked"))}
    best, update_check = None, None
    report = measure_fixed(model, batch, device, microbatch, output / "initial_images")
    _write_metrics_row(output / "checks.csv", {"updates": 0, **report})

    def payload(updates, metrics):
        return {"task": "segmentation_debug", "config": config, "manifest": manifest,
                "model_state_dict": model.state_dict(), "updates": updates, "metrics": metrics}

    for updates in range(1, training.get("max_updates", 300) + 1):
        model.train()
        diagnostics = {}
        loss = train_volume(model, batch, optimizer, device=device, microbatch=microbatch,
                            accumulation=accumulation, diagnostics=diagnostics)
        _write_metrics_row(output / "updates.csv", {"updates": updates, "loss": loss, **diagnostics})
        if updates == 1:
            unchanged = all(torch.equal(value, dict(model.named_buffers())[name]) for name, value in running.items())
            update_check = {**diagnostics, "batchnorm_statistics_unchanged": unchanged,
                            "passed": unchanged and all(diagnostics[f"{group}_{kind}_norm"] > 0
                                                         for group in ("encoder", "decoder") for kind in ("grad", "update"))}
            write_json(output / "update_check.json", update_check)
        if updates == 1 or updates % 25 == 0 or updates == training.get("max_updates", 300):
            report = measure_fixed(model, batch, device, microbatch)
            report["passed"] = report["passed"] and update_check["passed"]
            _write_metrics_row(output / "checks.csv", {"updates": updates, **report})
            if best is None or overfit_rank(report) > overfit_rank(best):
                best = {"updates": updates, **report}
                save_checkpoint(payload(updates, report), output / "best.pt")
                write_json(output / "best_metrics.json", best)
            print(json.dumps({"stage": output.name, "updates": updates, **report}), flush=True)
            if report["passed"] or not update_check["passed"]:
                break
    report = {"updates": updates, **measure_fixed(model, batch, device, microbatch, output / "images")}
    report["passed"] = report["passed"] and update_check["passed"]
    save_checkpoint(payload(updates, report), output / "last.pt")
    write_json(output / "result.json", report)
    return report, update_check


def run_debug(config, data_root, *, device="cpu", output_dir="outputs/segmentation_debug",
              cache_dir=None, real_scan_root=None):
    config = copy.deepcopy(config)
    validate_config(config)
    if config["stage"] != "overfit":
        raise ValueError("debug requires the original overfit configuration")
    manifest = read_json(config["manifest"])
    check_prerequisites(config, manifest, data_root, real_scan_root)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("debug output already contains files; use a new --output-dir")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "effective_config.json", config)
    write_json(output / "manifest.json", manifest)
    records = manifest_records(manifest, "overfit")
    if len(records) != 6:
        raise ValueError("debug requires the fixed six-scan overfit cohort")
    dataset = SegmentationDataset(records, data_root, patch_shape=config["patches"]["patch_shape"],
                                  preprocessing=config["preprocessing"], seed=manifest["seed"], fixed=True, augment=False,
                                  cache_dir=resolve_cache_dir(config, cache_dir, output), volume_cache_size=1,
                                  real_scan_root=real_scan_root)
    if not records or records[0].is_real:
        raise ValueError("the first overfit record must be manipulated")
    source = dataset[(0, 0)]
    positive = source["mask"].flatten(1).any(1)
    if not positive[0] or positive[4]:
        raise ValueError("expected positive crop 0 and clean negative crop 4")
    pair = {key: source[key][[0, 4]].clone() for key in ("image", "mask")}
    if not torch.isfinite(pair["image"]).all():
        raise ValueError("non-finite preprocessed fixed crop")
    del dataset, source
    selection = {"record": manifest["cohorts"]["overfit"][0], "crop_indices": [0, 4],
                 "threshold": 0.5, "selection": "first manipulated scan, first positive and first negative crop",
                 "image_sha256": hashlib.sha256(pair["image"].numpy().tobytes()).hexdigest(),
                 "mask_sha256": hashlib.sha256(pair["mask"].numpy().tobytes()).hexdigest(),
                 "target_voxels": [int(t.sum()) for t in pair["mask"]]}
    write_json(output / "selection.json", selection)
    report = {"task": "segmentation_debug", "manifest_id": manifest["manifest_id"],
              "passed": False, "stopped_at": "positive", "stages": {}}
    try:
        for name, count in (("positive", 1), ("positive_negative", 2)):
            report["stopped_at"] = name
            batch = {key: value[:count] for key, value in pair.items()}
            result, update_check = run_fixed_stage(config, manifest, batch, output / name, device=device)
            report["stages"][name] = result
            report["stages"][name + "_update_check"] = update_check
            if not update_check["passed"]:
                report["stopped_at"] = name + "_update_check"
                return report
            if not result["passed"]:
                return report
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        report["stopped_at"] = "six_scans"
        result = run_training(config, data_root, device=device, output_dir=output / "six_scans",
                               cache_dir=cache_dir, real_scan_root=real_scan_root, debug_updates=True)
        report["stages"]["six_scans"] = result
        report["passed"] = result["passed"]
        if report["passed"]:
            report["stopped_at"] = None
        return report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/segmentation_overfit.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--real-scan-root")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/segmentation_debug")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    report = run_debug(load_yaml_config(args.config), args.data_root, device=args.device, output_dir=args.output_dir,
                       cache_dir=args.cache_dir, real_scan_root=args.real_scan_root)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
