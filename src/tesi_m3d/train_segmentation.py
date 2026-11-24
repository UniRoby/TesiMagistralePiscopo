"""Gated overfit -> two pilots -> full supervised MedicalNet segmentation."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from tqdm import tqdm

from .diagnose_localization import write_csv, write_json
from .evaluate_segmentation import run_evaluation
from .segmentation import SEGMENTER_NAME, build_segmenter, file_digest, load_predictor, segmentation_loss_components
from .segmentation_data import SegmentationDataset, VolumeOrder, manifest_records
from .segmentation_metrics import cohort_id, compare_baseline, evaluate_records, preview_planes, save_preview
from .train import _load_torch_payload, _write_metrics_row, load_yaml_config, resolve_cache_dir, set_global_seed


def read_json(path):
    return json.loads(Path(path).read_text())


def save_checkpoint(payload, path):
    """Keep the previous complete checkpoint if writing is interrupted."""
    path = Path(path)
    temporary = path.with_suffix(".pt.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def overfit_rank(metrics):
    return metrics["passed"], metrics["positive_crop_dice"], -(metrics["negative_crop_fp_fraction"] or 0)


def check_prerequisites(config, manifest, data_root=None, real_scan_root=None):
    """Reports are bound to the cohort manifest, never reused across protocols."""
    stage, protocol = config["stage"], Path(config["manifest"]).parent
    audit = read_json(protocol / "audit.json")
    if data_root is not None and Path(audit.get("data_root", "")).resolve() != Path(data_root).resolve():
        raise ValueError("data root differs from the audited dataset; rerun prepare_segmentation")
    if data_root is not None:
        real_path = Path(real_scan_root or Path(data_root) / "real/scan").resolve()
        audited_real = Path(audit.get("real_scan_root") or Path(audit["data_root"]) / "real/scan").resolve()
        if audited_real != real_path:
            raise ValueError("real scan root differs from the audited dataset")
    required = "full_train" if stage == "full" else "overfit" if stage == "overfit" else "pilot_train"
    if not audit.get("passed") or audit.get("manifest_id") != manifest["manifest_id"] or required not in audit["cohorts"]:
        raise ValueError(f"run prepare_segmentation --stage {'full' if stage == 'full' else 'pilot'} for this manifest first")
    if stage in {"pilot", "full"}:
        overfit = read_json(config["overfit_report"])
        if (not overfit.get("passed") or overfit.get("manifest_id") != manifest["manifest_id"] or
                overfit.get("positive_crop_dice", 0) < 0.9 or overfit.get("negative_crop_fp_fraction", 1) >= 0.01 or
                not overfit.get("finite_gradients")):
            raise ValueError("overfit gate failed or belongs to a different manifest; full/pilot training blocked")
        if overfit.get("training_loss", {}).get("squared_pred", False) != config.get("training", {}).get("squared_pred", False):
            raise ValueError("overfit gate used a different Dice denominator; full/pilot training blocked")
        if overfit.get("decoder_padding_mode", "zeros") != config.get("model", {}).get("decoder_padding_mode", "zeros"):
            raise ValueError("overfit gate used a different decoder padding; full/pilot training blocked")
    if stage == "full":
        pilots = [read_json(path) for path in config["pilot_reports"]]
        if len(pilots) != 2 or {p["preprocessing"]["clip"] for p in pilots} != {False, True}:
            raise ValueError("full training requires both controlled pilots, with and without clipping")
        if any(p.get("manifest_id") != manifest["manifest_id"] or p.get("split") != "valid" for p in pilots):
            raise ValueError("pilot reports must use this validation manifest")
        signatures = {p.get("protocol_signature") for p in pilots}
        if len(signatures) != 1 or None in signatures:
            raise ValueError("pilots must differ only in clipping, not training/model settings")
        baseline_pilot = read_json(config["pilot_baseline_report"])
        expected_pilot = cohort_id(manifest_records(manifest, "pilot_valid"))
        if baseline_pilot.get("cohort_id") != expected_pilot:
            raise ValueError("pilot baseline cohort differs from manifest")
        eligible = [p for p in pilots if compare_baseline(p, baseline_pilot)["passed"]]
        if not eligible:
            raise ValueError("neither pilot passed the localization gate; full training blocked")
        chosen = max(eligible, key=lambda p: (p["dense_ap"], not p["preprocessing"]["clip"]))
        config["preprocessing"] = copy.deepcopy(chosen["preprocessing"])
        config["selected_pilot_sha256"] = chosen["checkpoint_sha256"]
    if stage in {"pilot", "full"}:
        baseline = read_json(config["baseline_report"])
        if baseline.get("manifest_id") != manifest["manifest_id"] or baseline.get("split") != "valid":
            raise ValueError("baseline report must use this validation manifest")
        expected = cohort_id(manifest_records(manifest, "common_valid" if stage == "full" else "pilot_valid"))
        if baseline.get("cohort_id") != expected or baseline.get("model_name") != "resnet18_medicalnet":
            raise ValueError("baseline must be the MedicalNet classifier on the required fixed cohort")


def make_loader(dataset, training, device):
    workers = int(training.get("num_workers", 4))
    if workers < 0:
        raise ValueError("num_workers cannot be negative")
    order = VolumeOrder(len(dataset), int(training.get("seed", 21)))
    options = dict(batch_size=None, sampler=order, num_workers=workers,
                   pin_memory=str(device).startswith("cuda"))
    if workers:
        options.update(persistent_workers=True, prefetch_factor=1)
    return torch.utils.data.DataLoader(dataset, **options), order


def make_optimizer(model, training):
    decoder = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    return torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": training.get("encoder_lr", 3e-5)},
                              {"params": decoder, "lr": training.get("decoder_lr", 3e-4)}],
                             weight_decay=training.get("weight_decay", 1e-4))


def train_volume(model, batch, optimizer, *, device="cpu", microbatch=2, accumulation=4, diagnostics=None,
                 foreground_normalizer=None, squared_pred=False):
    count = len(batch["image"])
    if microbatch < 1 or accumulation < 1 or count != microbatch * accumulation or len(batch["mask"]) != count:
        raise ValueError("crop count must match microbatch * accumulation")
    optimizer.zero_grad(set_to_none=True)
    losses, components = [], []
    before = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad} if diagnostics is not None else None
    n_positive = int(batch["mask"].flatten(1).any(1).sum())
    if foreground_normalizer is not None and (not np.isfinite(foreground_normalizer) or foreground_normalizer <= 0):
        raise ValueError("foreground_normalizer must be finite and positive")
    dice_denominator = n_positive if foreground_normalizer is None else foreground_normalizer
    for start in range(0, count, microbatch):
        image = batch["image"][start:start + microbatch].to(device, non_blocking=True)
        mask = batch["mask"][start:start + microbatch].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
            # Preserve the effective-batch Dice weighting when reducing microbatch size.
            bce, dice = segmentation_loss_components(model(image), mask, foreground_normalizer=dice_denominator / accumulation,
                                                      squared_pred=squared_pred)
            loss = 0.5 * bce + 0.5 * dice
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite segmentation loss")
        (loss / accumulation).backward()
        losses.append(loss.detach())
        if diagnostics is not None:
            components.append(torch.stack((bce.detach(), dice.detach())))
    if diagnostics is not None:
        for group in ("encoder", "decoder"):
            grads = [p.grad.detach().float().norm() for name, p in model.named_parameters()
                     if (name.startswith("encoder.") == (group == "encoder")) and p.grad is not None]
            diagnostics[f"{group}_grad_norm"] = float(torch.stack(grads).norm().cpu()) if grads else 0.0
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    if diagnostics is not None:
        diagnostics.update(bce=float(torch.stack(components).mean(0)[0].cpu()),
                           soft_dice_loss=float(torch.stack(components).mean(0)[1].cpu()),
                           grad_norm_before_clip=float(norm.cpu()))
        for group in ("encoder", "decoder"):
            changes = [(p.detach() - before[name]).float().norm() for name, p in model.named_parameters()
                       if name in before and (name.startswith("encoder.") == (group == "encoder"))]
            diagnostics[f"{group}_update_norm"] = float(torch.stack(changes).norm().cpu()) if changes else 0.0
        if not all(np.isfinite(value) for value in diagnostics.values()):
            raise FloatingPointError("non-finite gradient or parameter update diagnostic")
    return float(torch.stack(losses).mean().cpu())


def overfit_metrics(model, dataset, device="cpu", microbatch=2, output=None, *, amp=False, case_rows=None):
    model.eval()
    dice, false_positive, negative_voxels = [], 0, 0
    crops, cases = [], []
    with torch.no_grad():
        for index in range(len(dataset)):
            batch = dataset[(0, index)]
            record = dataset.records[index]
            case_crops, probabilities = [], []
            for start in range(0, 8, microbatch):
                image = batch["image"][start:start + microbatch].to(device)
                target = batch["mask"][start:start + microbatch].bool()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and str(device).startswith("cuda")):
                    probability = model(image).float().sigmoid().cpu()
                if not torch.isfinite(probability).all():
                    raise FloatingPointError("non-finite overfit prediction")
                pred = probability >= 0.5
                if output is not None:
                    probabilities.extend(probability[:, 0].numpy())
                for offset, (truth, detected) in enumerate(zip(target, pred)):
                    n_truth, n_pred = int(truth.sum()), int(detected.sum())
                    tp = int((truth & detected).sum())
                    crop_dice = 2 * tp / (n_truth + n_pred) if n_truth else None
                    if truth.any():
                        dice.append(crop_dice)
                    else:
                        false_positive += n_pred
                        negative_voxels += truth.numel()
                    case_crops.append({"mod": record.mod, "img_id": batch["img_id"], "crop_index": start + offset,
                                       "target_voxels": n_truth, "predicted_voxels": n_pred,
                                       "true_positive_voxels": tp, "false_positive_voxels": n_pred - tp,
                                       "false_negative_voxels": n_truth - tp, "dice": crop_dice,
                                       "precision": tp / n_pred if n_pred else None,
                                       "recall": tp / n_truth if n_truth else None,
                                       "negative_crop_fp_fraction": n_pred / truth.numel() if not n_truth else None,
                                       "peak_probability": float(probability[offset].max())})
            positive = [r for r in case_crops if r["target_voxels"]]
            negative = [r for r in case_crops if not r["target_voxels"]]
            cases.append({"mod": record.mod, "img_id": batch["img_id"], "positive_crops": len(positive),
                          "negative_crops": len(negative),
                          "positive_crop_dice": float(np.mean([r["dice"] for r in positive])) if positive else None,
                          "negative_crop_fp_fraction": float(np.mean([r["negative_crop_fp_fraction"] for r in negative])) if negative else None})
            crops.extend(case_crops)
            if output is not None:
                path = Path(output) / "images"
                path.mkdir(parents=True, exist_ok=True)
                selected = {"first": 0}
                if positive:
                    selected["worst_positive"] = min(positive, key=lambda r: r["dice"])["crop_index"]
                if negative:
                    selected["worst_negative"] = max(negative, key=lambda r: r["negative_crop_fp_fraction"])["crop_index"]
                for kind, crop in selected.items():
                    planes = preview_planes(batch["image"][crop, 0].numpy(), batch["mask"][crop, 0].numpy().astype(bool), probabilities[crop])
                    save_preview(path / f'{index:02d}_{record.mod}_{batch["img_id"]}_{kind}_crop{crop}.png', planes, 0.5)
    if output is not None:
        write_csv(Path(output) / "crops.csv", crops)
        write_csv(Path(output) / "cases.csv", cases)
    if case_rows is not None:
        case_rows.extend(cases)
    mean_dice = float(np.mean(dice)) if dice else 0.0
    fpr = false_positive / negative_voxels if negative_voxels else 1.0
    return {"positive_crop_dice": mean_dice, "negative_crop_fp_fraction": fpr,
            "passed": bool(dice and negative_voxels and mean_dice >= 0.90 and fpr < 0.01)}


def diagnose_overfit(checkpoint, data_root, *, device="cpu", output_dir=None, cache_dir=None, real_scan_root=None):
    """Re-evaluate fixed training crops without updating weights or the original gate."""
    model, payload = load_predictor(Path(checkpoint), device)
    config = payload["config"]
    if payload.get("task") != "segmentation" or config.get("stage") != "overfit":
        raise ValueError("diagnosis requires a segmentation overfit checkpoint")
    manifest = payload["manifest"]
    check_prerequisites(config, manifest, data_root, real_scan_root)
    output = Path(output_dir or Path(checkpoint).parent / "diagnostics")
    output.mkdir(parents=True, exist_ok=True)
    dataset = SegmentationDataset(manifest_records(manifest, "overfit"), data_root,
                                  patch_shape=config["patches"]["patch_shape"], preprocessing=config["preprocessing"],
                                  seed=manifest["seed"], augment=False, fixed=True,
                                  cache_dir=resolve_cache_dir(config, cache_dir, output),
                                  volume_cache_size=config["training"].get("volume_cache_size", 6), real_scan_root=real_scan_root)
    batch = config["training"]["batch_size"]
    report = {"evaluation_only": True, "checkpoint": str(checkpoint), "checkpoint_sha256": file_digest(checkpoint),
              "manifest_id": manifest["manifest_id"], "updates": payload["updates"], "threshold": 0.5,
              "fp32": overfit_metrics(model, dataset, device, batch, output / "fp32")}
    if str(device).startswith("cuda"):
        report["bfloat16"] = overfit_metrics(model, dataset, device, batch, output / "bfloat16", amp=True)
    write_json(output / "report.json", report)
    return report


def _rng_state():
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def validate_config(config):
    if config["stage"] not in {"overfit", "pilot", "full"} or config["model"]["name"] != SEGMENTER_NAME:
        raise ValueError("invalid segmentation stage/model")
    training = config["training"]
    if config["model"].get("decoder_padding_mode", "zeros") not in ("zeros", "replicate"):
        raise ValueError("decoder_padding_mode must be zeros or replicate")
    if not isinstance(training.get("squared_pred", False), bool):
        raise ValueError("training.squared_pred must be boolean")
    batch, accumulation = int(training["batch_size"]), int(training["accumulation_steps"])
    if batch not in (1, 2) or batch * accumulation != 8:
        raise ValueError("use batch_size=2, accumulation_steps=4 (or 1 and 8 after OOM)")
    shape = config["patches"]["patch_shape"]
    if len(shape) != 3 or min(shape) < 8 or any(p % 8 for p in shape):
        raise ValueError("crop dimensions must be multiples of eight")
    if training["epochs"] < 1 or training.get("validation_interval", 2) < 1:
        raise ValueError("epochs and validation interval must be positive")
    cap = {"overfit": 300, "pilot": 8, "full": 16}[config["stage"]]
    if training["epochs"] > cap:
        raise ValueError(f"{config['stage']} protocol permits at most {cap} epochs")
    if config["stage"] == "overfit" and not 1 <= training.get("max_updates", 300) <= 300:
        raise ValueError("overfit must use at most 300 optimizer updates")


def run_training(config, data_root, *, device="cpu", output_dir=None, cache_dir=None, resume=None, real_scan_root=None,
                 debug_updates=False):
    config = copy.deepcopy(config)
    validate_config(config)
    manifest = read_json(config["manifest"])
    check_prerequisites(config, manifest, data_root, real_scan_root)
    training, stage = config["training"], config["stage"]
    squared_pred = training.get("squared_pred", False)
    training_loss = {"bce_weight": 0.5, "dice_weight": 0.5, "squared_pred": squared_pred,
                     "empty_crops": "BCE_only"}
    output = Path(output_dir or config["output_dir"])
    if resume and Path(resume).resolve().parent != output.resolve():
        raise ValueError("resume must use the original run directory; set --output-dir to the checkpoint parent")
    if not resume and any((output / name).exists() for name in ("metrics.csv", "best.pt", "last.pt")):
        raise ValueError("run already exists; use --resume or a different --output-dir")
    output.mkdir(parents=True, exist_ok=True)
    cache = resolve_cache_dir(config, cache_dir, output)
    set_global_seed(training.get("seed", 21))
    restored = _load_torch_payload(Path(resume)) if resume else None
    if restored:
        if restored.get("task") != "segmentation" or restored["manifest"]["manifest_id"] != manifest["manifest_id"]:
            raise ValueError("resume task/manifest mismatch")
        old, new = copy.deepcopy(restored["config"]), copy.deepcopy(config)
        for cfg in (old, new):
            cfg.pop("output_dir", None)
            cfg.pop("cache_dir", None)
            cfg["training"].setdefault("squared_pred", False)
            cfg["model"].setdefault("decoder_padding_mode", "zeros")
            for key in ("epochs", "batch_size", "accumulation_steps", "num_workers"):
                cfg["training"].pop(key, None)
        if old != new:
            raise ValueError("resume would change model, preprocessing or training protocol")
    model = build_segmenter(config["model"], state_dict=restored["model_state_dict"] if restored else None).to(device)
    optimizer = make_optimizer(model, training)
    if restored:
        optimizer.load_state_dict(restored["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        _restore_rng(restored["rng_state"])
    cohort = "overfit" if stage == "overfit" else f"{stage}_train"
    records = manifest_records(manifest, cohort)
    dataset = SegmentationDataset(records, data_root, patch_shape=config["patches"]["patch_shape"],
                                  cache_dir=cache, preprocessing=config["preprocessing"], seed=manifest["seed"],
                                  augment=stage != "overfit", fixed=stage == "overfit",
                                  volume_cache_size=training.get("volume_cache_size", 2), real_scan_root=real_scan_root)
    loader, order = make_loader(dataset, training, device)
    best_ap = restored.get("best_ap", -1.0) if restored else -1.0
    stale = restored.get("stale_validations", 0) if restored else 0
    updates = restored.get("updates", 0) if restored else 0
    best_overfit = restored.get("best_overfit_metrics") if restored else None
    start_epoch = restored["epoch"] if restored else 0
    if restored and stage != "overfit" and best_ap >= 0 and not (output / "best.pt").exists():
        raise ValueError("resume needs best.pt alongside last.pt to preserve the previous model selection")
    if restored and stage == "overfit":
        if Path(resume).name != "last.pt":
            raise ValueError("overfit resume requires last.pt; best.pt may be from the middle of an epoch")
        if updates >= training.get("max_updates", 300):
            raise ValueError("overfit checkpoint has already reached max_updates")
        if best_overfit is not None and not (output / "best.pt").exists():
            raise ValueError("resume needs best.pt alongside last.pt to preserve the previous overfit selection")
    max_epochs = int(training["epochs"])
    if start_epoch >= max_epochs:
        raise ValueError("resume checkpoint has already reached training.epochs")
    if stage != "overfit" and stale >= training.get("early_stopping_validations", 3):
        raise ValueError("run already reached early stopping; inspect its localization report")
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    write_json(output / "manifest.json", manifest)
    write_json(output / "effective_config.json", config)

    def payload(epoch):
        return {"format_version": 1, "task": "segmentation", "config": config, "manifest": manifest,
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch, "updates": updates, "best_ap": best_ap, "stale_validations": stale,
                "best_overfit_metrics": best_overfit,
                "training_loss": training_loss,
                "rng_state": _rng_state(), "amp": "bfloat16" if str(device).startswith("cuda") else "disabled"}

    overfit_report = None
    for epoch in range(start_epoch, max_epochs):
        tick = perf_counter()
        order.epoch = epoch
        model.freeze_encoder(stage != "overfit" and epoch < training.get("freeze_encoder_epochs", 2))
        model.train()
        losses = []
        for batch in tqdm(loader, desc=f"seg epoch {epoch + 1}/{max_epochs}", unit="vol"):
            diagnostics = {} if debug_updates else None
            losses.append(train_volume(model, batch, optimizer, device=device, microbatch=training["batch_size"],
                                       accumulation=training["accumulation_steps"], diagnostics=diagnostics,
                                       squared_pred=squared_pred))
            updates += 1
            if diagnostics is not None:
                _write_metrics_row(output / "update_metrics.csv", {"updates": updates, "img_id": batch["img_id"],
                                                                   "loss": losses[-1], **diagnostics})
            if stage == "overfit" and (updates % 25 == 0 or updates >= training.get("max_updates", 300)):
                cases = [] if debug_updates else None
                overfit_report = overfit_metrics(model, dataset, device, training["batch_size"], case_rows=cases)
                _write_metrics_row(output / "overfit_metrics.csv", {"updates": updates, **overfit_report})
                for case in cases or []:
                    _write_metrics_row(output / "case_history.csv", {"updates": updates, **case})
                if best_overfit is None or overfit_rank(overfit_report) > overfit_rank(best_overfit):
                    best_overfit = {"updates": updates, **overfit_report}
                    save_checkpoint(payload(epoch + 1), output / "best.pt")
                    write_json(output / "best_gate.json", {"manifest_id": manifest["manifest_id"],
                        "training_loss": training_loss, "decoder_padding_mode": config["model"].get("decoder_padding_mode", "zeros"),
                        **best_overfit})
                model.train()
                if overfit_report["passed"] or updates >= training.get("max_updates", 300):
                    break
        seconds = perf_counter() - tick
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "updates": updates,
               "train_seconds": seconds, "max_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20 if str(device).startswith("cuda") else 0,
               "valid_dense_ap": None, "valid_micro_dice": None, "valid_peak_hits": None, "valid_seconds": None}
        if stage != "overfit" and ((epoch + 1) % training.get("validation_interval", 2) == 0 or epoch + 1 == max_epochs):
            valid = manifest_records(manifest, f"{stage}_valid")
            metrics = evaluate_records(model, {"config": config}, valid, data_root, cache_dir=cache,
                                       device=device, batch_size=training["batch_size"], spacing=manifest["spacing"], real_scan_root=real_scan_root)
            row.update(valid_dense_ap=metrics["dense_ap"], valid_micro_dice=metrics["micro_dice"],
                       valid_peak_hits=metrics["peak_hits"], valid_seconds=metrics["seconds"])
            if metrics["dense_ap"] > best_ap:
                best_ap, stale = metrics["dense_ap"], 0
                save_checkpoint(payload(epoch + 1), output / "best.pt")
            else:
                stale += 1
        save_checkpoint(payload(epoch + 1), output / "last.pt")
        _write_metrics_row(output / "metrics.csv", row)
        print(json.dumps(row), flush=True)
        if stage == "overfit" and overfit_report and (overfit_report["passed"] or updates >= training.get("max_updates", 300)):
            break
        if stage != "overfit" and stale >= training.get("early_stopping_validations", 3):
            break
    if stage == "overfit":
        overfit_report = overfit_metrics(model, dataset, device, training["batch_size"], output)
        if best_overfit is None or overfit_rank(overfit_report) > overfit_rank(best_overfit):
            best_overfit = {"updates": updates, **overfit_report}
            save_checkpoint(payload(epoch + 1), output / "best.pt")
            write_json(output / "best_gate.json", {"manifest_id": manifest["manifest_id"],
                "training_loss": training_loss, "decoder_padding_mode": config["model"].get("decoder_padding_mode", "zeros"),
                **best_overfit})
        save_checkpoint(payload(epoch + 1), output / "last.pt")
        overfit_report.update(manifest_id=manifest["manifest_id"], updates=updates, finite_gradients=True,
                              training_loss=training_loss, decoder_padding_mode=config["model"].get("decoder_padding_mode", "zeros"))
        write_json(output / "gate.json", overfit_report)
        return overfit_report
    # Free the training model/optimizer before restoring the selected checkpoint on the GPU.
    model.cpu()
    del optimizer, model, loader
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    report = run_evaluation(output / "best.pt", manifest,
                            manifest_records(manifest, "common_valid" if stage == "full" else "pilot_valid"),
                            data_root, output / "evaluation", device=device, batch_size=training["batch_size"],
                            cache_dir=cache, baseline=read_json(config["baseline_report"]), full=stage == "full", real_scan_root=real_scan_root)
    write_json(output / "calibration.json", report["calibration"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--diagnose-checkpoint")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--real-scan-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--cache-dir")
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--debug-updates", action="store_true", help="log per-update loss, gradients and parameter changes")
    args = parser.parse_args()
    if args.diagnose_checkpoint:
        if args.resume:
            parser.error("--diagnose-checkpoint does not train or resume")
        report = diagnose_overfit(args.diagnose_checkpoint, args.data_root, device=args.device,
                                  output_dir=args.output_dir, cache_dir=args.cache_dir, real_scan_root=args.real_scan_root)
        print(json.dumps(report, indent=2))
        return
    report = run_training(load_yaml_config(args.config), args.data_root, device=args.device,
                          output_dir=args.output_dir, cache_dir=args.cache_dir, resume=args.resume,
                          real_scan_root=args.real_scan_root, debug_updates=args.debug_updates)
    print(json.dumps(report, indent=2))
    if not report.get("passed", report.get("gate", {}).get("passed", False)):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
