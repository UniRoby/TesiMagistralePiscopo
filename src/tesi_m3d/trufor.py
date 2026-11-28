"""CLI for the Pix2Pix TruFor protocol. Run with --help."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from .trufor_data import (SliceDataset, atomic_json, digest, file_signature, inspect_scan,
                          load_manifest, prepare, slice_paths)
from .trufor_evaluation import (autocast, evaluate_volumes, export_prediction,
                                file_hash, metrics, select_threshold, summarize)
from .trufor_model import build_model, localization_loss
from .trufor_training import (SessionBudget, load_checkpoint, make_loader, optimizer_for,
                              seed_everything, train, validate_config)


def verify_inputs(manifest, destinations):
    for e in manifest["entries"]:
        scan = slice_paths(e["scan"])
        masks = slice_paths(e["mask"]) if e["mask"] else []
        signature = digest(dict(version=1, scan=file_signature(scan), mask=file_signature(masks),
                                percentiles=[1, 99]))
        if signature != e["signature"]:
            raise ValueError(f"Input changed since prepare: {e['id']}; rerun prepare")
        for source in (e["scan"], e["mask"]):
            if source and any(Path(p).resolve().is_relative_to(Path(source).resolve()) for p in destinations if p):
                raise ValueError(f"Output/cache must not be inside a scan or label directory: {source}")


def restored_model(checkpoint, upstream, device):
    saved = load_checkpoint(checkpoint)
    model = build_model(upstream, saved["contract"]["mode"] == "context")
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval(), saved["contract"]


def evaluate(args, manifest, device):
    model, contract = restored_model(args.checkpoint, args.upstream, device)
    if contract["manifest"] != manifest["fingerprint"]:
        raise ValueError("Checkpoint and evaluation manifest differ")
    output = Path(args.output)
    identity = dict(checkpoint_sha256=file_hash(args.checkpoint), manifest=manifest["fingerprint"])
    output.mkdir(parents=True, exist_ok=True)
    protocol = output / "evaluation_protocol.json"
    expected = dict(**identity, split=args.split)
    if protocol.exists() and json.loads(protocol.read_text()) != expected:
        raise ValueError("Evaluation directory belongs to a different checkpoint or split")
    atomic_json(protocol, expected)
    if args.split == "test":
        if not args.calibration:
            raise ValueError("Test requires --calibration from the full validation split")
        calibration = json.loads(Path(args.calibration).read_text())
        if (calibration.get("identity") != identity or calibration.get("split") != "valid"
                or not calibration.get("complete")):
            raise ValueError("Calibration does not match this checkpoint and validation manifest")
        threshold = calibration["threshold"]
    with SessionBudget(args.session_hours) as budget:
        entries = [e for e in manifest["entries"] if e["split"] == args.split]
        if not entries:
            raise ValueError(f"Empty split: {args.split}")
        rows = evaluate_volumes(model, entries, device, output / "volumes", identity, budget.expired)
    if rows is None:
        print("Evaluation paused. Repeat the same command to resume.")
        return
    if args.split == "valid":
        threshold = select_threshold(rows)
        atomic_json(output / "calibration.json", dict(identity=identity, split="valid", complete=True,
                    threshold=threshold, objective="macro F1 on manipulated whole volumes",
                    volumes=len(rows)))
    report = dict(identity=identity, split=args.split, calibrated=summarize(rows, threshold),
                  fixed_05=summarize(rows, 0.5))
    atomic_json(output / "report.json", report)
    print(json.dumps(report["calibrated"]["groups"], indent=2))


def profile(args, manifest, config, device):
    config = validate_config(config)
    entries = [e for e in manifest["entries"] if e["split"] == "train"]
    fake = next(i for i, e in enumerate(entries) if not e["real"])
    real = next(i for i, e in enumerate(entries) if e["real"])
    positive = entries[fake]["positives"]
    depth = entries[real]["shape"][0]
    tokens = [(fake, positive[0], 0, 0, 0), (real, depth // 2, 0, 0, 0),
              (fake, positive[len(positive) // 2], 0, 0, 0), (real, depth // 3, 0, 0, 0)]
    results = {}
    for mode in ("2d", "context"):
        seed_everything(config["seed"])
        model = build_model(args.upstream, mode == "context", args.weights).to(device)
        dataset = SliceDataset(entries, mode == "context")
        loader = make_loader(dataset, tokens, config, device)
        started = time.monotonic()
        samples = [[x.to(device, non_blocking=True) for x in batch] for batch in loader]
        if device.type == "cuda":
            torch.cuda.synchronize()
        loading_seconds = time.monotonic() - started
        del loader
        optimizer = optimizer_for(model, config)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not torch.cuda.is_bf16_supported())
        def measure():
            model.eval()
            losses, tp, fp, fn, negative_fp, negative_pixels = [], 0, 0, 0, 0, 0
            with torch.no_grad():
                for images, target, valid, distances in samples:
                    with autocast(device):
                        logits = model(images, valid, distances)
                        loss = localization_loss(logits, target)
                    p = logits.float().softmax(1)[:, 1] >= 0.5
                    y = target.bool()
                    if y.any():
                        tp += int((p & y).sum())
                        fp += int((p & ~y).sum())
                        fn += int((~p & y).sum())
                    else:
                        negative_fp += int(p.sum())
                        negative_pixels += p.numel()
                    losses.append(float(loss))
            return dict(loss=float(np.mean(losses)), positive_f1=float(metrics([tp, fp, fn])[0]),
                        negative_fraction=negative_fp / max(1, negative_pixels))
        before = measure()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.monotonic()
        steps = args.overfit_steps or args.steps
        model.train()
        for i in range(steps):
            images, target, valid, distances = samples[i % len(samples)]
            optimizer.zero_grad(set_to_none=True)
            with autocast(device):
                loss = localization_loss(model(images, valid, distances), target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite profiling loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            if (i + 1) % 10 == 0:
                print(f"Profile {mode}: {i + 1}/{steps}, loss={float(loss.detach()):.5f}", flush=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.monotonic() - started
        peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        after = measure()
        passed = after["loss"] < before["loss"] * 0.5 and after["positive_f1"] >= 0.8 and after["negative_fraction"] <= 0.01
        results[mode] = dict(before=before, after=after, steps=steps, seconds=seconds,
                              seconds_per_training_slice=seconds / steps, peak_allocated_bytes=peak,
                              initial_loading_seconds=loading_seconds,
                              overfit_passed=passed if args.overfit_steps else None)
        print(json.dumps({mode: results[mode]}, indent=2), flush=True)
        atomic_json(args.output, dict(device=str(device), config=config, results=results))
        del model, optimizer, samples, scaler, loss
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if args.overfit_steps and not all(r["overfit_passed"] for r in results.values()):
        raise RuntimeError(f"Overfit check failed; inspect {args.output} before starting full training")


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Validate all Pix2Pix/real series and cache percentiles")
    p.add_argument("--metadata-dir", default="metadata/m3dsynth")
    p.add_argument("--data-root", required=True)
    p.add_argument("--real-scan-root")
    p.add_argument("--cache-dir", default="outputs/trufor/cache")
    p.add_argument("--output", default="outputs/trufor/manifest.json")
    for name in ("profile", "train", "evaluate", "predict"):
        p = commands.add_parser(name)
        p.add_argument("--upstream", default="external/TruFor")
        p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        p.add_argument("--output", required=True)
        if name != "predict":
            p.add_argument("--manifest", default="outputs/trufor/manifest.json")
        if name in ("train", "profile"):
            p.add_argument("--config", default="configs/trufor_pix2pix.yaml")
            p.add_argument("--weights", default="checkpoints/TruFor/weights/trufor.pth.tar")
        if name in ("evaluate", "predict"):
            p.add_argument("--checkpoint", required=True)
            p.add_argument("--calibration")
        if name in ("train", "evaluate"):
            p.add_argument("--session-hours", type=float, default=4)
        if name == "train":
            p.add_argument("--mode", choices=("2d", "context"), default="2d")
            p.add_argument("--stage", choices=("adapt", "continue"), default="adapt")
            p.add_argument("--init-from")
            p.add_argument("--resume", action="store_true")
        elif name == "profile":
            p.add_argument("--steps", type=int, default=8)
            p.add_argument("--overfit-steps", type=int, default=0)
        elif name == "evaluate":
            p.add_argument("--split", choices=("valid", "test"), required=True)
        elif name == "predict":
            p.add_argument("--scan-dir", required=True)
            p.add_argument("--spacing-zyx", nargs=3, type=float, required=True)
            p.add_argument("--cache-dir", default="outputs/trufor/cache")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "prepare":
        prepare(args.metadata_dir, args.data_root, args.real_scan_root, args.cache_dir, args.output)
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; verify the laboratory PyTorch installation or use --device cpu")
    if args.command == "predict":
        if not args.calibration:
            raise ValueError("predict requires --calibration")
        if not all(np.isfinite(v) and v > 0 for v in args.spacing_zyx):
            raise ValueError("Spacing must contain three positive finite values")
        for path in (args.output, args.cache_dir):
            if Path(path).resolve().is_relative_to(Path(args.scan_dir).resolve()):
                raise ValueError("Output/cache cannot be inside the input scan")
        calibration = json.loads(Path(args.calibration).read_text())
        checkpoint_hash = file_hash(args.checkpoint)
        if (not calibration.get("complete") or calibration.get("split") != "valid"
                or calibration["identity"]["checkpoint_sha256"] != checkpoint_hash):
            raise ValueError("Calibration does not match checkpoint")
        info = inspect_scan(args.scan_dir, args.cache_dir)
        entry = dict(scan=str(Path(args.scan_dir).resolve()), spacing=args.spacing_zyx, **info)
        model, contract = restored_model(args.checkpoint, args.upstream, device)
        if calibration["identity"]["manifest"] != contract["manifest"]:
            raise ValueError("Calibration used a different dataset manifest")
        if not np.isfinite(calibration["threshold"]) or not 0 < calibration["threshold"] < 1:
            raise ValueError("Invalid calibrated threshold")
        export_prediction(model, entry, device, args.output, calibration["threshold"], checkpoint_hash)
        return
    manifest = load_manifest(args.manifest)
    verify_inputs(manifest, [args.output])
    if args.command == "evaluate":
        evaluate(args, manifest, device)
        return
    config = yaml.safe_load(Path(args.config).read_text())
    if args.command == "profile":
        if args.steps < 1 or args.overfit_steps < 0:
            raise ValueError("Invalid profiling step count")
        profile(args, manifest, config, device)
    else:
        with SessionBudget(args.session_hours) as budget:
            train(manifest, config, args.upstream, args.weights, args.output, args.mode,
                  args.stage, args.init_from, args.resume, device, budget)


if __name__ == "__main__":
    main()
