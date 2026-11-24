"""Paired grouped/mixed overfit on exactly the same 48 fixed CT crops."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .diagnose_localization import write_csv, write_json
from .segmentation import build_segmenter
from .segmentation_data import SegmentationDataset, VolumeOrder, category, manifest_records
from .train import _load_torch_payload, _write_metrics_row, load_yaml_config, resolve_cache_dir, set_global_seed
from .train_segmentation import (check_prerequisites, make_optimizer, overfit_metrics, overfit_rank,
                                 read_json, save_checkpoint, train_volume, validate_config, _rng_state)


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def named_tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, tensor in tensors:
        digest.update(name.encode())
        digest.update(tensor_digest(tensor).encode())
    return digest.hexdigest()


class FixedCropPool:
    """Keep only the 48 crops in RAM; evaluation never rereads a CT or mask."""
    def __init__(self, dataset):
        self.records = dataset.records
        if sorted(category(r) for r in self.records) != ["inj", "inj", "real", "real", "rem", "rem"]:
            raise ValueError("comparison requires two injection, two removal and two real scans")
        batches = []
        for i, record in enumerate(self.records):
            print(f"preparing fixed crops: {i + 1}/6 {record.mod}/{record.img_id}", flush=True)
            batches.append(dataset[(0, i)])
        self.image = torch.cat([b["image"] for b in batches])
        self.mask = torch.cat([b["mask"] for b in batches])
        if (self.image.ndim != 5 or self.image.shape != self.mask.shape or self.image.shape[0:2] != (48, 1)
                or not torch.isfinite(self.image).all() or not ((self.mask == 0) | (self.mask == 1)).all()):
            raise ValueError("comparison needs 48 finite image crops and binary matching targets")
        self.positive = self.mask.flatten(1).any(1).numpy()
        self.real = np.repeat([r.is_real for r in self.records], 8)
        for i, record in enumerate(self.records):
            expected = np.zeros(8, dtype=bool) if record.is_real else np.arange(8) < 4
            if not np.array_equal(self.positive[i * 8:(i + 1) * 8], expected):
                raise ValueError("expected four positive/four negative crops per fake and eight negatives per real")
        self.rows = [{"pool_index": i, "scan_index": i // 8, "crop_index": i % 8,
                      "mod": self.records[i // 8].mod, "img_id": self.records[i // 8].img_id,
                      "target_voxels": int(self.mask[i].sum()), "image_sha256": tensor_digest(self.image[i]),
                      "mask_sha256": tensor_digest(self.mask[i])} for i in range(48)]
        self.digest = hashlib.sha256(json.dumps(self.rows, sort_keys=True).encode()).hexdigest()

    def __len__(self):
        return 6

    def __getitem__(self, key):
        index = key[1] if isinstance(key, tuple) else key
        return {"image": self.image[index * 8:(index + 1) * 8],
                "mask": self.mask[index * 8:(index + 1) * 8], "img_id": self.records[index].img_id}


def epoch_batches(pool, seed, epoch, *, mixed):
    if not mixed:
        order = VolumeOrder(6, seed)
        order.epoch = epoch
        return np.asarray([np.arange(index * 8, (index + 1) * 8) for _, index in order])
    rng = np.random.default_rng([seed, epoch])
    positives = rng.permutation(np.flatnonzero(pool.positive))
    real_scans = np.flatnonzero([r.is_real for r in pool.records])
    real_negatives = [rng.permutation(np.arange(i * 8, (i + 1) * 8)) for i in real_scans]
    remainder = rng.permutation(np.concatenate((np.flatnonzero(~pool.positive & ~pool.real),
                                                real_negatives[0][6:], real_negatives[1][6:])))
    counts = rng.permutation([3, 3, 2, 3, 3, 2])
    batches, pos, neg = [], 0, 0
    for i, count in enumerate(counts):
        extra = 6 - count
        batch = np.concatenate((positives[pos:pos + count], [real_negatives[0][i], real_negatives[1][i]],
                                remainder[neg:neg + extra]))
        batches.append(rng.permutation(batch))
        pos, neg = pos + count, neg + extra
    result = np.asarray(batches)
    if result.shape != (6, 8) or not np.array_equal(np.sort(result.flatten()), np.arange(48)):
        raise ValueError("invalid mixed crop schedule")
    return result


def run_arm(config, manifest, pool, output, *, mixed, device, expected_initial_state_sha256=None,
            experiment="batch_composition_comparison", deterministic=False):
    started = perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    padding = config["model"].get("decoder_padding_mode", "zeros")
    squared_pred = training.get("squared_pred", False)
    training_loss = {"bce_weight": 0.5, "dice_weight": 0.5, "squared_pred": squared_pred,
                     "empty_crops": "BCE_only"}
    set_global_seed(training.get("seed", 21))
    if deterministic and (not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled() or torch.backends.cudnn.benchmark):
        raise ValueError("deterministic arm requires strict algorithms and disabled cuDNN autotuning")
    model = build_segmenter(config["model"]).to(device)
    model.deterministic_trilinear_backward = deterministic
    initial_hash = named_tensor_digest(model.state_dict().items())
    if expected_initial_state_sha256 is not None and initial_hash != expected_initial_state_sha256:
        raise ValueError("comparison arms did not start from identical model states: "
                         f"expected={expected_initial_state_sha256}, actual={initial_hash}; "
                         "check the original weights, PyTorch/MONAI versions and model code")
    optimizer = make_optimizer(model, training)
    layout = "mixed" if mixed else "grouped"
    print(f"starting {layout}: {training['epochs']} epochs, {training['max_updates']} updates", flush=True)
    sampling = {"layout": layout, "fixed_foreground_normalizer": 4, "crop_pool_sha256": pool.digest}
    write_json(output / "effective_config.json", config)
    write_json(output / "manifest.json", manifest)
    write_json(output / "sampling.json", sampling)
    exposures = np.zeros(48, dtype=int)
    updates, best, first_passed = 0, None, None

    def payload(epoch):
        return {"format_version": 1, "task": "segmentation", "experiment": experiment,
                "config": config, "manifest": manifest, "sampling": sampling, "training_loss": training_loss,
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch, "updates": updates, "best_overfit_metrics": best,
                "best_ap": -1.0, "stale_validations": 0, "rng_state": _rng_state(),
                "execution_policy": {"strict_determinism": deterministic,
                                     "trilinear_backward": "cpu_first_order" if deterministic else "native"},
                "amp": "bfloat16" if str(device).startswith("cuda") else "disabled"}

    def select_best(metrics, epoch):
        nonlocal best, first_passed
        if metrics["passed"] and first_passed is None:
            first_passed = updates
        if best is None or overfit_rank(metrics) > overfit_rank(best):
            best = {"updates": updates, **metrics}
            save_checkpoint(payload(epoch), output / "best.pt")
            write_json(output / "best_gate.json", {"manifest_id": manifest["manifest_id"],
                                                   "training_loss": training_loss, "decoder_padding_mode": padding, **best})

    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    # Complete both budgets after an early success; write last.pt only at the end.
    for epoch in range(training["epochs"]):
        for indices in epoch_batches(pool, training.get("seed", 21), epoch, mixed=mixed):
            model.train()
            batch = {"image": pool.image[indices], "mask": pool.mask[indices]}
            diagnostics = {}
            # Keep each positive crop's original Dice weight when changing batch composition.
            loss = train_volume(model, batch, optimizer, device=device, microbatch=training["batch_size"],
                                accumulation=training["accumulation_steps"], diagnostics=diagnostics,
                                foreground_normalizer=4, squared_pred=squared_pred)
            updates += 1
            exposures[indices] += 1
            if deterministic:
                diagnostics.update(model_state_sha256=named_tensor_digest(model.state_dict().items()),
                    gradient_sha256=named_tensor_digest((name, p.grad) for name, p in model.named_parameters() if p.grad is not None))
                if updates == 1:
                    print("Strict first update complete; trilinear backward=CPU; exact state/gradient tracing active", flush=True)
            _write_metrics_row(output / "update_metrics.csv", {"updates": updates, "epoch": epoch + 1,
                               "crop_indices": ";".join(map(str, indices)), "positive_crops": int(pool.positive[indices].sum()),
                               "real_crops": int(pool.real[indices].sum()), "scans": len(set(indices // 8)),
                               "loss": loss, "squared_pred": squared_pred, **diagnostics})
            if updates % 25 == 0 or updates == training["max_updates"]:
                cases = []
                metrics = overfit_metrics(model, pool, device, training["batch_size"], case_rows=cases)
                _write_metrics_row(output / "overfit_metrics.csv", {"updates": updates, **metrics})
                for row in cases:
                    _write_metrics_row(output / "case_history.csv", {"updates": updates, **row})
                select_best(metrics, epoch + 1)
                print(json.dumps({"layout": layout, "updates": updates, **metrics}), flush=True)
    final = overfit_metrics(model, pool, device, training["batch_size"], output)
    select_best(final, training["epochs"])
    final.update(manifest_id=manifest["manifest_id"], updates=updates, finite_gradients=True,
                 training_loss=training_loss, decoder_padding_mode=padding)
    write_json(output / "gate.json", final)
    save_checkpoint(payload(training["epochs"]), output / "last.pt")
    write_csv(output / "exposures.csv", [{"pool_index": i, "exposures": int(n)} for i, n in enumerate(exposures)])
    if updates != training["max_updates"] or not np.all(exposures == training["epochs"]):
        raise ValueError("comparison budget or crop exposures differ from protocol")
    # Export the best with the same evaluator, after preserving the final weights.
    model.load_state_dict(_load_torch_payload(output / "best.pt")["model_state_dict"], strict=True)
    measured_best = overfit_metrics(model, pool, device, training["batch_size"], output / "best")
    reproduced = (measured_best["passed"] == best["passed"] and all(
        abs(measured_best[key] - best[key]) <= 1e-4
        for key in ("positive_crop_dice", "negative_crop_fp_fraction")))
    if not reproduced:
        raise ValueError("best checkpoint metrics were not reproduced within 1e-4")
    report = {**sampling, "training_loss": training_loss, "decoder_padding_mode": padding,
              "deterministic": deterministic,
              "initial_state_sha256": initial_hash, "updates": updates,
              "epochs": training["epochs"], "exposures_per_crop": int(exposures[0]),
              "first_gate_passed_update": first_passed, "best": best, "last": final,
              "best_reproduced_within_1e_4": reproduced,
              "seconds": perf_counter() - started,
              "max_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20 if str(device).startswith("cuda") else None}
    write_json(output / "report.json", report)
    return report


def run_comparison(config, data_root, *, device="cpu", output_dir="outputs/segmentation_batch_comparison",
                   cache_dir=None, real_scan_root=None):
    config = copy.deepcopy(config)
    validate_config(config)
    training = config["training"]
    if (config["stage"] != "overfit" or training["epochs"] > 50 or
            training["max_updates"] != 6 * training["epochs"]):
        raise ValueError("comparison needs overfit with max_updates = 6 * epochs, at most 50 epochs")
    manifest = read_json(config["manifest"])
    check_prerequisites(config, manifest, data_root, real_scan_root)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("comparison output already contains files; choose a new --output-dir")
    dataset = SegmentationDataset(manifest_records(manifest, "overfit"), data_root,
                                  patch_shape=config["patches"]["patch_shape"], preprocessing=config["preprocessing"],
                                  seed=manifest["seed"], augment=False, fixed=True, volume_cache_size=1,
                                  cache_dir=resolve_cache_dir(config, cache_dir, output), real_scan_root=real_scan_root)
    pool = FixedCropPool(dataset)
    del dataset
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "crop_pool.csv", pool.rows)
    write_json(output / "effective_config.json", config)
    write_json(output / "manifest.json", manifest)
    report = {"experiment": "batch_composition_comparison", "manifest_id": manifest["manifest_id"],
              "crop_pool_sha256": pool.digest, "complete": False, "passed": False,
              "active_arm": "grouped", "arms": {}}
    try:
        for layout in ("grouped", "mixed"):
            report["active_arm"] = layout
            report["arms"][layout] = run_arm(config, manifest, pool, output / layout,
                mixed=layout == "mixed", device=device,
                expected_initial_state_sha256=report["arms"].get("grouped", {}).get("initial_state_sha256"))
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        grouped, mixed = report["arms"]["grouped"], report["arms"]["mixed"]
        if grouped["initial_state_sha256"] != mixed["initial_state_sha256"]:
            raise ValueError("comparison arms did not start from identical model states")
        report.update(complete=True, active_arm=None, training_loss=grouped["training_loss"],
                      passed=mixed["last"]["passed"],
                      delta_last_dice=mixed["last"]["positive_crop_dice"] - grouped["last"]["positive_crop_dice"],
                      delta_last_fp_fraction=mixed["last"]["negative_crop_fp_fraction"] - grouped["last"]["negative_crop_fp_fraction"])
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
    parser.add_argument("--output-dir", default="outputs/segmentation_batch_comparison")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    report = run_comparison(load_yaml_config(args.config), args.data_root, device=args.device,
                            output_dir=args.output_dir, cache_dir=args.cache_dir, real_scan_root=args.real_scan_root)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
