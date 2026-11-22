"""Memorize the two prespecified difficult crops, alone and with a clean negative."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import torch

from .compare_segmentation_batches import tensor_digest
from .debug_segmentation import run_fixed_stage
from .diagnose_localization import write_json
from .segmentation_data import SegmentationDataset, manifest_records
from .train import load_yaml_config, resolve_cache_dir
from .train_segmentation import check_prerequisites, read_json, validate_config


HARD_CROPS = (("inj_2092", 3), ("rem_133", 2))


def prepare_cases(config, manifest, reference, data_root, *, cache_dir, real_scan_root):
    """Regenerate crops at their original cohort indices and reject any changed input."""
    comparison = read_json(reference / "comparison.json")
    if (comparison.get("error") or comparison.get("active_arm") is not None or
            comparison.get("experiment") != "batch_composition_comparison" or
            set(comparison.get("arms", {})) != {"grouped", "mixed"}):
        raise ValueError("reference must be a completed grouped/mixed comparison")
    if manifest != read_json(reference / "manifest.json") or comparison["manifest_id"] != manifest["manifest_id"]:
        raise ValueError("reference manifest differs from the current overfit cohort")
    if config != read_json(reference / "effective_config.json"):
        raise ValueError("use exactly the reference model, preprocessing and training config")
    grouped, mixed = (comparison["arms"][name] for name in ("grouped", "mixed"))
    if grouped["initial_state_sha256"] != mixed["initial_state_sha256"]:
        raise ValueError("reference initial model states differ")
    with (reference / "crop_pool.csv").open(newline="") as handle:
        ledger = list(csv.DictReader(handle))
    for row in ledger:
        for key in ("pool_index", "scan_index", "crop_index", "target_voxels"):
            row[key] = int(row[key])
    digest = hashlib.sha256(json.dumps(ledger, sort_keys=True).encode()).hexdigest()
    if digest != comparison["crop_pool_sha256"] or any(arm["crop_pool_sha256"] != digest for arm in (grouped, mixed)):
        raise ValueError("reference crop ledger hash mismatch")
    records = manifest_records(manifest, "overfit")
    if len(records) != 6 or len(ledger) != 48:
        raise ValueError("reference must contain the six scans and 48 fixed crops")
    dataset = SegmentationDataset(records, data_root, patch_shape=config["patches"]["patch_shape"],
                                  preprocessing=config["preprocessing"], seed=manifest["seed"],
                                  augment=False, fixed=True, volume_cache_size=1, cache_dir=cache_dir,
                                  real_scan_root=real_scan_root)
    cases = []
    for img_id, crop_index in HARD_CROPS:
        indices = [i for i, r in enumerate(records) if r.mod == "pix2pix" and r.img_id == img_id]
        if len(indices) != 1:
            raise ValueError(f"expected one pix2pix/{img_id} in the original cohort")
        index = indices[0]
        source = dataset[(0, index)]
        pair = {key: source[key][[crop_index, 4]].clone() for key in ("image", "mask")}
        if (pair["image"].shape != pair["mask"].shape or not torch.isfinite(pair["image"]).all() or
                not ((pair["mask"] == 0) | (pair["mask"] == 1)).all() or
                not pair["mask"][0].any() or pair["mask"][1].any()):
            raise ValueError("expected a finite positive crop and a clean fixed negative crop 4")
        selected = []
        for offset, crop in enumerate((crop_index, 4)):
            actual = {"pool_index": index * 8 + crop, "scan_index": index, "crop_index": crop,
                      "mod": "pix2pix", "img_id": img_id, "target_voxels": int(pair["mask"][offset].sum()),
                      "image_sha256": tensor_digest(pair["image"][offset]),
                      "mask_sha256": tensor_digest(pair["mask"][offset])}
            if actual != ledger[index * 8 + crop]:
                raise ValueError(f"input/target hash or identity changed for {img_id} crop {crop}")
            target = pair["mask"][offset, 0].bool()
            points = target.nonzero()
            actual["bbox_zyx_inclusive"] = [points.min(0).values.tolist(), points.max(0).values.tolist()] if len(points) else None
            actual["target_touches_faces"] = [f"{axis}_{face}" for dim, axis in enumerate("zyx")
                                               for face, position in (("low", 0), ("high", -1))
                                               if target.select(dim, position).any()]
            selected.append(actual)
        cases.append((img_id, pair, selected))
    return cases, comparison


def run_hard_crops(config, data_root, *, device="cpu", output_dir="outputs/segmentation_hard_crops",
                   reference_dir="outputs/segmentation_batch_comparison", cache_dir=None, real_scan_root=None):
    config = copy.deepcopy(config)
    validate_config(config)
    if config["stage"] != "overfit":
        raise ValueError("hard-crop diagnosis requires the original overfit config")
    manifest = read_json(config["manifest"])
    check_prerequisites(config, manifest, data_root, real_scan_root)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("hard-crop output already contains files; choose a new --output-dir")
    cases, comparison = prepare_cases(config, manifest, Path(reference_dir), data_root,
                                      cache_dir=resolve_cache_dir(config, cache_dir, output), real_scan_root=real_scan_root)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "effective_config.json", config)
    write_json(output / "manifest.json", manifest)
    write_json(output / "selection.json", {"reference_dir": str(reference_dir),
               "reference_crop_pool_sha256": comparison["crop_pool_sha256"], "threshold": 0.5,
               "selection": "prespecified inj_2092/3 and rem_133/2; first clean negative crop 4 of each scan",
               "crops": [row for _, _, selected in cases for row in selected]})
    expected_hash = comparison["arms"]["mixed"]["initial_state_sha256"]
    report = {"task": "segmentation_hard_crops", "manifest_id": manifest["manifest_id"],
              "initial_state_sha256": expected_hash, "reference_crop_pool_sha256": comparison["crop_pool_sha256"],
              "max_updates_per_stage": config["training"].get("max_updates", 300),
              "runtime": {"torch": torch.__version__, "monai": version("monai"), "device": str(device)},
              "passed": False, "active_stage": None, "stages": {}}
    try:
        for img_id, pair, _ in cases:
            for name, count in (("positive", 1), ("positive_negative", 2)):
                key = f"{img_id}/{name}"
                report["active_stage"] = key
                print(f"starting hard-crop diagnosis: {key}", flush=True)
                batch = {kind: tensor[:count] for kind, tensor in pair.items()}
                result, checked = run_fixed_stage(config, manifest, batch, output / key, device=device,
                                                  complete_budget=True, expected_initial_state_sha256=expected_hash)
                report["stages"][key] = {"last": result, "best": read_json(output / key / "best_metrics.json"),
                                          "update_check": checked}
                if not checked["passed"]:
                    raise RuntimeError(f"parameter/BatchNorm preflight failed at {key}")
                write_json(output / "report.json", report)
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
        report.update(active_stage=None, passed=all(s["last"]["passed"] for s in report["stages"].values()))
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
    parser.add_argument("--output-dir", default="outputs/segmentation_hard_crops")
    parser.add_argument("--reference-dir", default="outputs/segmentation_batch_comparison")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    report = run_hard_crops(load_yaml_config(args.config), args.data_root, device=args.device,
                            output_dir=args.output_dir, reference_dir=args.reference_dir,
                            cache_dir=args.cache_dir, real_scan_root=args.real_scan_root)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
