"""Evaluate segmenters or the existing classifier with identical voxel metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import assert_no_orig_id_leakage, filter_records, read_records
from .diagnose_localization import write_json
from .segmentation import file_digest, load_predictor, protocol_signature, read_calibration
from .segmentation_data import manifest_records, unique_records
from .segmentation_metrics import compare_baseline, evaluate_records
from .train import _load_torch_payload, resolve_cache_dir


def run_evaluation(checkpoint, manifest, records, data_root, output_dir, *, device="cpu", batch_size=2,
                   cache_dir=None, calibration=None, split="valid", baseline=None, full=False, real_scan_root=None):
    if split != "valid" and calibration is None:
        raise ValueError("test evaluation requires frozen validation thresholds")
    digest = file_digest(checkpoint)
    if calibration is not None and calibration.get("checkpoint_sha256") != digest:
        raise ValueError("frozen calibration belongs to a different checkpoint")
    model, payload = load_predictor(Path(checkpoint), device)
    trained_manifest = payload.get("manifest")
    if trained_manifest and trained_manifest["manifest_id"] != manifest["manifest_id"]:
        raise ValueError("checkpoint and evaluation manifest differ")
    report = evaluate_records(model, payload, records, data_root, cache_dir=cache_dir, device=device,
                              batch_size=batch_size, calibration=calibration, output_dir=output_dir,
                              spacing=manifest.get("spacing"), real_scan_root=real_scan_root)
    report.update(split=split, manifest_id=manifest["manifest_id"], checkpoint=str(checkpoint),
                  checkpoint_sha256=digest, preprocessing=payload["config"].get("preprocessing", {"clip": False}),
                  model_name=payload["config"]["model"]["name"], protocol_signature=protocol_signature(payload["config"]))
    if calibration is None:
        report["calibration"].update(checkpoint_sha256=report["checkpoint_sha256"],
                                      manifest_id=manifest["manifest_id"], cohort_id=report["cohort_id"])
    if baseline is not None:
        if split != "valid":
            raise ValueError("success gates are validation-only; do not tune on test")
        report["gate"] = compare_baseline(report, baseline, full=full)
    output = Path(output_dir)
    write_json(output / "metrics.json", report)
    write_json(output / "calibration.json", report["calibration"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="outputs/segmentation_protocol/manifest.json")
    parser.add_argument("--cohort", choices=("pilot_valid", "full_valid", "common_valid"), default="common_valid")
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--mods", nargs="+", choices=("cycle", "diffusion"))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--real-scan-root")
    parser.add_argument("--out-dir", "--output-dir", dest="out_dir", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--calibration")
    parser.add_argument("--baseline-report")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    calibration = read_calibration(args.calibration, args.checkpoint) if args.calibration else None
    if args.split == "test":
        if not args.mods or len(args.mods) != 1 or calibration is None:
            parser.error("test requires one held-out --mods and --calibration; cycle/diffusion must be separate")
        payload = _load_torch_payload(Path(args.checkpoint))
        config = payload["config"]
        del payload
        if set(args.mods) & set(config["split"]["train_mods"]):
            parser.error("requested test generator was used for training")
        records = unique_records(filter_records(read_records(config.get("metadata_dir", "metadata/m3dsynth")),
                                                args.mods, ("test",), include_real=True), args.data_root, args.real_scan_root)
        assert_no_orig_id_leakage({"train": manifest_records(manifest, "full_train"),
                                  "valid": manifest_records(manifest, "common_valid"), "test": records})
    else:
        records = manifest_records(manifest, args.cohort)
    output = Path(args.out_dir)
    cache = resolve_cache_dir({}, args.cache_dir, output)
    baseline = json.loads(Path(args.baseline_report).read_text()) if args.baseline_report else None
    report = run_evaluation(args.checkpoint, manifest, records, args.data_root, output, device=args.device,
                            batch_size=args.batch_size, cache_dir=cache, calibration=calibration,
                            split=args.split, baseline=baseline, full=args.cohort == "common_valid", real_scan_root=args.real_scan_root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
