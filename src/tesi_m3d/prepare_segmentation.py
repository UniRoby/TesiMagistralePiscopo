"""Prepare deterministic cohorts and audit annotations before segmentation training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit_local_artifacts import LocalArtifactAuditor
from .diagnose_localization import write_json
from .segmentation_data import audit_records, make_manifest, manifest_records, stratified, unique_records
from .train import load_yaml_config


class BalancedArtifactAuditor(LocalArtifactAuditor):
    def _pairs(self):
        pairs = super()._pairs()
        selected = stratified([fake for _, fake in pairs], {"inj": 16, "rem": 16}, self.seed)
        by_id = {fake.img_id: (real, fake) for real, fake in pairs}
        return [by_id[fake.img_id] for fake in selected]


def prepare(config, data_root, output_dir, *, stage="pilot", real_scan_root=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = make_manifest(config, data_root, real_scan_root)
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text())["manifest_id"] != manifest["manifest_id"]:
        raise ValueError("protocol directory already belongs to a different manifest; use a new output directory")
    write_json(path, manifest)
    cohorts = ["overfit", "pilot_train", "common_valid"]
    if stage == "full":
        cohorts.append("full_train")
    records = unique_records([r for name in cohorts for r in manifest_records(manifest, name)], data_root, real_scan_root)
    report = {"manifest_id": manifest["manifest_id"], "data_root": str(Path(data_root).resolve()),
              "real_scan_root": str(Path(real_scan_root or Path(data_root) / "real/scan").resolve()),
              "stage": stage, "passed": False, "cohorts": cohorts}
    try:
        report["records"] = audit_records(records, data_root, real_scan_root)
        paired_path = output / "paired_artifacts/summary.json"
        auditor = BalancedArtifactAuditor(data_root, config.get("metadata_dir", "metadata/m3dsynth"),
                                          "train", None, manifest["seed"], real_scan_root)
        paired = auditor.run(output / "paired_artifacts")
        paired["manifest_id"] = manifest["manifest_id"]
        write_json(paired_path, paired)
        report["paired_audit"] = str(paired_path)
        report["passed"] = True
        report["annotation_review"] = "Inspect three-plane overlays and bbox_fill_fraction; masks label edits, not tumor contours."
    except (OSError, ValueError, RuntimeError) as exc:
        report["error"] = str(exc)
        write_json(output / "audit.json", report)
        raise
    write_json(output / "audit.json", report)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--real-scan-root")
    parser.add_argument("--out-dir", default="outputs/segmentation_protocol")
    parser.add_argument("--stage", choices=("pilot", "full"), default="pilot")
    args = parser.parse_args()
    prepare(load_yaml_config(args.config), args.data_root, args.out_dir, stage=args.stage, real_scan_root=args.real_scan_root)


if __name__ == "__main__":
    main()
