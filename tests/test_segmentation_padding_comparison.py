"""Only decoder padding changes; preserve paired budgets, provenance and old gates."""
import copy
import csv
from dataclasses import asdict, replace
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from _tiff_fixtures import make_fake_corpus
from test_segmentation import TinySegmenter
from tesi_m3d import compare_segmentation_batches as batches
from tesi_m3d import compare_segmentation_padding as comparison
from tesi_m3d import diagnose_segmentation_batch_probabilities as audit
from tesi_m3d.debug_segmentation import crop_regions
from tesi_m3d.diagnose_segmentation_probabilities import probability_statistics
from tesi_m3d.segmentation import file_digest
from tesi_m3d.train import load_yaml_config
from tesi_m3d.train_segmentation import check_prerequisites


class TinyPaddingSegmenter(TinySegmenter):
    def __init__(self, padding):
        super().__init__()
        self.output = torch.nn.Conv3d(2, 1, 3, padding=1, padding_mode=padding)


@pytest.fixture
def reference(tmp_path, monkeypatch):
    root, records = make_fake_corpus(tmp_path / "data", n_records=4, n_real=2, real_shared_dirs=2,
        scan_z=16, shape=(16, 16), cube_origin=(6, 6, 6), cube_size=(2, 2, 2))
    records = [replace(r, ty="rem") if i in (2, 3) else r for i, r in enumerate(records)]
    manifest = {"manifest_id": "fixture", "seed": 21, "cohorts": {"overfit": [asdict(r) for r in records]}}
    protocol = tmp_path / "protocol"; protocol.mkdir()
    (protocol / "manifest.json").write_text(json.dumps(manifest))
    (protocol / "audit.json").write_text(json.dumps({"manifest_id": "fixture", "passed": True,
        "cohorts": ["overfit"], "data_root": str(root.resolve())}))
    config = load_yaml_config("configs/segmentation_overfit_squared.yaml")
    config.update(manifest=str(protocol / "manifest.json"), cache_dir=str(tmp_path / "cache"))
    config["patches"]["patch_shape"] = [8, 8, 8]
    config["training"].update(epochs=1, max_updates=6)

    def build(model_config, *, state_dict=None):
        model = TinyPaddingSegmenter(model_config.get("decoder_padding_mode", "zeros"))
        if state_dict is not None:
            model.load_state_dict(state_dict, strict=True)
        return model

    monkeypatch.setattr(batches, "build_segmenter", build)
    monkeypatch.setattr(audit, "build_segmenter", build)
    source = tmp_path / "reference"
    batches.run_comparison(config, root, output_dir=source)
    return config, root, source


def test_paired_padding_run_with_tiffs_exports_all_diagnostics_and_preserves_reference(reference, tmp_path):
    config, root, source = reference
    original_config = copy.deepcopy(config)
    before = {p: file_digest(p) for p in source.rglob("*") if p.is_file()}
    output = tmp_path / "padding"
    result = comparison.run_comparison(config, root, reference_dir=source, output_dir=output)
    assert result["complete"] and result["active_arm"] is None and result["active_checkpoint"] is None
    assert config == original_config and before == {p: file_digest(p) for p in source.rglob("*") if p.is_file()}
    configs, schedules = [], []
    for padding in ("zeros", "replicate"):
        arm = result["arms"][padding]
        assert arm["layout"] == "mixed" and arm["decoder_padding_mode"] == padding
        assert arm["updates"] == 6 and arm["exposures_per_crop"] == 1
        assert arm["initial_state_sha256"] == result["initial_state_sha256"]
        assert arm["crop_pool_sha256"] == result["crop_pool_sha256"]
        updates = audit.read_rows(output / padding / "update_metrics.csv")
        schedules.append([r["crop_indices"] for r in updates])
        assert all(float(r["encoder_update_norm"]) > 0 and float(r["decoder_update_norm"]) > 0 for r in updates)
        cfg = json.loads((output / padding / "effective_config.json").read_text())
        assert cfg["model"].pop("decoder_padding_mode") == padding
        configs.append(cfg)
        for checkpoint in ("best", "last"):
            payload = torch.load(output / padding / f"{checkpoint}.pt", weights_only=False)
            assert payload["experiment"] == comparison.EXPERIMENT
            assert payload["config"]["model"]["decoder_padding_mode"] == padding
            gate_path = output / padding / ("best_gate.json" if checkpoint == "best" else "gate.json")
            assert json.loads(gate_path.read_text())["decoder_padding_mode"] == padding
            report = arm["probability_diagnostics"][checkpoint]
            assert report["reproduced_within_1e_4"] and report["reproduction_failures"] == []
            folder = output / padding / "probabilities" / checkpoint
            assert (folder / "metrics.npz").is_file()
            assert len(list(folder.glob("*_fixed.png"))) == 2
            regions = audit.read_rows(folder / "boundary_metrics.csv")
            negative = next(r for r in regions if r["group"] == "negative_crops" and r["region"] == "all")
            assert int(negative["fp"]) == report["fixed_threshold"]["negative_crop_fp"]
            assert float(negative["background_fp_fraction"]) == report["fixed_threshold"]["negative_crop_fp_fraction"]
            assert not (folder / "calibration.json").exists()
    assert configs[0] == configs[1] and schedules[0] == schedules[1]
    assert json.loads((output / "comparison.json").read_text()) == result
    with pytest.raises(ValueError, match="already contains"):
        comparison.run_comparison(config, root, reference_dir=source, output_dir=output)
    with pytest.raises(ValueError, match="outside the reference"):
        comparison.run_comparison(config, root, reference_dir=source, output_dir=source / "child")


@pytest.mark.parametrize("change", ["config", "incomplete", "ledger", "initial_state", "pool"])
def test_changed_reference_or_protocol_stops_before_training(reference, tmp_path, monkeypatch, change):
    config, root, source = reference
    if change == "config":
        config["preprocessing"]["clip"] = True
    elif change == "ledger":
        path = source / "crop_pool.csv"; path.write_text(path.read_text().replace("pix2pix", "changed", 1))
    else:
        path = source / "comparison.json"; report = json.loads(path.read_text())
        if change == "incomplete": report["complete"] = False
        elif change == "pool": report["crop_pool_sha256"] = "different"
        else: report["arms"]["grouped"]["initial_state_sha256"] = "different"
        path.write_text(json.dumps(report))
    monkeypatch.setattr(comparison, "run_arm", lambda *a, **kw: pytest.fail("changed reference must not train"))
    with pytest.raises(ValueError):
        comparison.run_comparison(config, root, reference_dir=source, output_dir=tmp_path / "bad")


def test_different_replicate_initialization_stops_before_its_training(reference, tmp_path, monkeypatch):
    config, root, source = reference
    build = batches.build_segmenter

    def changed(model_config):
        model = build(model_config)
        if model_config["decoder_padding_mode"] == "replicate":
            with torch.no_grad(): model.output.bias.add_(1)
        return model

    monkeypatch.setattr(batches, "build_segmenter", changed)
    output = tmp_path / "mismatch"
    with pytest.raises(ValueError, match="identical model states"):
        comparison.run_comparison(config, root, reference_dir=source, output_dir=output)
    report = json.loads((output / "comparison.json").read_text())
    assert not report["complete"] and not report["passed"] and report["active_arm"] == "replicate"
    assert "identical model states" in report["error"]
    assert not (output / "replicate/update_metrics.csv").exists()


def test_boundary_counts_match_direct_voxel_counts_including_empty_interiors():
    rng = np.random.default_rng(21)
    distributions, crops, inputs = [], [], []
    for i, shape in enumerate([(36, 36, 36), (36, 36, 36), (8, 8, 8)]):
        p = rng.random(shape).astype(np.float32); target = np.zeros(shape, dtype=bool)
        if i == 0: target[:20, :20, :20] = True
        _, dist = probability_statistics(p, target)
        crops.append(dict(pool_index=i, target_voxels=int(target.sum()), mod="pix2pix" if i == 0 else "real"))
        distributions.extend(dict(pool_index=i, **r) for r in dist)
        inputs.append((p, target))
    rows = comparison.boundary_summary(distributions, crops)
    for row in rows:
        selected = [0] if row["group"] == "positive_crops" else [1, 2]
        tp = fp = truth = background = 0
        for i in selected:
            p, target = inputs[i]; region = dict(crop_regions(p.shape))[row["region"]].numpy()
            tp += int(((p >= .5) & target & region).sum()); fp += int(((p >= .5) & ~target & region).sum())
            truth += int((target & region).sum()); background += int((~target & region).sum())
        assert (row["tp"], row["fp"], row["fn"], row["target_voxels"], row["background_voxels"]) == (tp, fp, truth-tp, truth, background)


def test_gate_rejects_different_padding_and_accepts_legacy_zero_default(tmp_path):
    manifest = {"manifest_id": "fixture"}
    (tmp_path / "audit.json").write_text(json.dumps({"passed": True, "manifest_id": "fixture", "cohorts": ["pilot_train"]}))
    gate = {"passed": True, "manifest_id": "fixture", "positive_crop_dice": .95,
            "negative_crop_fp_fraction": .001, "finite_gradients": True, "decoder_padding_mode": "replicate"}
    (tmp_path / "gate.json").write_text(json.dumps(gate))
    config = {"stage": "pilot", "manifest": str(tmp_path / "manifest.json"), "overfit_report": str(tmp_path / "gate.json")}
    with pytest.raises(ValueError, match="different decoder padding"):
        check_prerequisites(config, manifest)
    gate.pop("decoder_padding_mode"); (tmp_path / "gate.json").write_text(json.dumps(gate))
    config["model"] = {"decoder_padding_mode": "replicate"}
    with pytest.raises(ValueError, match="different decoder padding"):
        check_prerequisites(config, manifest)
    config["model"] = {}
    with pytest.raises(KeyError, match="baseline_report"):
        check_prerequisites(config, manifest)


def test_cli_runs_padding_comparison_and_returns_scientific_failure(tmp_path, monkeypatch):
    import sys
    calls = []
    monkeypatch.setattr(comparison, "run_comparison", lambda *a, **kw: calls.append((a, kw)) or {"passed": False})
    monkeypatch.setattr(sys, "argv", ["compare_segmentation_padding", "--data-root", str(tmp_path), "--device", "cuda"])
    with pytest.raises(SystemExit) as stopped:
        comparison.main()
    assert stopped.value.code == 2
    assert calls[0][0][0]["training"]["squared_pred"] is True and calls[0][1]["device"] == "cuda"
