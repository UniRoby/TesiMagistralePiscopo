"""Exact threshold counts, streaming AP, checkpoint binding and read-only crop audit."""
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from sklearn.metrics import average_precision_score
from _tiff_fixtures import make_fake_corpus
from test_segmentation import TinySegmenter
from tesi_m3d import compare_segmentation_batches as comparison
from tesi_m3d import diagnose_segmentation_batch_probabilities as audit
from tesi_m3d.diagnose_localization import write_json
from tesi_m3d.segmentation import file_digest
from tesi_m3d.segmentation_metrics import BINS, ScoreHistogram
from tesi_m3d.train import load_yaml_config


def test_threshold_counts_are_exact_including_ties_zero_and_one():
    p = np.array([0, .1, .5, .5, .7, .999999, 1, 1], dtype=np.float32).reshape(2, 2, 2)
    target = np.array([0, 1, 1, 0, 1, 0, 1, 0], dtype=bool).reshape(p.shape)
    counts = audit.threshold_counts(p, target)
    for index, threshold in enumerate(audit.THRESHOLDS):
        pred = p.astype(np.float64) >= threshold
        assert counts[index].tolist() == [int((pred & target).sum()), int((pred & ~target).sum()),
                                         int((~pred & target).sum()), int((~pred & ~target).sum())]
    assert counts[-1].tolist() == [1, 1, 3, 3]


def test_histogram_ap_matches_exact_quantized_reference_and_exposes_approximation():
    p = np.array([.500001, .500002, .9, .1, 1], dtype=np.float32)
    target = np.array([0, 1, 1, 0, 1], dtype=bool)
    hist = ScoreHistogram()
    hist.update(target, p)
    quantized = np.minimum((p.astype(np.float64) * BINS).astype(np.int64), BINS - 1)
    assert hist.ap() == pytest.approx(average_precision_score(target, quantized))
    assert hist.ap() != pytest.approx(average_precision_score(target, p))


def test_common_thresholds_are_not_per_crop_oracles_and_negative_constraint_is_strict():
    mask = np.array([1, 0], dtype=bool).reshape(1, 1, 2)
    probabilities = [np.array([.4, .3]), np.array([.8, .7]), np.array([.65, .65])]
    counts = np.stack([audit.threshold_counts(p.reshape(mask.shape), mask if i < 2 else np.zeros_like(mask))
                       for i, p in enumerate(probabilities)])
    rows, diagnostic = audit.profiles(counts, np.array([True, True, False]), np.array([False, False, True]))
    assert diagnostic["mean_per_positive_crop_oracle_dice"] == 1
    assert diagnostic["best_common_mean_dice"]["mean_positive_crop_dice"] == pytest.approx(5 / 6)
    feasible = diagnostic["best_common_with_negative_fp_below_1pct"]
    assert feasible["mean_positive_crop_dice"] == .5 and feasible["negative_crop_fp_fraction"] == 0
    assert not diagnostic["grid_contains_numeric_gate_solution"]
    assert rows[500]["mean_positive_crop_dice"] == pytest.approx(1 / 3)
    # Exactly 1% FP is excluded; accepting <= would produce a different diagnostic optimum.
    p = np.zeros((1, 10, 10), dtype=np.float32); p[0, 0, 0] = .9
    target = np.zeros_like(p, dtype=bool); target[0, 0, 0] = True
    positive = np.where(target, .8, 0)
    counts = np.stack([audit.threshold_counts(positive, target), audit.threshold_counts(p, np.zeros_like(target))])
    _, diagnostic = audit.profiles(counts, np.array([True, False]), np.array([False, True]))
    assert diagnostic["best_common_mean_dice"]["mean_positive_crop_dice"] == 1
    assert diagnostic["best_common_with_negative_fp_below_1pct"]["mean_positive_crop_dice"] == 0


def test_no_feasible_threshold_and_single_class_ap_remain_undefined():
    target = np.array([1, 0], dtype=bool).reshape(1, 1, 2)
    counts = np.stack([audit.threshold_counts(np.ones_like(target, dtype=float), m)
                       for m in (target, np.zeros_like(target))])
    rows, diagnostic = audit.profiles(counts, np.array([True, False]), np.array([False, True]))
    assert diagnostic["best_common_with_negative_fp_below_1pct"] is None
    assert not diagnostic["grid_contains_numeric_gate_solution"]
    assert rows[-1]["negative_crop_fp_fraction"] == 1
    hist = ScoreHistogram(); hist.update(np.zeros_like(target), np.ones_like(target, dtype=float))
    assert hist.ap() is None


@pytest.fixture
def source_run(tmp_path, monkeypatch):
    root, records = make_fake_corpus(tmp_path / "data", n_records=4, n_real=2, real_shared_dirs=2,
                                     scan_z=16, shape=(16, 16), cube_origin=(6, 6, 6), cube_size=(2, 2, 2))
    records = [replace(r, ty="rem") if i in (2, 3) else r for i, r in enumerate(records)]
    manifest = {"manifest_id": "fixture", "seed": 21, "cohorts": {"overfit": [asdict(r) for r in records]}}
    protocol = tmp_path / "protocol"; protocol.mkdir()
    write_json(protocol / "manifest.json", manifest)
    write_json(protocol / "audit.json", {"manifest_id": "fixture", "passed": True,
        "cohorts": ["overfit"], "data_root": str(root.resolve())})
    config = load_yaml_config("configs/segmentation_overfit_squared.yaml")
    config.update(manifest=str(protocol / "manifest.json"), cache_dir=str(tmp_path / "cache"))
    config["patches"]["patch_shape"] = [8, 8, 8]
    config["model"]["medicalnet_weights"] = str(tmp_path / "missing_medicalnet.pth")
    config["training"].update(epochs=1, max_updates=6)
    monkeypatch.setattr(comparison, "build_segmenter", lambda config: TinySegmenter())
    source = tmp_path / "source"
    comparison.run_comparison(config, root, output_dir=source)
    models = []

    def restore(config, *, state_dict):
        assert not Path(config["medicalnet_weights"]).exists()
        model = TinySegmenter()
        model.load_state_dict(state_dict, strict=True)
        models.append(model)
        return model

    monkeypatch.setattr(audit, "build_segmenter", restore)
    return root, source, models


def test_four_checkpoints_export_recomputable_metrics_without_training_or_source_changes(source_run, tmp_path, monkeypatch):
    root, source, models = source_run
    before = {p: file_digest(p) for p in source.rglob("*") if p.is_file()}
    monkeypatch.setattr(comparison, "make_optimizer", lambda *a: pytest.fail("audit cannot create an optimizer"))
    monkeypatch.setattr(comparison, "train_volume", lambda *a, **kw: pytest.fail("audit cannot train"))
    torch.manual_seed(999)
    output = tmp_path / "audit"
    result = audit.run_diagnosis(root, run_dir=source, output_dir=output)
    assert result["complete"] and result["evaluation_only"] and result["all_reproduced_within_1e_4"]
    assert result["active_checkpoint"] is None and "passed" not in result
    assert before == {p: file_digest(p) for p in source.rglob("*") if p.is_file()}
    assert len(models) == len(result["checkpoints"]) == 4
    assert all(not m.training and all(p.grad is None for p in m.parameters()) for m in models)
    for label, report in result["checkpoints"].items():
        folder = output / label
        assert report["checkpoint_sha256"] == file_digest(source / f"{label}.pt")
        assert report["training_loss"]["squared_pred"] is True
        with np.load(folder / "metrics.npz") as metrics:
            counts, histograms = metrics["counts"], metrics["histograms"]
            assert counts.shape == (48, 1001, 4) and histograms.shape == (48, 2, BINS)
            assert counts.dtype == histograms.dtype == np.int64
            np.testing.assert_array_equal(metrics["thresholds"], audit.THRESHOLDS)
            np.testing.assert_array_equal(counts[:, :, 0] + counts[:, :, 2],
                np.broadcast_to(histograms[:, 0].sum(1)[:, None], (48, 1001)))
            hist = ScoreHistogram(); hist.positive, hist.negative = histograms.sum(0)
            assert hist.ap() == report["voxel_ap"]["all_crops"]
            assert report["voxel_ap"]["real_crops"] is None
        crops = list(csv.DictReader((folder / "crops.csv").open()))
        positive = np.array([int(r["target_voxels"]) > 0 for r in crops])
        real = np.array([r["mod"] == "real" for r in crops])
        curves, diagnostic = audit.profiles(counts, positive, real)
        assert curves[500] == report["fixed_threshold"]
        assert diagnostic == report["diagnostic_train_only"]
        assert len(list(csv.DictReader((folder / "thresholds.csv").open()))) == 1001
        assert len(list(csv.DictReader((folder / "distributions.csv").open()))) == 48 * 15
        assert len(list(folder.glob("*_fixed.png"))) == 2
        assert (folder / "diagnostic_crops.csv").exists()
        assert not (folder / "gate.json").exists() and not (folder / "calibration.json").exists()
    assert json.loads((output / "report.json").read_text()) == result
    with pytest.raises(ValueError, match="already contains"):
        audit.run_diagnosis(root, run_dir=source, output_dir=output)
    with pytest.raises(ValueError, match="outside the source"):
        audit.run_diagnosis(root, run_dir=source, output_dir=source / "child")


def test_audit_matches_reference_seed_grad_mode_and_sigmoid_before_cpu(source_run, tmp_path, monkeypatch):
    root, source, _ = source_run
    seed, restore = audit.set_global_seed, audit.build_segmenter
    sigmoid, cpu = torch.Tensor.sigmoid, torch.Tensor.cpu
    events, current = [], {}

    def setup(value):
        events.append(("seed", value))
        seed(value)

    def capture(module, inputs, output):
        assert not torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
        current.clear()
        current["logits"] = output

    def build(*args, **kwargs):
        assert events[-1] == ("seed", 21)
        model = restore(*args, **kwargs)
        model.register_forward_hook(capture)
        return model

    def predict(self):
        result = sigmoid(self)
        if self is current.get("logits"):
            current["probability"] = result
        return result

    def transfer(self, *args, **kwargs):
        assert self is not current.get("logits"), "sigmoid must precede CPU transfer, as in overfit_metrics"
        if self is current.get("probability"):
            events.append(("probability_to_cpu", None))
        return cpu(self, *args, **kwargs)

    monkeypatch.setattr(audit, "set_global_seed", setup)
    monkeypatch.setattr(audit, "build_segmenter", build)
    monkeypatch.setattr(torch.Tensor, "sigmoid", predict)
    monkeypatch.setattr(torch.Tensor, "cpu", transfer)
    result = audit.run_diagnosis(root, run_dir=source, output_dir=tmp_path / "parity")
    assert events.count(("seed", 21)) == 4
    assert events.count(("probability_to_cpu", None)) == 4 * 48 // 2
    for report in result["checkpoints"].values():
        assert report["reproduction_failures"] == [] and report["sigmoid_device"] == "cpu"


@pytest.mark.parametrize("change", ["task", "config", "manifest", "sampling", "training_loss", "updates", "best_metrics", "state_dict"])
def test_checkpoint_binding_rejects_wrong_payload_and_reports_active_checkpoint(source_run, tmp_path, change):
    root, source, _ = source_run
    checkpoint = source / "grouped/best.pt"
    payload = torch.load(checkpoint, weights_only=False)
    if change == "task": payload["task"] = "segmentation_debug"
    elif change == "config": payload["config"]["preprocessing"]["clip"] = True
    elif change == "manifest": payload["manifest"]["manifest_id"] = "different"
    elif change == "sampling": payload["sampling"]["crop_pool_sha256"] = "different"
    elif change == "training_loss": payload["training_loss"]["squared_pred"] = False
    elif change == "updates": payload["updates"] += 1
    elif change == "best_metrics": payload["best_overfit_metrics"]["positive_crop_dice"] += .1
    else: payload["model_state_dict"].pop("output.weight")
    torch.save(payload, checkpoint)
    output = tmp_path / "rejected"
    with pytest.raises((ValueError, RuntimeError)):
        audit.run_diagnosis(root, run_dir=source, output_dir=output)
    report = json.loads((output / "report.json").read_text())
    assert not report["complete"] and report["active_checkpoint"] == "grouped/best" and report["error"]


def test_missing_checkpoint_is_rejected_before_loading_ct(source_run, tmp_path, monkeypatch):
    root, source, _ = source_run
    (source / "grouped/best.pt").unlink()
    monkeypatch.setattr(audit, "SegmentationDataset", lambda *a, **kw: pytest.fail("must not decode CT"))
    with pytest.raises(FileNotFoundError, match="laboratory PC"):
        audit.run_diagnosis(root, run_dir=source, output_dir=tmp_path / "audit")


@pytest.mark.parametrize("change", ["ledger", "image", "incomplete"])
def test_changed_inputs_or_incomplete_comparison_fail_before_model_restore(source_run, tmp_path, monkeypatch, change):
    root, source, _ = source_run
    if change == "ledger":
        path = source / "crop_pool.csv"
        path.write_text(path.read_text().replace("pix2pix", "changed", 1))
    elif change == "image":
        dataset = audit.SegmentationDataset

        class Changed(dataset):
            def __getitem__(self, key):
                batch = super().__getitem__(key)
                batch["image"] = batch["image"] + .1
                return batch

        monkeypatch.setattr(audit, "SegmentationDataset", Changed)
    else:
        path = source / "comparison.json"
        report = json.loads(path.read_text()); report["complete"] = False; write_json(path, report)
    monkeypatch.setattr(audit, "build_segmenter", lambda *a, **kw: pytest.fail("changed inputs must not predict"))
    with pytest.raises(ValueError):
        audit.run_diagnosis(root, run_dir=source, output_dir=tmp_path / "audit")


def test_prediction_drift_and_nonfinite_logits_stop_diagnosis(source_run, tmp_path, monkeypatch):
    root, source, _ = source_run
    checkpoint = source / "grouped/best.pt"
    payload = torch.load(checkpoint, weights_only=False)
    payload["model_state_dict"]["output.bias"].fill_(9)
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="not reproduced.*delta_saved_peak_probability.*pool_index=0") as error:
        audit.run_diagnosis(root, run_dir=source, output_dir=tmp_path / "drift")
    report = json.loads((tmp_path / "drift/grouped/best/report.json").read_text())
    assert not report["reproduced_within_1e_4"]
    assert all("delta_saved_peak_probability" in failure for failure in report["reproduction_failures"])
    assert "report.json" in str(error.value)

    class Nonfinite(TinySegmenter):
        def forward(self, image): return image * float("nan")

    monkeypatch.setattr(audit, "build_segmenter", lambda *a, **kw: Nonfinite())
    with pytest.raises(ValueError, match="non-finite"):
        audit.run_diagnosis(root, run_dir=source, output_dir=tmp_path / "nan")


def test_cli_exposes_only_evaluation_arguments(tmp_path, monkeypatch, capsys):
    import sys
    calls = []
    monkeypatch.setattr(audit, "run_diagnosis", lambda *a, **kw: calls.append((a, kw)) or {"checkpoints": dict.fromkeys(range(4))})
    monkeypatch.setattr(sys, "argv", ["diagnose_segmentation_batch_probabilities", "--data-root", str(tmp_path), "--device", "cuda"])
    audit.main()
    assert calls[0][1]["device"] == "cuda"
    assert "4 checkpoints; fixed .5 gate unchanged" in capsys.readouterr().out
