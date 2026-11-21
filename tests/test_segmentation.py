"""MedicalNet segmentation contracts, geometry, streaming metrics and staged training."""
from dataclasses import asdict, replace
import copy
import hashlib
import json
import pickle
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from test_model import encoder_weights
from _tiff_fixtures import make_fake_corpus, write_tiff_stack
from tesi_m3d.segmentation import (SEGMENTER_NAME, build_segmenter, predict_segmentation,
                                   preprocess_scan, read_calibration, segmentation_loss)
from tesi_m3d.segmentation_data import (SegmentationDataset, audit_records, crop_plan, make_manifest,
                                        stratified, validate_stack)
from tesi_m3d.segmentation_metrics import BINS, ScoreHistogram, compare_baseline, evaluate_records, summarize
from tesi_m3d.train import load_yaml_config
from tesi_m3d.train_segmentation import check_prerequisites, run_training, train_volume
from tesi_m3d.volume_io import align_mask_to_scan


def test_segmenter_loads_all_weights_freezes_bn_and_restores(tmp_path, monkeypatch, encoder_weights):
    import tesi_m3d.model as module
    weight_file = tmp_path / "weights.pth"
    weight_file.write_bytes(b"offline segmentation fixture")
    monkeypatch.setattr(module, "MEDICALNET_SHA256", hashlib.sha256(weight_file.read_bytes()).hexdigest())
    config = {"name": SEGMENTER_NAME, "medicalnet_weights": str(weight_file)}
    with patch.object(torch, "load", return_value={"state_dict": encoder_weights}):
        model = build_segmenter(config)
    for name, tensor in encoder_weights.items():
        assert torch.equal(model.encoder.state_dict()[name.removeprefix("module.")], tensor)
    model.train()
    norms = [m for m in model.encoder.modules() if isinstance(m, torch.nn.BatchNorm3d)]
    running = [m.running_mean.clone() for m in norms]
    assert len(norms) == 17 and all(not m.training for m in norms)
    assert all(m.weight.requires_grad for m in norms)
    image = torch.randn(2, 1, 16, 16, 16)
    target = torch.zeros_like(image)
    target[0, :, 3:8, 4:9, 4:9] = 1
    logits = model(image)
    assert logits.shape == target.shape
    loss = segmentation_loss(logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.output.weight.grad.abs().sum() > 0
    assert model.encoder.conv1.weight.grad.abs().sum() > 0
    for norm, before in zip(norms, running):
        assert torch.equal(norm.running_mean, before)
    model.freeze_encoder(True)
    assert not any(p.requires_grad for p in model.encoder.parameters())
    model.freeze_encoder(False)
    restored = build_segmenter({"name": SEGMENTER_NAME, "medicalnet_weights": "missing.pth"}, state_dict=model.state_dict())
    model.eval()
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(image), restored(image))
        odd = restored(torch.zeros(1, 1, 17, 19, 21))
        assert odd.shape == (1, 1, 17, 19, 21)


def test_segmenter_rejects_missing_weights_and_incomplete_state(tmp_path):
    with pytest.raises(FileNotFoundError, match="random initialization"):
        build_segmenter({"name": SEGMENTER_NAME, "medicalnet_weights": str(tmp_path / "absent")})
    with pytest.raises(RuntimeError, match="Missing key"):
        build_segmenter({"name": SEGMENTER_NAME}, state_dict={})


def test_empty_and_partial_targets_have_finite_loss_and_gradients():
    logits = torch.zeros(2, 1, 8, 8, 8, requires_grad=True)
    target = torch.zeros_like(logits)
    loss = segmentation_loss(logits, target)
    torch.testing.assert_close(loss, 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(logits, target))
    loss.backward()
    assert logits.grad.min() > 0
    target[0, 0, 0, 0, 0] = 1
    assert torch.isfinite(segmentation_loss(logits, target))
    with pytest.raises(ValueError, match="matching"):
        segmentation_loss(logits, target[:, :, :2])


def test_gradient_accumulation_matches_effective_batch_loss():
    torch.manual_seed(7)
    first = torch.nn.Conv3d(1, 1, 1)
    second = copy.deepcopy(first)
    image = torch.randn(8, 1, 8, 8, 8)
    target = torch.zeros_like(image)
    target[:4, :, 2:5, 2:5, 2:5] = 1
    optimizer = torch.optim.SGD(first.parameters(), lr=0.01)
    reference = torch.optim.SGD(second.parameters(), lr=0.01)
    measured = train_volume(first, {"image": image, "mask": target}, optimizer, microbatch=1, accumulation=8)
    loss = segmentation_loss(second(image), target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(second.parameters(), 1.0)
    reference.step()
    assert measured == pytest.approx(float(loss.detach()), rel=1e-6)
    for a, b in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(a, b)


def test_streaming_metrics_match_exact_quantized_ap_and_auc():
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(2)
    scores = rng.uniform(0, 1, 4096)
    scores[:4] = [0, 0.5, 1, 0.5]
    truth = rng.uniform(0, 1, 4096) > 0.9
    hist = ScoreHistogram()
    hist.update(truth[:2048], scores[:2048])
    hist.update(truth[2048:], scores[2048:])
    quantized = np.minimum(np.floor(scores * BINS), BINS - 1) / BINS
    assert hist.ap() == pytest.approx(average_precision_score(truth, quantized), abs=1e-12)
    assert hist.auc() == pytest.approx(roc_auc_score(truth, quantized), abs=1e-12)
    assert abs(hist.ap() - average_precision_score(truth, scores)) < 1e-3
    threshold = hist.best_threshold()
    counts = hist.counts(threshold)
    assert counts["tp"] == np.count_nonzero(truth & (quantized >= threshold))
    empty = ScoreHistogram()
    empty.update(np.zeros(4, bool), np.zeros(4))
    assert empty.ap() is None and empty.auc() is None
    foreground_only = ScoreHistogram()
    foreground_only.update(np.ones(4, bool), np.zeros(4))
    assert foreground_only.ap() is None and foreground_only.auc() is None
    with pytest.raises(ValueError, match="calibration"):
        empty.best_threshold()
    with pytest.raises(ValueError, match="finite"):
        hist.update([True], [np.nan])


class PointwisePredictor(torch.nn.Module):
    def forward(self, x):
        return 4 * x - 2


@pytest.mark.parametrize("shape", [(17, 19, 21), (5, 6, 7)])
def test_sliding_windows_preserve_coordinates_coverage_and_batch_independence(shape):
    image = np.zeros(shape, np.float32)
    image[tuple(s // 2 for s in shape)] = 1
    expected = torch.sigmoid(4 * torch.from_numpy(image) - 2).numpy()
    for batch in (1, 3):
        actual = predict_segmentation(PointwisePredictor(), image, patch_shape=(8, 8, 8), batch_size=batch)
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        assert np.argmax(actual) == np.argmax(image)


def test_strict_masks_and_stack_gaps_are_rejected(tmp_path):
    mask = np.zeros((3, 4, 4), bool)
    mask[-1] = True
    with pytest.raises(ValueError, match="discard positive"):
        align_mask_to_scan(mask, (2, 4, 4), "bad", strict=True)
    mask[-1] = False
    assert align_mask_to_scan(mask, (2, 4, 4), strict=True).shape == (2, 4, 4)
    path = write_tiff_stack(tmp_path / "scan", np.zeros((3, 4, 4), np.uint16))
    (path / "slide0001.tiff").unlink()
    with pytest.raises(ValueError, match="gappy"):
        validate_stack(path)


def test_crops_keep_small_intersections_and_clean_negatives():
    mask = np.zeros((24, 24, 24), bool)
    mask[11, 11, 11] = True
    lung = np.ones_like(mask)
    plan = crop_plan(mask, lung, (8, 8, 8), np.random.default_rng(21))
    assert plan == crop_plan(mask, lung, (8, 8, 8), np.random.default_rng(21))
    for i, coord in enumerate(plan):
        selected = mask[tuple(slice(c, c + 8) for c in coord)]
        assert selected.any() == (i < 4)
    assert len(set(plan[:4])) > 1
    with pytest.raises(ValueError, match="no clean crop"):
        crop_plan(np.ones((8, 8, 8), bool), np.ones((8, 8, 8), bool), (8, 8, 8), np.random.default_rng(21))


def test_dataset_caches_bundle_and_preserves_flip_alignment(tmp_path, monkeypatch):
    import tesi_m3d.segmentation_data as module
    root, records = make_fake_corpus(tmp_path, n_records=1, scan_z=16, shape=(16, 16),
                                     cube_origin=(6, 6, 6), cube_size=(2, 2, 2))
    mask = np.zeros((16, 16, 16), bool)
    mask[6:8, 6:8, 6:8] = True
    calls = []

    def bundle(*args):
        calls.append(1)
        return mask.astype(np.float32), mask, np.ones_like(mask), -3.0

    monkeypatch.setattr(module, "load_bundle", bundle)
    dataset = SegmentationDataset(records, root, patch_shape=(8, 8, 8), augment=True)
    for _ in range(2):
        batch = dataset[(0, 0)]
        torch.testing.assert_close(batch["image"], batch["mask"])
        assert batch["image"].shape == (8, 1, 8, 8, 8)
    assert len(calls) == 1
    assert len(pickle.loads(pickle.dumps(dataset)).cache) == 0
    fixed = SegmentationDataset(records, root, patch_shape=(24, 24, 24), fixed=True, augment=False)
    # Mask is tiny and a 24^3 crop covers the entire padded volume: no clean crop exists.
    with pytest.raises(ValueError, match="no clean crop"):
        fixed[(0, 0)]


def test_preprocessing_clipping_is_the_only_input_change(tmp_path):
    raw = np.arange(512, dtype=np.uint16).reshape(8, 8, 8)
    raw[-1, -1, -1] = 65000
    path = write_tiff_stack(tmp_path / "scan", raw)
    original, bg = preprocess_scan(raw, path, tmp_path / "cache", {"clip": False})
    clipped, bg_clip = preprocess_scan(raw, path, tmp_path / "cache", {"clip": True})
    from tesi_m3d.dataset import normalize_percentile
    np.testing.assert_array_equal(original, normalize_percentile(raw))
    np.testing.assert_array_equal(clipped, np.clip(original, 0, 1))
    assert bg < 0 and bg_clip == 0
    assert len(list((tmp_path / "cache/percentiles").glob("*.json"))) == 1


def test_official_manifests_match_baseline_and_cohorts_are_nested(tmp_path):
    manifest = make_manifest(load_yaml_config("configs/segmentation_pilot.yaml"), tmp_path)
    assert {k: len(v) for k, v in manifest["cohorts"].items()} == {
        "overfit": 6, "pilot_train": 64, "pilot_valid": 8, "full_train": 2026, "full_valid": 16, "common_valid": 58}
    ids = lambda name: {(r["mod"], r["img_id"]) for r in manifest["cohorts"][name]}
    assert ids("overfit") <= ids("pilot_train") <= ids("full_train")
    assert ids("pilot_valid") <= ids("full_valid") <= ids("common_valid")
    train_patients = {r["orig_id"] for r in manifest["cohorts"]["full_train"]}
    assert not train_patients & {r["orig_id"] for r in manifest["cohorts"]["common_valid"]}


def test_frozen_thresholds_and_baseline_gates():
    pos, neg = ScoreHistogram(), ScoreHistogram()
    pos.update([True, False], [0.8, 0.1])
    neg.update([False, False], [0.2, 0.1])
    entries = [({"img_id": "a", "is_manipulated": True, "detection_score": 0.8,
                 "peak_inside_mask": True, "spacing_band": "unknown"}, pos, None),
               ({"img_id": "b", "is_manipulated": False, "detection_score": 0.2,
                 "peak_inside_mask": False, "spacing_band": "unknown"}, neg, None)]
    calibrated, _ = summarize(entries)
    frozen = {**calibrated["calibration"], "localization_threshold": 0.9}
    evaluated, _ = summarize(entries, frozen)
    assert evaluated["micro_dice"] == 0 and evaluated["calibration"]["localization_threshold"] == 0.9
    calibrated["cohort_id"] = "same"
    baseline = {**calibrated, "mean_positive_ap": 0.1}
    assert compare_baseline(calibrated, baseline)["passed"]
    with pytest.raises(ValueError, match="same evaluation cohort"):
        compare_baseline(calibrated, {**baseline, "cohort_id": "different"})


def test_calibration_is_bound_to_checkpoint(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"one checkpoint")
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"source_split": "valid", "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                "localization_threshold": 0.4, "detection_threshold": 0.7}))
    assert read_calibration(path, checkpoint)["localization_threshold"] == 0.4
    checkpoint.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="exact checkpoint"):
        read_calibration(path, checkpoint)


class TinySegmenter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Conv3d(1, 2, 1)
        self.output = torch.nn.Conv3d(2, 1, 1)

    def freeze_encoder(self, frozen):
        self.encoder.requires_grad_(not frozen)

    def forward(self, image):
        return self.output(self.encoder(image))


def test_staged_training_evaluation_and_resume_with_tiff_fixture(tmp_path, monkeypatch):
    import tesi_m3d.train_segmentation as trainer
    import tesi_m3d.evaluate_segmentation as evaluator
    root, records = make_fake_corpus(tmp_path / "data", n_records=1, n_real=1, scan_z=16, shape=(16, 16),
                                     cube_origin=(6, 6, 6), cube_size=(2, 2, 2))
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    manifest = {"manifest_id": "synthetic", "seed": 21, "spacing": {}, "cohorts":
                {name: [asdict(r) for r in records] for name in ("overfit", "pilot_train", "pilot_valid", "common_valid")}}
    (protocol / "manifest.json").write_text(json.dumps(manifest))
    (protocol / "audit.json").write_text(json.dumps({"passed": True, "manifest_id": "synthetic", "cohorts": ["pilot_train"],
                                                    "data_root": str(root.resolve())}))
    gate = protocol / "overfit.json"
    gate.write_text(json.dumps({"passed": False, "manifest_id": "synthetic"}))
    config = load_yaml_config("configs/segmentation_pilot.yaml")
    config.update(manifest=str(protocol / "manifest.json"), overfit_report=str(gate),
                  output_dir=str(tmp_path / "run"), cache_dir=str(tmp_path / "cache"))
    config["patches"]["patch_shape"] = [8, 8, 8]
    config["training"].update(epochs=2, num_workers=0, freeze_encoder_epochs=0)
    config["baseline_report"] = str(protocol / "baseline.json")
    with pytest.raises(ValueError, match="overfit gate failed"):
        check_prerequisites(config, manifest)
    gate.write_text(json.dumps({"passed": True, "manifest_id": "synthetic", "positive_crop_dice": 0.95,
                                "negative_crop_fp_fraction": 0.001, "finite_gradients": True}))

    def build(config, state_dict=None):
        model = TinySegmenter()
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return model

    def restore(path, device="cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return build(payload["config"]["model"], payload["model_state_dict"]).to(device), payload

    monkeypatch.setattr(trainer, "build_segmenter", build)
    monkeypatch.setattr(evaluator, "load_predictor", restore)
    baseline = evaluate_records(TinySegmenter(), {"config": config}, records, root, batch_size=2)
    baseline.update(manifest_id="synthetic", split="valid", model_name="resnet18_medicalnet")
    (protocol / "baseline.json").write_text(json.dumps(baseline))
    result = run_training(config, root)
    assert result["n_volumes"] == 2
    assert (tmp_path / "run/best.pt").exists() and (tmp_path / "run/last.pt").exists()
    assert (tmp_path / "run/calibration.json").exists()
    assert len(list((tmp_path / "run/evaluation/images").glob("*.png"))) == 2
    with pytest.raises(ValueError, match="already exists"):
        run_training(config, root)
    config["training"]["epochs"] = 3
    run_training(config, root, resume=tmp_path / "run/last.pt")
    assert len((tmp_path / "run/metrics.csv").read_text().splitlines()) == 4
    config["preprocessing"]["clip"] = True
    with pytest.raises(ValueError, match="resume would change"):
        run_training(config, root, resume=tmp_path / "run/last.pt")
    debug = copy.deepcopy(config)
    debug.update(stage="overfit", output_dir=str(tmp_path / "overfit"))
    debug["training"].update(epochs=1, max_updates=1)
    (protocol / "audit.json").write_text(json.dumps({"passed": True, "manifest_id": "synthetic", "cohorts": ["overfit"],
                                                    "data_root": str(root.resolve())}))
    overfit = run_training(debug, root)
    assert overfit["updates"] == 1 and overfit["finite_gradients"]
    assert not overfit["passed"]
    assert (tmp_path / "overfit/gate.json").exists()


def test_full_training_recomputes_gates_and_selects_only_controlled_pilots(tmp_path):
    from tesi_m3d.segmentation_metrics import cohort_id
    _, records = make_fake_corpus(tmp_path / "data", n_records=1, n_real=1, scan_z=8, shape=(8, 8))
    manifest = {"manifest_id": "fixed", "cohorts": {name: [asdict(r) for r in records]
                for name in ("pilot_valid", "common_valid")}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "audit.json").write_text(json.dumps({"passed": True, "manifest_id": "fixed", "cohorts": ["full_train"]}))
    overfit = {"passed": True, "manifest_id": "fixed", "positive_crop_dice": 0.95,
               "negative_crop_fp_fraction": 0, "finite_gradients": True}
    (tmp_path / "overfit.json").write_text(json.dumps(overfit))
    base = {"manifest_id": "fixed", "split": "valid", "model_name": "resnet18_medicalnet",
            "cohort_id": cohort_id(records), "histogram_bins": BINS,
            "mean_positive_ap": 0.1, "real_false_positive_fraction": 0.01}
    (tmp_path / "base.json").write_text(json.dumps(base))
    config = {"stage": "full", "manifest": str(tmp_path / "manifest.json"),
              "overfit_report": str(tmp_path / "overfit.json"), "baseline_report": str(tmp_path / "base.json"),
              "pilot_baseline_report": str(tmp_path / "base.json"),
              "pilot_reports": [str(tmp_path / "a.json"), str(tmp_path / "b.json")]}
    report = {**base, "protocol_signature": "same", "preprocessing": {"clip": False}, "checkpoint_sha256": "a",
              "n_positive_volumes": 1, "micro_dice": 0.5, "peak_hits": 1, "dense_ap": 0.9,
              "gate": {"passed": True}}  # An old/edited passed flag cannot bypass recomputation.
    (tmp_path / "a.json").write_text(json.dumps(report))
    second = {**report, "preprocessing": {"clip": True}, "checkpoint_sha256": "b"}
    (tmp_path / "b.json").write_text(json.dumps(second))
    with pytest.raises(ValueError, match="neither pilot"):
        check_prerequisites(config, manifest)
    second["mean_positive_ap"] = 0.5
    (tmp_path / "b.json").write_text(json.dumps(second))
    check_prerequisites(config, manifest)
    assert config["preprocessing"]["clip"] is True and config["selected_pilot_sha256"] == "b"
    second["protocol_signature"] = "different training settings"
    (tmp_path / "b.json").write_text(json.dumps(second))
    with pytest.raises(ValueError, match="differ only in clipping"):
        check_prerequisites(config, manifest)


def test_volume_items_work_with_spawned_persistent_workers(tmp_path):
    from tesi_m3d.segmentation_data import VolumeOrder
    root, records = make_fake_corpus(tmp_path, n_records=1, n_real=1, scan_z=16, shape=(16, 16),
                                     cube_origin=(6, 6, 6), cube_size=(2, 2, 2))
    dataset = SegmentationDataset(records, root, patch_shape=(8, 8, 8), cache_dir=tmp_path / "cache")
    order = VolumeOrder(len(dataset))
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, sampler=order, num_workers=2,
                                         multiprocessing_context="spawn", persistent_workers=True, timeout=30)
    for epoch in (0, 1):
        order.epoch = epoch
        batches = list(loader)
        assert len(batches) == 2
        for batch in batches:
            assert batch["image"].shape == batch["mask"].shape == (8, 1, 8, 8, 8)
            assert int(batch["mask"].flatten(1).any(1).sum()) in (0, 4)
    assert not len(dataset.cache)  # Workers own their bounded caches.


def test_inference_cli_uses_only_ct_checkpoint_and_bound_calibration(tmp_path, monkeypatch):
    import sys
    import tesi_m3d.infer_segmentation as cli
    raw = np.arange(512, dtype=np.uint16).reshape(8, 8, 8)
    directory = write_tiff_stack(tmp_path / "scan", raw)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint identity")
    calibration = {"source_split": "valid", "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   "localization_threshold": 0.5, "detection_threshold": 0.7, "histogram_bins": BINS}
    (tmp_path / "calibration.json").write_text(json.dumps(calibration))
    payload = {"config": {"model": {"name": SEGMENTER_NAME}, "patches": {"patch_shape": [8, 8, 8]},
                           "preprocessing": {"low": 1.0, "high": 99.0, "clip": True}}}
    monkeypatch.setattr(cli, "load_predictor", lambda *a: (PointwisePredictor(), payload))
    output = tmp_path / "prediction"
    monkeypatch.setattr(sys, "argv", ["infer_segmentation", "--checkpoint", str(checkpoint),
                                      "--volume-dir", str(directory), "--out-dir", str(output)])
    cli.main()
    probability = np.load(output / "probability.npy")
    mask = np.load(output / "mask.npy")
    normalized, _ = preprocess_scan(raw, directory, preprocessing=payload["config"]["preprocessing"])
    expected = torch.sigmoid(4 * torch.from_numpy(normalized) - 2).numpy()
    np.testing.assert_allclose(probability, expected, atol=1e-6)
    np.testing.assert_array_equal(mask, expected >= 0.5)
    assert probability.dtype == np.float32 and mask.dtype == bool


def test_separate_real_scan_root_is_used_for_audit_and_loading(tmp_path):
    from tesi_m3d.segmentation_data import load_bundle
    root, records = make_fake_corpus(tmp_path / "original", n_records=0, n_real=1, scan_z=16, shape=(16, 16))
    moved = tmp_path / "separate_reals"
    (root / "real/scan").replace(moved)
    report = audit_records(records, root, moved)
    image, mask, _, _ = load_bundle(root, records[0], real_scan_root=moved)
    assert report[0]["positive_voxels"] == 0
    assert image.shape == mask.shape == (16, 16, 16)


def test_balanced_paired_audit_selects_sixteen_of_each_type(tmp_path, monkeypatch):
    from _tiff_fixtures import make_fake_record
    from tesi_m3d.audit_intensity import IntensityAuditor
    from tesi_m3d.prepare_segmentation import BalancedArtifactAuditor
    fake = [replace(make_fake_record(img_id=f"{kind}_{index}"), ty=kind)
            for kind in ("inj", "rem") for index in range(20)]
    pairs = [(replace(record, mod="real"), record) for record in fake]
    monkeypatch.setattr(IntensityAuditor, "_pairs", lambda self: pairs)
    auditor = BalancedArtifactAuditor(tmp_path, max_pairs=None, seed=21)
    selected = auditor._pairs()
    assert len(selected) == 32
    assert sum(fake.ty == "inj" for _, fake in selected) == 16
    assert selected == auditor._pairs()


def test_prepare_writes_audits_and_rejects_positive_mask_truncation(tmp_path, monkeypatch):
    import tesi_m3d.prepare_segmentation as preparer
    root, records = make_fake_corpus(tmp_path / "data", n_records=1, n_real=1, scan_z=8, shape=(8, 8),
                                     cube_origin=(2, 2, 2), cube_size=(2, 2, 2))
    manifest = {"manifest_id": "prepared", "seed": 21, "cohorts":
                {name: [asdict(r) for r in records] for name in ("overfit", "pilot_train", "common_valid", "full_train")}}
    monkeypatch.setattr(preparer, "make_manifest", lambda *a: manifest)
    def paired(self, destination):
        destination.mkdir(parents=True, exist_ok=True)
        return {"n_pairs": 32}

    monkeypatch.setattr(preparer.BalancedArtifactAuditor, "run", paired)
    output = tmp_path / "protocol"
    preparer.prepare({}, root, output)
    report = json.loads((output / "audit.json").read_text())
    assert report["passed"] and report["data_root"] == str(root.resolve())
    assert json.loads((output / "paired_artifacts/summary.json").read_text())["n_pairs"] == 32
    # A later invalid annotation must replace the old successful audit with a failure.
    extra = write_tiff_stack(tmp_path / "extra", np.ones((1, 8, 8), bool))
    (extra / "slide0000.tiff").replace(root / "pix2pix/label/inj_1/slide0008.tiff")
    with pytest.raises(ValueError, match="discard positive"):
        preparer.prepare({}, root, output)
    assert not json.loads((output / "audit.json").read_text())["passed"]


def test_checkpoint_write_failure_keeps_previous_complete_file(tmp_path, monkeypatch):
    from tesi_m3d.train_segmentation import save_checkpoint
    path = tmp_path / "best.pt"
    path.write_bytes(b"previous complete checkpoint")

    def interrupted(payload, temporary):
        temporary.write_bytes(b"partial")
        raise OSError("disk write interrupted")

    monkeypatch.setattr(torch, "save", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        save_checkpoint({}, path)
    assert path.read_bytes() == b"previous complete checkpoint"
    assert not path.with_suffix(".pt.tmp").exists()


def test_probability_plot_does_not_invent_anatomical_contrast(tmp_path):
    from PIL import Image
    from tesi_m3d.segmentation_metrics import save_preview
    planes = [("constant", np.full((8, 8), index % 2, np.float32), np.zeros((8, 8), bool),
               np.full((8, 8), 0.25, np.float32)) for index in range(6)]
    path = tmp_path / "constant.png"
    save_preview(path, planes, 0.5)
    rgb = np.asarray(Image.open(path))
    np.testing.assert_array_equal(rgb[100, 400], [63, 0, 191])
    np.testing.assert_array_equal(rgb[444, 400], [63, 0, 191])


def test_pilot_configs_differ_only_in_clipping_and_output():
    from tesi_m3d.segmentation import protocol_signature
    first = load_yaml_config("configs/segmentation_pilot.yaml")
    second = load_yaml_config("configs/segmentation_pilot_clip.yaml")
    assert first["preprocessing"]["clip"] is False and second["preprocessing"]["clip"] is True
    assert protocol_signature(first) == protocol_signature(second)
    second["preprocessing"]["low"] = 2
    assert protocol_signature(first) != protocol_signature(second)
