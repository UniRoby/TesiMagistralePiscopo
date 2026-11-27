"""Window fusion identity and controlled paired-sampling experiments."""
from dataclasses import asdict, replace
import csv
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from _tiff_fixtures import make_fake_corpus, write_tiff_stack
from test_segmentation import TinySegmenter
from tesi_m3d import compare_segmentation_sampling as comparison
from tesi_m3d import diagnose_segmentation_windows as windows
from tesi_m3d.dataset import load_tiff_stack, scan_dir
from tesi_m3d.segmentation import predict_segmentation
from tesi_m3d.segmentation_data import SegmentationDataset, validate_experiment_output
from tesi_m3d.train import load_yaml_config


class Pointwise(torch.nn.Module):
    def forward(self, image):
        return image * 12 - 6


class PositionDependent(torch.nn.Module):
    def forward(self, image):
        z = torch.arange(image.shape[2], device=image.device)[None, None, :, None, None]
        return (z.float() - 3).expand_as(image)


@pytest.mark.parametrize("model", [Pointwise(), PositionDependent()])
def test_roi_fusion_matches_production_monai_even_at_shifted_end_windows(model):
    image = np.random.default_rng(3).random((17, 19, 21)).astype(np.float32)
    mask = np.zeros(image.shape, bool)
    mask[7:10, 8:12, 10:14] = True
    summary, rows, shared, fixed, arrays, _ = windows.inspect_windows(
        model, image, mask, [(5, 6, 7), (7, 8, 9)], (8, 8, 8))
    origin = summary["roi_origin_zyx"]
    roi = windows.slices(origin, summary["roi_shape"])
    production = predict_segmentation(model, image, patch_shape=(8, 8, 8), amp=False)
    np.testing.assert_allclose(arrays["probability"], production[roi], atol=2e-6)
    assert arrays["window_count"].min() > 0 and len(fixed) == 2
    assert shared and rows and summary["windows_intersecting_mask"] > 0
    if isinstance(model, Pointwise):
        assert max(r["max_abs_difference"] for r in shared) == 0
        assert np.max(arrays["window_max"] - arrays["window_min"]) == 0
        assert np.max(arrays["window_std"]) == 0
    else:
        assert max(r["max_abs_difference"] for r in shared) > 0.5
        assert summary["foreground_mean_window_range"] > 0.5
    json.dumps(summary)  # No numpy scalars in exported reports.


def test_window_missing_weights_and_output_protection(tmp_path):
    with pytest.raises(FileNotFoundError, match="trained checkpoint"):
        windows.run_diagnostic(tmp_path / "data", source_dir=tmp_path / "source", output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()
    with pytest.raises(ValueError, match="separate"):
        windows.run_diagnostic(tmp_path / "data", source_dir=tmp_path / "source", output_dir=tmp_path / "source/diag")


def test_project_inside_dataset_root_output_protection(tmp_path):
    root = tmp_path / "dataset"
    scan = root / "pix2pix/scan/case"
    label = root / "pix2pix/label/case"
    real = tmp_path / "originals"
    source = root / "project/outputs/source"
    for path in (scan, label, real, source):
        path.mkdir(parents=True)
    out = root / "project/outputs/diagnostic"
    validate_experiment_output(out, [source], root, real)
    with pytest.raises(FileNotFoundError, match="trained checkpoint"):
        windows.run_diagnostic(root, source_dir=source, output_dir=out, real_scan_root=real)
    for forbidden in (root, root.parent, scan, scan / "out", label, label / "out",
                      scan.parent, scan.parent.parent, real, real / "out",
                      source, source / "out", source.parent):
        with pytest.raises(ValueError, match="separate"):
            validate_experiment_output(forbidden, [source], root, real)
    alias = root / "project/data_alias"
    alias.symlink_to(scan, target_is_directory=True)
    with pytest.raises(ValueError, match="separate"):
        validate_experiment_output(alias / "out", [source], root, real)


def paired_fixture(tmp_path):
    root, records = make_fake_corpus(tmp_path / "data", n_records=1, n_real=1, scan_z=16,
                                     shape=(16, 16), cube_origin=(6, 6, 6), cube_size=(2, 2, 2))
    original = replace(records[0], mod="real")
    write_tiff_stack(root / "real/scan" / f"{original.orig_id}__{original.sdir_id}",
                     load_tiff_stack(scan_dir(root, records[0])))
    return root, records, original


@pytest.mark.parametrize("augment", [False, True])
def test_paired_changes_only_two_negatives_and_keeps_matching_flips(tmp_path, augment):
    root, records, original = paired_fixture(tmp_path)
    options = dict(patch_shape=(8, 8, 8), seed=21, augment=augment, volume_cache_size=1)
    baseline = SegmentationDataset(records, root, **options)
    paired = SegmentationDataset(records, root, paired_records=[original, records[1]], **options)
    for epoch in (0, 1, 3):
        a, b = baseline[(epoch, 0)], paired[(epoch, 0)]
        torch.testing.assert_close(a["image"][[0, 1, 2, 3, 6, 7]], b["image"][[0, 1, 2, 3, 6, 7]], rtol=0, atol=0)
        torch.testing.assert_close(a["mask"], b["mask"], rtol=0, atol=0)
        # Identical original image makes coordinates AND augmentation observable.
        torch.testing.assert_close(b["image"][:2], b["image"][4:6], rtol=0, atol=0)
        assert not b["mask"][4:].any()
        torch.testing.assert_close(baseline[(epoch, 1)]["image"], paired[(epoch, 1)]["image"], rtol=0, atol=0)
    bad = SegmentationDataset(records, root, paired_records=[replace(original, split="test"), records[1]], **options)
    with pytest.raises(ValueError, match="original train"):
        bad[0]


def test_pair_metadata_rejects_missing_cross_split_and_wrong_series(tmp_path):
    _, records, original = paired_fixture(tmp_path)
    for metadata in ([], [replace(original, split="valid")], [replace(original, sdir_id="other")]):
        with pytest.raises(ValueError, match="missing or mismatched"):
            comparison.resolve_pairs(records, metadata)
    pairs = comparison.resolve_pairs(records, [original, records[1]])
    assert pairs == [original, records[1]]


@pytest.mark.parametrize("variant", ["paired", "grid"])
def test_two_complete_pilots_share_initialization_and_budget(tmp_path, monkeypatch, variant):
    from tesi_m3d import train_segmentation as trainer, evaluate_segmentation as evaluator
    from tesi_m3d.segmentation_metrics import evaluate_records
    root, records, original = paired_fixture(tmp_path)
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    manifest = {"manifest_id": "fixture", "seed": 21, "spacing": {}, "cohorts": {
        k: [asdict(r) for r in records] for k in ("pilot_train", "pilot_valid", "common_valid")}}
    (protocol / "manifest.json").write_text(json.dumps(manifest))
    (protocol / "audit.json").write_text(json.dumps({"passed": True, "manifest_id": "fixture",
        "cohorts": ["pilot_train"], "data_root": str(root.resolve())}))
    (protocol / "gate.json").write_text(json.dumps({"passed": True, "manifest_id": "fixture",
        "positive_crop_dice": 0.95, "negative_crop_fp_fraction": 0, "finite_gradients": True,
        "training_loss": {"squared_pred": True}}))
    diagnostic = tmp_path / "diagnostic"
    diagnostic.mkdir()
    (diagnostic / "report.json").write_text(json.dumps({"task": "segmentation_windows", "complete": True, "manifest_id": "fixture"}))
    config = load_yaml_config("configs/segmentation_pilot_recovery.yaml")
    config.update(manifest=str(protocol / "manifest.json"), overfit_report=str(protocol / "gate.json"),
                  output_dir=str(tmp_path / "old"), cache_dir=str(tmp_path / "cache"), baseline_report=str(protocol / "baseline.json"))
    config["patches"]["patch_shape"] = [8, 8, 8]
    config["training"].update(epochs=2, num_workers=0, freeze_encoder_epochs=0, validation_interval=1)

    def build(_config, state_dict=None):
        model = TinySegmenter()
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return model

    def restore(path, device="cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return build(None, payload["model_state_dict"]).to(device), payload

    monkeypatch.setattr(comparison, "read_records", lambda _: [original, records[1]])
    monkeypatch.setattr(comparison, "build_segmenter", build)
    monkeypatch.setattr(trainer, "build_segmenter", build)
    monkeypatch.setattr(evaluator, "load_predictor", restore)
    baseline = evaluate_records(TinySegmenter(), {"config": config}, records, root)
    baseline.update(manifest_id="fixture", split="valid", model_name="resnet18_medicalnet")
    (protocol / "baseline.json").write_text(json.dumps(baseline))
    phase2 = tmp_path / "phase2"
    phase2.mkdir()
    (phase2 / "comparison.json").write_text(json.dumps({"task": "segmentation_sampling", "complete": True, "manifest_id": "fixture"}))
    extra = dict(variant=variant, sampling_report=phase2 / "comparison.json")
    out = tmp_path / "comparison"
    if variant == "grid":
        with pytest.raises(ValueError, match="phase-2"):
            comparison.run_comparison(config, root, window_report=diagnostic / "report.json",
                                      output_dir=out, variant="grid")
        assert not out.exists()
    result = comparison.run_comparison(config, root, window_report=diagnostic / "report.json", output_dir=out, **extra)
    assert result["complete"] and result["control"]["updates_per_arm"] == 4
    assert result["control"]["crops_per_arm"] == 32
    assert result["arms"]["standard"]["protocol_signature"] != result["arms"][variant]["protocol_signature"]
    for arm in ("standard", variant):
        payload = torch.load(out / arm / "last.pt", weights_only=False)
        assert payload["updates"] == 4 and payload["epoch"] == 2
        assert payload["config"]["sampling_protocol"] == arm
        assert (out / arm / "calibration.json").exists()
    with pytest.raises(ValueError, match="not empty"):
        comparison.run_comparison(config, root, window_report=diagnostic / "report.json", output_dir=out, **extra)
    rows = list(csv.DictReader((out / variant / "sampling.csv").open()))
    if variant == "grid":
        original_starts = rows[0]["crop_starts"]
        rows[0]["crop_starts"] = json.dumps([[1, 1, 1]] * 8)
        with (out / variant / "sampling.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
        with pytest.raises(ValueError, match="inference grid"):
            comparison.verify_sampling(out, 4, variant, (8, 8, 8))
        rows[0]["crop_starts"] = original_starts
    rows[0]["retained_image_sha256" if variant == "paired" else "flips"] = "tampered"
    with (out / variant / "sampling.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError, match="not controlled"):
        comparison.verify_sampling(out, 4, variant, (8, 8, 8))


def test_complete_window_diagnostic_preserves_checkpoint_and_binds_crop_ledger(tmp_path, monkeypatch):
    from tesi_m3d.compare_segmentation_batches import FixedCropPool
    from tesi_m3d.segmentation import file_digest
    root, records = make_fake_corpus(tmp_path / "ct", n_records=4, n_real=2, real_shared_dirs=2,
                                     scan_z=16, shape=(16, 16), cube_origin=(6, 6, 6), cube_size=(2, 2, 2))
    records[2:4] = [replace(r, ty="rem") for r in records[2:4]]
    manifest = {"manifest_id": "fixture", "seed": 21, "cohorts": {"overfit": [asdict(r) for r in records]}}
    config = {"overfit_protocol": "full_pool", "patches": {"patch_shape": [8, 8, 8]},
              "preprocessing": {"low": 1, "high": 99, "clip": False}}
    source = tmp_path / "source"
    (source / "full_pool/probabilities/best").mkdir(parents=True)
    checkpoint = source / "full_pool/best.pt"
    checkpoint.write_bytes(b"immutable fixture checkpoint")
    digest = file_digest(checkpoint)
    (source / "manifest.json").write_text(json.dumps(manifest))
    (source / "comparison.json").write_text(json.dumps({"complete": True, "passed": True, "manifest_id": "fixture"}))
    (source / "full_pool/probabilities/best/report.json").write_text(json.dumps({"checkpoint_sha256": digest}))
    pool = FixedCropPool(SegmentationDataset(records, root, patch_shape=(8, 8, 8), fixed=True, augment=False, seed=21))
    with (source / "crop_pool.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pool.rows[0]); writer.writeheader(); writer.writerows(pool.rows)
    monkeypatch.setattr(windows, "load_predictor", lambda *_: (Pointwise(), {
        "task": "segmentation", "manifest": manifest, "config": config}))
    report = windows.run_diagnostic(root, source_dir=source, output_dir=tmp_path / "out")
    assert report["complete"] and report["evaluation_only"] and len(report["cases"]) == 4
    assert len(list((tmp_path / "out").glob("*/shared_voxels.csv"))) == 4
    assert len(list((tmp_path / "out").glob("*/fused.png"))) == 4
    assert file_digest(checkpoint) == digest
    checkpoint.write_bytes(b"wrong checkpoint")
    with pytest.raises(ValueError, match="mismatch"):
        windows.run_diagnostic(root, source_dir=source, output_dir=tmp_path / "other")
    assert not (tmp_path / "other").exists()


def test_standard_sampler_preserves_original_random_draws(tmp_path):
    from tesi_m3d.segmentation_data import load_bundle, crop_plan
    root, records, _ = paired_fixture(tmp_path)
    image, mask, lung, _ = load_bundle(root, records[0])
    rng = np.random.default_rng([21, 3, 0])
    coords = crop_plan(mask, lung, (8, 8, 8), rng)
    expected_images, expected_masks = [], []
    for coord in coords:
        region = windows.slices(coord, (8, 8, 8))
        crop, target = image[region], mask[region]
        for axis in (1, 2):
            if rng.random() < 0.5:
                crop, target = np.flip(crop, axis), np.flip(target, axis)
        expected_images.append(crop.copy())
        expected_masks.append(target.copy())
    actual = SegmentationDataset(records, root, patch_shape=(8, 8, 8), seed=21, augment=True)[(3, 0)]
    np.testing.assert_array_equal(actual['image'].numpy()[:, 0], np.stack(expected_images))
    np.testing.assert_array_equal(actual['mask'].numpy()[:, 0], np.stack(expected_masks))


def test_grid_alignment_preserves_class_and_includes_shifted_end_windows():
    from monai.data.utils import dense_patch_slices
    from tesi_m3d.segmentation_data import align_crop_plan
    mask = np.zeros((19, 21, 23), dtype=bool)
    mask[9, 10, 12] = True
    patch = (8, 8, 8)
    coords = [(2, 3, 5), (11, 13, 15)]  # First crop barely contains the target.
    aligned = align_crop_plan(mask, patch, coords)
    allowed = {tuple(s.start for s in w) for w in dense_patch_slices(mask.shape, patch, (4, 4, 4))}
    assert aligned[1] == (11, 13, 15)
    for before, after in zip(coords, aligned):
        assert after in allowed
        assert mask[windows.slices(before, patch)].any() == mask[windows.slices(after, patch)].any()
    assert aligned == align_crop_plan(mask, patch, coords)
    # A clean arbitrary crop can exist without any clean inference window.
    almost_full = np.ones((9, 9, 9), dtype=bool)
    almost_full[1:8, 1:8, 1:8] = False
    with pytest.raises(ValueError, match="no window"):
        align_crop_plan(almost_full, (7, 7, 7), [(1, 1, 1)])
    with pytest.raises(ValueError, match="unpadded"):
        align_crop_plan(np.zeros((4, 4, 4), bool), patch, [(0, 0, 0)])


def test_grid_dataset_keeps_class_counts_flips_and_global_negative_slots(tmp_path):
    root, records, _ = paired_fixture(tmp_path)
    options = dict(patch_shape=(8, 8, 8), seed=21, augment=True)
    standard = SegmentationDataset(records, root, **options)
    grid = SegmentationDataset(records, root, grid_aligned=True, **options)
    for epoch in (0, 1, 3):
        for index in (0, 1):
            a, b = standard[(epoch, index)], grid[(epoch, index)]
            assert a['flips'] == b['flips']
            assert torch.equal(a['mask'].flatten(1).any(1), b['mask'].flatten(1).any(1))
            assert not b['mask'][4:].any()
            assert all(all(v % 4 == 0 for v in c) for c in json.loads(b['crop_starts']))
