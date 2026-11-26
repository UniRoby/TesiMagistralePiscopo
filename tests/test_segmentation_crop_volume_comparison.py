"""The diagnostic compares fixed crops and original volumes without changing weights."""
import csv
from dataclasses import asdict, replace
import json
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from _tiff_fixtures import make_fake_corpus, write_tiff_stack
from test_segmentation_batch_comparison import FakeDataset
from tesi_m3d import compare_segmentation_crop_volume as comparison
from tesi_m3d.compare_segmentation_batches import FixedCropPool
from tesi_m3d.segmentation import file_digest
from tesi_m3d.segmentation_data import SegmentationDataset


class PointwiseSegmenter(torch.nn.Module):
    def forward(self, image):
        return 20 * image - 10


def test_fixed_crops_keep_positive_and_real_metrics_separate():
    pool = FixedCropPool(FakeDataset())
    summary, rows, calibrated = comparison.evaluate_crops(
        PointwiseSegmenter(), pool, batch_size=2, validation_threshold=0.7)
    assert len(rows) == 48 and summary["n_positive_crops"] == 16
    assert summary["mean_positive_dice"] == calibrated["mean_positive_dice"] == 1
    assert summary["mean_positive_ap"] == 1
    assert summary["negative_crop_fp_fraction"] == 0
    assert summary["real_false_positive_fraction"] == calibrated["real_false_positive_fraction"] == 0
    assert sum(row["positive_voxels"] for row in rows) == 16 * 8
    assert all(row["validation_tp"] == row["tp"] for row in rows)


def test_dense_metrics_use_original_voxels_and_fixed_validation_threshold(tmp_path, monkeypatch):
    root, records = make_fake_corpus(tmp_path / "ct", n_records=1, n_real=1, scan_z=16,
                                     shape=(16, 16), cube_origin=(6, 6, 6), cube_size=(2, 2, 2))
    fake = root / "pix2pix/scan" / records[0].img_id
    real = root / "real/scan" / f"{records[1].orig_id}__{records[1].sdir_id}"
    scan = np.zeros((16, 16, 16), dtype=np.uint16)
    scan[6:8, 6:8, 6:8] = 4000
    write_tiff_stack(fake, scan)
    write_tiff_stack(real, np.zeros_like(scan))

    def predict(_model, _payload, raw, _dirname, _cache_dir=None, **_kwargs):
        probability = np.where(raw > 1000, 0.9, 0.1).astype(np.float32)
        return probability, raw.astype(np.float32) / 4000

    monkeypatch.setattr(comparison, "predict_checkpoint", predict)
    summary, rows, calibrated = comparison.evaluate_dense(
        PointwiseSegmenter(), {"config": {}}, records, root, output=tmp_path / "out",
        validation_threshold=0.95)
    assert summary["n_volumes"] == 2 and summary["n_positive_volumes"] == 1
    assert summary["mean_positive_dice"] == summary["micro_dice"] == 1
    assert summary["dense_ap"] == 1 and summary["peak_hits"] == 1
    assert summary["real_false_positive_fraction"] == 0
    assert calibrated["mean_positive_dice"] == 0 and calibrated["peak_hits"] == 1
    assert rows[0]["tp"] == 8 and rows[0]["validation_fn"] == 8
    assert rows[1]["positive_voxels"] == 0 and rows[1]["ap"] is None
    assert len(list((tmp_path / "out/images").glob("*.png"))) == 2


def test_missing_trained_checkpoints_fail_before_creating_output(tmp_path):
    output = tmp_path / "diagnosis"
    with pytest.raises(FileNotFoundError, match="trained checkpoints"):
        comparison.run_comparison(tmp_path, output_dir=output, source_dir=tmp_path / "source",
                                  pilot_dir=tmp_path / "pilot", clip_dir=tmp_path / "clip")
    assert not output.exists()


def test_complete_comparison_binds_checkpoints_and_reuses_six_source_scans(tmp_path, monkeypatch):
    root, records = make_fake_corpus(tmp_path / "data", n_records=4, n_real=2,
                                     real_shared_dirs=2, scan_z=16, shape=(16, 16),
                                     cube_origin=(6, 6, 6), cube_size=(2, 2, 2))
    records[2:4] = [replace(record, ty="rem") for record in records[2:4]]
    manifest = {"manifest_id": "fixture", "seed": 21,
                "cohorts": {"overfit": [asdict(r) for r in records],
                            "pilot_train": [asdict(r) for r in records]}}
    source, pilot, clip = (tmp_path / name for name in ("source", "pilot", "clip"))
    source.mkdir()
    (source / "full_pool/probabilities/best").mkdir(parents=True)
    (pilot / "evaluation").mkdir(parents=True)
    (clip / "evaluation").mkdir(parents=True)
    (source / "manifest.json").write_text(json.dumps(manifest))
    config = {"stage": "overfit", "overfit_protocol": "full_pool",
              "model": {"name": "resnet18_medicalnet_seg"},
              "patches": {"patch_shape": [8, 8, 8]},
              "preprocessing": {"low": 1.0, "high": 99.0, "clip": False}}
    pool = FixedCropPool(SegmentationDataset(records, root, patch_shape=(8, 8, 8),
                                              seed=21, augment=False, fixed=True,
                                              preprocessing=config["preprocessing"], volume_cache_size=1))
    (source / "comparison.json").write_text(json.dumps({"complete": True, "passed": True,
        "manifest_id": "fixture", "crop_pool_sha256": pool.digest, "arms": {"full_pool": {}}}))
    with (source / "crop_pool.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pool.rows[0])
        writer.writeheader()
        writer.writerows(pool.rows)
    crop_summary, _, _ = comparison.evaluate_crops(PointwiseSegmenter(), pool)
    (source / "full_pool/gate.json").write_text(json.dumps({
        "passed": True, "manifest_id": "fixture",
        "positive_crop_dice": crop_summary["mean_positive_dice"],
        "negative_crop_fp_fraction": crop_summary["negative_crop_fp_fraction"]}))
    for name, folder in (("overfit", source / "full_pool"), ("pilot", pilot), ("pilot_clip", clip)):
        candidate = {**config, "stage": "pilot" if name != "overfit" else "overfit",
                     "preprocessing": {**config["preprocessing"], "clip": name == "pilot_clip"}}
        checkpoint = folder / "best.pt"
        torch.save({"task": "segmentation", "manifest": manifest, "config": candidate,
                    "model_state_dict": {}}, checkpoint)
        digest = file_digest(checkpoint)
        if name == "overfit":
            (folder / "probabilities/best/report.json").write_text(json.dumps({"checkpoint_sha256": digest}))
        else:
            (folder / "evaluation/metrics.json").write_text(json.dumps({"checkpoint_sha256": digest,
                                                                         "cohort_id": "pilot_fixture"}))
            (folder / "calibration.json").write_text(json.dumps({"source_split": "valid",
                "checkpoint_sha256": digest, "localization_threshold": 0.6, "detection_threshold": 0.5}))

    monkeypatch.setattr(comparison, "load_predictor", lambda path, device: (
        PointwiseSegmenter(), torch.load(path, map_location="cpu", weights_only=False)))
    monkeypatch.setattr(comparison, "save_preview", lambda *args: None)
    output = tmp_path / "comparison"
    result = comparison.run_comparison(root, output_dir=output, source_dir=source,
                                       pilot_dir=pilot, clip_dir=clip)
    assert result["complete"] and result["evaluation_only"]
    assert list(result["runs"]) == ["overfit", "pilot", "pilot_clip"]
    assert result["runs"]["overfit"]["crop_fixed_0_5"]["mean_positive_dice"] == crop_summary["mean_positive_dice"]
    assert result["runs"]["overfit"]["dense_fixed_0_5"]["n_volumes"] == 6
    assert result["runs"]["pilot"]["crop_validation_threshold"]["threshold"] == 0.6
    assert result["runs"]["pilot_clip"]["dense_validation_threshold"]["threshold"] == 0.6
    for name in result["runs"]:
        with (output / name / "crops.csv").open() as handle:
            assert len(list(csv.DictReader(handle))) == 48
        with (output / name / "volumes.csv").open() as handle:
            assert len(list(csv.DictReader(handle))) == 6
    assert json.loads((output / "report.json").read_text()) == result
