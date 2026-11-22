"""Bound hard-crop diagnosis to the prior experiment and preserve scientific gates."""
import csv
from dataclasses import asdict, replace
import hashlib
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from _tiff_fixtures import make_fake_corpus
from test_segmentation import TinySegmenter
from test_debug_segmentation import fixed_batch
from tesi_m3d import debug_segmentation as debug
from tesi_m3d import debug_segmentation_hard_crops as hard
from tesi_m3d.compare_segmentation_batches import FixedCropPool, tensor_digest
from tesi_m3d.diagnose_localization import write_csv, write_json
from tesi_m3d.segmentation_data import SegmentationDataset
from tesi_m3d.train import load_yaml_config, set_global_seed


def setup_reference(tmp_path):
    root, records = make_fake_corpus(tmp_path / "data", n_records=4, n_real=2, real_shared_dirs=2,
                                     scan_z=16, shape=(16, 16), cube_origin=(6, 6, 6), cube_size=(2, 2, 2))
    for index, img_id in ((0, "inj_2092"), (2, "rem_133")):
        for folder in ("scan", "label"):
            (root / "pix2pix" / folder / records[index].img_id).rename(root / "pix2pix" / folder / img_id)
        records[index] = replace(records[index], img_id=img_id, ty="inj" if index == 0 else "rem")
    records[3] = replace(records[3], ty="rem")
    manifest = {"manifest_id": "fixture", "seed": 21, "cohorts": {"overfit": [asdict(r) for r in records]}}
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    write_json(protocol / "manifest.json", manifest)
    write_json(protocol / "audit.json", {"manifest_id": "fixture", "passed": True,
                                         "cohorts": ["overfit"], "data_root": str(root.resolve())})
    config = load_yaml_config("configs/segmentation_overfit.yaml")
    config.update(manifest=str(protocol / "manifest.json"), cache_dir=str(tmp_path / "cache"))
    config["patches"]["patch_shape"] = [8, 8, 8]
    config["training"].update(epochs=1, max_updates=2)
    pool = FixedCropPool(SegmentationDataset(records, root, patch_shape=(8, 8, 8), seed=21,
                                              fixed=True, augment=False, preprocessing=config["preprocessing"]))
    reference = tmp_path / "reference"
    reference.mkdir()
    write_json(reference / "effective_config.json", config)
    write_json(reference / "manifest.json", manifest)
    write_csv(reference / "crop_pool.csv", pool.rows)
    set_global_seed(21)
    digest = hashlib.sha256()
    for name, tensor in TinySegmenter().state_dict().items():
        digest.update(name.encode())
        digest.update(tensor_digest(tensor).encode())
    arm = {"initial_state_sha256": digest.hexdigest(), "crop_pool_sha256": pool.digest}
    comparison = {"experiment": "batch_composition_comparison", "manifest_id": "fixture", "active_arm": None,
                  "crop_pool_sha256": pool.digest, "arms": {"grouped": arm, "mixed": arm}}
    write_json(reference / "comparison.json", comparison)
    return config, root, reference, pool


def test_spatial_shells_partition_volume_and_count_known_errors():
    target = torch.zeros(2, 1, 40, 40, 40, dtype=torch.bool)
    target[0, 0, 0, 10, 10] = True
    probability = target.float()
    probability[0, 0, 0, 10, 10] = 0  # one false negative on a face
    probability[0, 0, 6, 10, 10] = .5  # one false positive in shell 4..7
    probability[1, 0, 20, 20, 20] = 1  # one false positive in the interior
    rows = debug.spatial_errors(target, probability)
    for crop in ("positive", "negative"):
        selected = [r for r in rows if r["crop"] == crop]
        total, *shells = selected
        for key in ("region_voxels", "target_voxels", "predicted_voxels", "true_positive_voxels",
                    "false_positive_voxels", "false_negative_voxels"):
            assert total[key] == sum(r[key] for r in shells)
    by_key = {(r["crop"], r["region"]): r for r in rows}
    assert by_key["positive", "border_0_3"]["false_negative_voxels"] == 1
    assert by_key["positive", "border_4_7"]["false_positive_voxels"] == 1
    assert by_key["negative", "interior_16_plus"]["false_positive_voxels"] == 1
    assert by_key["negative", "interior_16_plus"]["recall"] is None
    assert by_key["negative", "interior_16_plus"]["background_fp_fraction"] == pytest.approx(1 / 512)


def test_hard_crops_reuses_exact_tiff_crops_and_runs_all_four_stages(tmp_path, monkeypatch):
    config, root, reference, pool = setup_reference(tmp_path)
    monkeypatch.setattr(debug, "build_segmenter", lambda config: TinySegmenter())
    gate = tmp_path / "gate.json"
    gate.write_text('{"passed":false}')
    # Keep the reference configuration identical while binding an existing failed gate.
    config["overfit_report"] = str(gate)
    write_json(reference / "effective_config.json", config)
    output = tmp_path / "hard"
    result = hard.run_hard_crops(config, root, reference_dir=reference, output_dir=output)
    assert result["active_stage"] is None and "error" not in result
    assert len(result["stages"]) == 4
    for key, stage in result["stages"].items():
        assert stage["last"]["updates"] == 2 and stage["update_check"]["passed"]
        assert stage["last"]["initial_state_sha256"] == result["initial_state_sha256"]
        assert (output / key / "images/spatial_errors.csv").is_file()
        assert (output / key / "initial_images/spatial_errors.csv").is_file()
        assert len(list(csv.DictReader((output / key / "updates.csv").open()))) == 2
        payload = torch.load(output / key / "last.pt", weights_only=False)
        assert payload["task"] == "segmentation_debug"  # deliberately cannot unlock normal training
        assert payload["fixed_batch"]["mask_sha256"]
    selection = json.loads((output / "selection.json").read_text())["crops"]
    assert [r["pool_index"] for r in selection] == [3, 4, 18, 20]
    for row in selection:
        assert row["image_sha256"] == pool.rows[row["pool_index"]]["image_sha256"]
        assert row["mask_sha256"] == pool.rows[row["pool_index"]]["mask_sha256"]
    assert gate.read_text() == '{"passed":false}'
    assert json.loads((output / "report.json").read_text()) == result
    with pytest.raises(ValueError, match="already contains"):
        hard.run_hard_crops(config, root, reference_dir=reference, output_dir=output)


@pytest.mark.parametrize("change", ["image", "target", "ledger", "config", "manifest", "incomplete"])
def test_changed_inputs_or_protocol_are_rejected_before_training(tmp_path, monkeypatch, change):
    config, root, reference, _ = setup_reference(tmp_path)
    monkeypatch.setattr(hard, "run_fixed_stage", lambda *a, **kw: pytest.fail("must reject before training"))
    if change in ("image", "target"):
        from PIL import Image
        import numpy as np
        folder = "scan" if change == "image" else "label"
        for path in (root / "pix2pix" / folder / "rem_133").glob("*.tiff"):
            with Image.open(path) as image:
                array = np.array(image)
            if change == "image":
                array[:] = 0
            else:
                array[:] = False
            Image.fromarray(array).save(path)
    elif change == "ledger":
        path = reference / "crop_pool.csv"
        path.write_text(path.read_text().replace("inj_2092", "changed"))
    elif change == "config":
        config["preprocessing"]["clip"] = True
    elif change == "manifest":
        path = reference / "manifest.json"
        value = json.loads(path.read_text())
        value["seed"] = 42
        write_json(path, value)
    else:
        path = reference / "comparison.json"
        value = json.loads(path.read_text())
        value["active_arm"] = "mixed"
        write_json(path, value)
    with pytest.raises(ValueError):
        hard.run_hard_crops(config, root, reference_dir=reference, output_dir=tmp_path / "rejected")


def test_complete_budget_preserves_first_success_and_intermediate_best(tmp_path, monkeypatch):
    monkeypatch.setattr(debug, "build_segmenter", lambda config: TinySegmenter())
    calls = 0

    def measure(*args, **kwargs):
        nonlocal calls
        calls += 1
        dice = .95 if calls == 2 else .7
        return {"positive_crop_dice": dice, "negative_crop_fp_fraction": .001, "passed": dice >= .9}

    monkeypatch.setattr(debug, "measure_fixed", measure)
    config = load_yaml_config("configs/segmentation_overfit.yaml")
    config["training"]["max_updates"] = 3
    result, _ = debug.run_fixed_stage(config, {}, fixed_batch(), tmp_path, complete_budget=True)
    assert result["updates"] == 3 and result["first_gate_passed_update"] == 1
    assert not result["passed"]
    best = json.loads((tmp_path / "best_metrics.json").read_text())
    assert best["updates"] == 1 and best["passed"]
    assert len(list(csv.DictReader((tmp_path / "updates.csv").open()))) == 3


def test_wrong_initial_weights_are_rejected_before_any_update(tmp_path, monkeypatch):
    monkeypatch.setattr(debug, "build_segmenter", lambda config: TinySegmenter())
    monkeypatch.setattr(debug, "train_volume", lambda *a, **kw: pytest.fail("must not train"))
    with pytest.raises(ValueError, match="reference model state"):
        debug.run_fixed_stage(load_yaml_config("configs/segmentation_overfit.yaml"), {}, fixed_batch(), tmp_path,
                              expected_initial_state_sha256="changed")


def test_technical_failure_is_reported_and_cli_distinguishes_failed_gate(tmp_path, monkeypatch):
    config, root, reference, _ = setup_reference(tmp_path)

    def fail(*args, **kwargs):
        raise FloatingPointError("non-finite gradient")

    monkeypatch.setattr(hard, "run_fixed_stage", fail)
    output = tmp_path / "failed"
    with pytest.raises(FloatingPointError):
        hard.run_hard_crops(config, root, reference_dir=reference, output_dir=output)
    report = json.loads((output / "report.json").read_text())
    assert report["active_stage"] == "inj_2092/positive" and not report["passed"]
    assert "non-finite gradient" in report["error"]
    import sys
    monkeypatch.setattr(hard, "run_hard_crops", lambda *a, **kw: {"passed": False})
    monkeypatch.setattr(sys, "argv", ["debug_segmentation_hard_crops", "--data-root", str(root)])
    with pytest.raises(SystemExit) as stopped:
        hard.main()
    assert stopped.value.code == 2
