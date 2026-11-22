"""Matched crop exposures and objective weights for the grouped/mixed comparison."""
import copy
import csv
from dataclasses import asdict, replace
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from _tiff_fixtures import make_fake_corpus, make_fake_record
from test_segmentation import TinySegmenter
from tesi_m3d import compare_segmentation_batches as comparison
from tesi_m3d.segmentation import segmentation_loss_components
from tesi_m3d.segmentation_data import VolumeOrder
from tesi_m3d.train import load_yaml_config
from tesi_m3d.train_segmentation import train_volume


class FakeDataset:
    def __init__(self):
        self.records = [replace(make_fake_record(img_id=f"case_{i}"), ty="inj" if i < 2 else "rem") for i in range(4)]
        self.records += [replace(make_fake_record(img_id=f"real_{i}"), mod="real") for i in range(2)]

    def __getitem__(self, key):
        index = key[1]
        mask = torch.zeros(8, 1, 8, 8, 8)
        if not self.records[index].is_real:
            mask[:4, :, 2:4, 2:4, 2:4] = 1
        return {"image": mask.clone(), "mask": mask, "img_id": self.records[index].img_id}


def test_schedules_are_deterministic_mixed_and_visit_all_crops_once():
    pool = comparison.FixedCropPool(FakeDataset())
    for epoch in (0, 1, 49):
        grouped = comparison.epoch_batches(pool, 21, epoch, mixed=False)
        order = VolumeOrder(6, 21)
        order.epoch = epoch
        assert list(grouped[:, 0] // 8) == [i for _, i in order]
        batches = comparison.epoch_batches(pool, 21, epoch, mixed=True)
        np.testing.assert_array_equal(batches, comparison.epoch_batches(pool, 21, epoch, mixed=True))
        np.testing.assert_array_equal(np.sort(batches.flatten()), np.arange(48))
        for indices in batches:
            assert len(set(indices // 8)) >= 3
            assert pool.positive[indices].sum() in (2, 3)
            assert pool.real[indices].sum() >= 2
    assert not np.array_equal(comparison.epoch_batches(pool, 21, 0, mixed=True),
                              comparison.epoch_batches(pool, 21, 1, mixed=True))


def test_pool_rejects_dirty_negatives_and_hashes_inputs():
    first = comparison.FixedCropPool(FakeDataset())
    second = comparison.FixedCropPool(FakeDataset())
    assert first.digest == second.digest
    assert len(first.rows) == 48

    class Dirty(FakeDataset):
        def __getitem__(self, key):
            batch = super().__getitem__(key)
            batch["mask"][4, 0, 0, 0, 0] = 1
            return batch

    with pytest.raises(ValueError, match="four positive"):
        comparison.FixedCropPool(Dirty())


def test_fixed_normalizer_preserves_whole_epoch_loss_and_gradients():
    pool = comparison.FixedCropPool(FakeDataset())
    reference = torch.nn.Conv3d(1, 1, 1)
    reference.weight.data.zero_()
    reference.bias.data.zero_()
    results = []
    for mixed, denominator in [(False, 4), (True, 4), (True, None)]:
        model = copy.deepcopy(reference)
        total_bce, total_dice = 0., 0.
        for indices in comparison.epoch_batches(pool, 21, 0, mixed=mixed):
            bce, dice = segmentation_loss_components(model(pool.image[indices]), pool.mask[indices],
                                                      foreground_normalizer=denominator)
            (0.5 * bce + 0.5 * dice).backward()
            total_bce += float(bce.detach())
            total_dice += float(dice.detach())
        results.append((total_bce, total_dice, torch.cat([p.grad.flatten() for p in model.parameters()])))
    assert results[0][:2] == pytest.approx(results[1][:2], rel=1e-6)
    torch.testing.assert_close(results[0][2], results[1][2], rtol=1e-5, atol=1e-6)
    # A naive per-batch foreground mean raises total Dice weight from four to six batches.
    assert results[2][1] == pytest.approx(1.5 * results[0][1], rel=1e-6)


@pytest.mark.parametrize("microbatch,accumulation", [(1, 8), (2, 4)])
def test_mixed_accumulation_keeps_fixed_foreground_weight(microbatch, accumulation):
    torch.manual_seed(21)
    pool = comparison.FixedCropPool(FakeDataset())
    indices = comparison.epoch_batches(pool, 21, 0, mixed=True)[0]
    model = TinySegmenter()
    reference = copy.deepcopy(model)
    batch = {"image": pool.image[indices], "mask": pool.mask[indices]}
    measured = train_volume(model, batch, torch.optim.SGD(model.parameters(), lr=.01), microbatch=microbatch,
                            accumulation=accumulation, foreground_normalizer=4)
    bce, dice = segmentation_loss_components(reference(batch["image"]), batch["mask"], foreground_normalizer=4)
    loss = .5 * bce + .5 * dice
    loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1)
    torch.optim.SGD(reference.parameters(), lr=.01).step()
    assert measured == pytest.approx(float(loss.detach()), rel=1e-6)
    for first, second in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(first, second)
    with pytest.raises(ValueError, match="foreground_normalizer"):
        train_volume(model, batch, torch.optim.SGD(model.parameters(), lr=.01), foreground_normalizer=float("nan"))


def test_paired_run_with_tiffs_preserves_inputs_initialization_and_exposures(tmp_path, monkeypatch):
    root, records = make_fake_corpus(tmp_path / "data", n_records=4, n_real=2, real_shared_dirs=2,
                                     scan_z=16, shape=(16, 16), cube_origin=(6, 6, 6), cube_size=(2, 2, 2))
    records = [replace(r, ty="rem") if i in (2, 3) else r for i, r in enumerate(records)]
    manifest = {"manifest_id": "fixture", "seed": 21, "cohorts": {"overfit": [asdict(r) for r in records]}}
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    (protocol / "manifest.json").write_text(json.dumps(manifest))
    (protocol / "audit.json").write_text(json.dumps({"manifest_id": "fixture", "passed": True,
                                                     "cohorts": ["overfit"], "data_root": str(root.resolve())}))
    config = load_yaml_config("configs/segmentation_overfit.yaml")
    config.update(manifest=str(protocol / "manifest.json"), cache_dir=str(tmp_path / "cache"))
    config["patches"]["patch_shape"] = [8, 8, 8]
    config["training"].update(epochs=2, max_updates=12)
    original_gate = tmp_path / "old_gate.json"
    original_gate.write_text('{"passed":false}')
    config["overfit_report"] = str(original_gate)
    monkeypatch.setattr(comparison, "build_segmenter", lambda config: TinySegmenter())
    output = tmp_path / "comparison"
    result = comparison.run_comparison(config, root, output_dir=output)
    first, second = result["arms"]["grouped"], result["arms"]["mixed"]
    assert first["initial_state_sha256"] == second["initial_state_sha256"]
    assert first["crop_pool_sha256"] == second["crop_pool_sha256"]
    for name in ("grouped", "mixed"):
        arm = result["arms"][name]
        assert arm["updates"] == 12 and arm["exposures_per_crop"] == 2
        exposures = list(csv.DictReader((output / name / "exposures.csv").open()))
        assert len(exposures) == 48 and {r["exposures"] for r in exposures} == {"2"}
        assert (output / name / "best.pt").is_file() and (output / name / "last.pt").is_file()
        assert len(list(csv.DictReader((output / name / "crops.csv").open()))) == 48
    assert json.loads((output / "comparison.json").read_text()) == result
    assert original_gate.read_text() == '{"passed":false}'
    with pytest.raises(ValueError, match="already contains"):
        comparison.run_comparison(config, root, output_dir=output)


def test_arm_finishes_equal_budget_even_when_gate_passes_early(tmp_path, monkeypatch):
    pool = comparison.FixedCropPool(FakeDataset())
    monkeypatch.setattr(comparison, "build_segmenter", lambda config: TinySegmenter())
    monkeypatch.setattr(comparison, "overfit_metrics", lambda *a, **kw: {
        "positive_crop_dice": .95, "negative_crop_fp_fraction": .001, "passed": True})
    config = load_yaml_config("configs/segmentation_overfit.yaml")
    config["training"].update(epochs=5, max_updates=30)
    report = comparison.run_arm(config, {"manifest_id": "fixture"}, pool, tmp_path, mixed=True, device="cpu")
    assert report["first_gate_passed_update"] == 25
    assert report["updates"] == 30 and report["exposures_per_crop"] == 5


def test_comparison_cli_returns_failed_scientific_gate(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(comparison, "run_comparison", lambda *a, **kw: {"passed": False})
    monkeypatch.setattr(sys, "argv", ["compare_segmentation_batches", "--data-root", str(tmp_path)])
    with pytest.raises(SystemExit) as stopped:
        comparison.main()
    assert stopped.value.code == 2


def test_comparison_rejects_different_initial_states_and_reports_failure(tmp_path, monkeypatch):
    dataset = FakeDataset()
    manifest = {"manifest_id": "fixture", "seed": 21,
                "cohorts": {"overfit": [asdict(r) for r in dataset.records]}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    config = load_yaml_config("configs/segmentation_overfit.yaml")
    config["manifest"] = str(path)
    monkeypatch.setattr(comparison, "check_prerequisites", lambda *a: None)
    monkeypatch.setattr(comparison, "SegmentationDataset", lambda *a, **kw: dataset)
    monkeypatch.setattr(comparison, "run_arm", lambda cfg, manifest, pool, output, **kw: {
        "initial_state_sha256": output.name})
    output = tmp_path / "comparison"
    with pytest.raises(ValueError, match="identical model states"):
        comparison.run_comparison(config, tmp_path, output_dir=output)
    report = json.loads((output / "comparison.json").read_text())
    assert not report["passed"] and "identical model states" in report["error"]
    assert report["active_arm"] == "mixed"
    config["training"]["max_updates"] = 299
    with pytest.raises(ValueError, match="max_updates = 6"):
        comparison.run_comparison(config, tmp_path, output_dir=tmp_path / "bad_budget")
