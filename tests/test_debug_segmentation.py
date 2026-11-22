"""Diagnostic stage gates, real optimizer updates and preservation of best weights."""
from dataclasses import asdict, replace
import csv
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from _tiff_fixtures import make_fake_record
from test_segmentation import TinySegmenter
from tesi_m3d import debug_segmentation as debug
from tesi_m3d import train_segmentation as trainer
from tesi_m3d.train import load_yaml_config


def fixed_batch(count=2):
    mask = torch.zeros(count, 1, 8, 8, 8)
    mask[0, :, 2:6, 2:6, 2:6] = 1
    return {"image": mask.clone(), "mask": mask}


def test_update_diagnostics_match_loss_and_detect_both_parameter_groups():
    torch.manual_seed(21)
    model = TinySegmenter()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    report = {}
    loss = trainer.train_volume(model, fixed_batch(), optimizer, microbatch=1, accumulation=2, diagnostics=report)
    assert loss == pytest.approx(0.5 * report["bce"] + 0.5 * report["soft_dice_loss"])
    assert all(report[f"{group}_{kind}_norm"] > 0 for group in ("encoder", "decoder") for kind in ("grad", "update"))
    with pytest.raises(ValueError, match="crop count"):
        trainer.train_volume(model, fixed_batch(), optimizer, microbatch=2, accumulation=4)


def test_fixed_positive_and_pair_can_memorize_with_fresh_initializations(tmp_path, monkeypatch):
    initial = []

    def build(config):
        model = TinySegmenter()
        initial.append(model.encoder.weight.detach().clone())
        return model

    monkeypatch.setattr(debug, "build_segmenter", build)
    config = load_yaml_config("configs/segmentation_overfit.yaml")
    config["training"].update(max_updates=100, encoder_lr=0.1, decoder_lr=0.1)
    for count in (1, 2):
        path = tmp_path / str(count)
        report, checked = debug.run_fixed_stage(config, {"manifest_id": "fixture"}, fixed_batch(count), path)
        assert checked["passed"] and report["passed"]
        assert report["positive_crop_dice"] >= 0.9
        if count == 1:
            assert report["negative_crop_fp_fraction"] is None
        else:
            assert report["negative_crop_fp_fraction"] < .01
        assert (path / "best.pt").is_file() and (path / "last.pt").is_file()
        assert (path / "images/positive.png").is_file()
        payload = torch.load(path / "last.pt", weights_only=False)
        assert payload["task"] == "segmentation_debug"
        assert "encoder_grad_norm" in (path / "updates.csv").read_text()
    torch.testing.assert_close(initial[0], initial[1])


@pytest.mark.parametrize("failure", ["zero_lr", "detached_encoder"])
def test_update_gate_stops_when_weights_or_gradients_are_missing(tmp_path, monkeypatch, failure):
    class DetachedEncoder(TinySegmenter):
        def forward(self, image):
            return self.output(self.encoder(image).detach())

    monkeypatch.setattr(debug, "build_segmenter", lambda config: TinySegmenter() if failure == "zero_lr" else DetachedEncoder())
    monkeypatch.setattr(debug, "make_optimizer", lambda model, training: torch.optim.SGD(model.parameters(), lr=0 if failure == "zero_lr" else .1))
    config = load_yaml_config("configs/segmentation_overfit.yaml")
    report, checked = debug.run_fixed_stage(config, {}, fixed_batch(), tmp_path)
    assert not checked["passed"] and not report["passed"]
    assert report["updates"] == 1
    assert checked["encoder_update_norm"] == 0
    assert (checked["encoder_grad_norm"] > 0) == (failure == "zero_lr")


def setup_workflow(tmp_path, monkeypatch):
    records = [make_fake_record(img_id=f"inj_{i}") for i in range(4)]
    records += [replace(make_fake_record(img_id=f"rem_{i}"), mod="real") for i in range(2)]
    manifest = {"manifest_id": "fixture", "seed": 21, "cohorts": {"overfit": [asdict(r) for r in records]}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    config = load_yaml_config("configs/segmentation_overfit.yaml")
    config["manifest"] = str(path)

    class Dataset:
        def __init__(self, records, *args, **kwargs):
            self.records = records

        def __len__(self):
            return len(self.records)

        def __getitem__(self, key):
            batch = fixed_batch(8)
            batch["img_id"] = self.records[key[1]].img_id
            return batch

    monkeypatch.setattr(debug, "SegmentationDataset", Dataset)
    monkeypatch.setattr(debug, "check_prerequisites", lambda *a: None)
    return config, manifest, Dataset


@pytest.mark.parametrize("failure", ["update", "positive", "positive_negative", None])
def test_workflow_stops_before_dependent_stages_and_preserves_original_gate(tmp_path, monkeypatch, failure):
    config, manifest, _ = setup_workflow(tmp_path, monkeypatch)
    original_gate = tmp_path / "gate.json"
    original_gate.write_text('{"passed":false}')
    config["overfit_report"] = str(original_gate)
    stages = []

    def stage(config, manifest, batch, path, **kwargs):
        stages.append(path.name)
        assert len(batch["image"]) == (1 if path.name == "positive" else 2)
        return {"passed": failure != path.name}, {"passed": failure != "update"}

    monkeypatch.setattr(debug, "run_fixed_stage", stage)
    monkeypatch.setattr(debug, "run_training", lambda *a, **kw: stages.append("six_scans") or {"passed": True})
    output = tmp_path / "debug"
    report = debug.run_debug(config, tmp_path, output_dir=output)
    assert report["passed"] == (failure is None)
    expected = ["positive"]
    if failure not in ("update", "positive"):
        expected += ["positive_negative"]
        if failure is None:
            expected += ["six_scans"]
    assert stages == expected
    assert json.loads((output / "selection.json").read_text())["crop_indices"] == [0, 4]
    assert json.loads((output / "report.json").read_text()) == report
    assert original_gate.read_text() == '{"passed":false}'
    with pytest.raises(ValueError, match="already contains"):
        debug.run_debug(config, tmp_path, output_dir=output)


def test_six_scan_overfit_preserves_intermediate_best_and_logs_components(tmp_path, monkeypatch):
    config, manifest, dataset = setup_workflow(tmp_path, monkeypatch)
    config["training"].update(epochs=9, max_updates=50, num_workers=0)
    monkeypatch.setattr(trainer, "SegmentationDataset", dataset)
    monkeypatch.setattr(trainer, "check_prerequisites", lambda *a: None)
    monkeypatch.setattr(trainer, "build_segmenter", lambda *a, **kw: TinySegmenter())
    values = iter([.8, .3, .3])

    def measure(*args, case_rows=None, **kwargs):
        if case_rows is not None:
            case_rows.append({"img_id": "inj_0", "positive_crop_dice": .8})
        return {"passed": False, "positive_crop_dice": next(values), "negative_crop_fp_fraction": .02}

    monkeypatch.setattr(trainer, "overfit_metrics", measure)
    output = tmp_path / "run"
    result = trainer.run_training(config, tmp_path, output_dir=output, debug_updates=True)
    best = torch.load(output / "best.pt", weights_only=False)
    last = torch.load(output / "last.pt", weights_only=False)
    assert best["updates"] == 25 and last["updates"] == 50
    assert best["best_overfit_metrics"]["positive_crop_dice"] == .8
    assert result["positive_crop_dice"] == .3
    assert json.loads((output / "best_gate.json").read_text())["updates"] == 25
    assert len(list(csv.DictReader((output / "update_metrics.csv").open()))) == 50
    assert len(list(csv.DictReader((output / "case_history.csv").open()))) == 2
    with pytest.raises(ValueError, match="already reached max_updates"):
        trainer.run_training(config, tmp_path, output_dir=output, resume=output / "last.pt")
    with pytest.raises(ValueError, match="resume requires last.pt"):
        trainer.run_training(config, tmp_path, output_dir=output, resume=output / "best.pt")


def test_workflow_writes_technical_failure_report_without_advancing(tmp_path, monkeypatch):
    config, _, _ = setup_workflow(tmp_path, monkeypatch)

    def fail(*args, **kwargs):
        raise FloatingPointError("non-finite gradient")

    monkeypatch.setattr(debug, "run_fixed_stage", fail)
    monkeypatch.setattr(debug, "run_training", lambda *a, **kw: pytest.fail("must not run six scans"))
    with pytest.raises(FloatingPointError):
        debug.run_debug(config, tmp_path, output_dir=tmp_path / "debug")
    report = json.loads((tmp_path / "debug/report.json").read_text())
    assert not report["passed"] and report["stopped_at"] == "positive"
    assert "non-finite gradient" in report["error"]


def test_debug_cli_reports_failed_gate_without_starting_other_work(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(debug, "run_debug", lambda *a, **kw: {"passed": False, "stopped_at": "positive"})
    monkeypatch.setattr(sys, "argv", ["debug_segmentation", "--data-root", str(tmp_path)])
    with pytest.raises(SystemExit) as stopped:
        debug.main()
    assert stopped.value.code == 2
