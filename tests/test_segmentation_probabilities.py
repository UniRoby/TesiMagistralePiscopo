"""Full-crop probability statistics, offline checkpoint restore and read-only diagnosis."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from test_segmentation import TinySegmenter
from test_segmentation_hard_crops import setup_reference
from tesi_m3d import debug_segmentation as debug
from tesi_m3d import debug_segmentation_hard_crops as hard
from tesi_m3d import diagnose_segmentation_probabilities as diagnosis
from tesi_m3d.diagnose_localization import write_json
from tesi_m3d.segmentation import file_digest


def test_diffuse_background_mass_explains_low_soft_dice_despite_perfect_hard_dice():
    probability = np.full((40, 40, 40), .125, dtype=np.float32)
    target = np.zeros_like(probability, dtype=bool)
    target[0, 0, 0] = True
    probability[target] = .99
    summary, rows = diagnosis.probability_statistics(probability, target)
    assert summary["hard_dice"] == 1
    mass_fg, mass_bg = float(probability[target][0]), .125 * (40**3 - 1)
    assert summary["probability_mass_inside"] == mass_fg
    assert summary["probability_mass_outside"] == mass_bg
    assert summary["soft_dice_float64"] == pytest.approx((2 * mass_fg + 1e-5) / (mass_fg + mass_bg + 1 + 1e-5))
    assert summary["soft_dice_float64"] < .001
    assert summary["background_mass_fraction"] > .999
    assert summary["mean_probability_outside"] == .125
    bg = next(r for r in rows if r["region"] == "all" and r["compartment"] == "background")
    assert bg["n_voxels"] == 40**3 - 1
    assert bg["p01"] == bg["p50"] == bg["p99"] == .125
    assert bg["fraction_ge_0.1"] == 1 and bg["fraction_ge_0.5"] == 0
    for compartment in ("all", "foreground", "background"):
        selected = [r for r in rows if r["compartment"] == compartment]
        total, *shells = selected
        assert total["n_voxels"] == sum(r["n_voxels"] for r in shells)
        assert total["probability_mass"] == pytest.approx(sum(r["probability_mass"] for r in shells))


@pytest.mark.parametrize("foreground", [False, True])
def test_empty_compartments_are_undefined_and_quantiles_use_every_voxel(foreground):
    probability = np.arange(8, dtype=np.float32).reshape(2, 2, 2) / 8
    target = np.full((2, 2, 2), foreground)
    summary, rows = diagnosis.probability_statistics(probability, target)
    allrow = next(r for r in rows if r["region"] == "all" and r["compartment"] == "all")
    assert allrow["p50"] == pytest.approx(np.quantile(probability, .5))
    empty = next(r for r in rows if r["region"] == "all" and r["compartment"] == ("background" if foreground else "foreground"))
    assert empty["n_voxels"] == 0 and empty["probability_mass"] == 0
    assert empty["mean"] is empty["p50"] is empty["fraction_ge_0.5"] is None
    if not foreground:
        assert summary["soft_dice_float64"] is summary["hard_dice"] is summary["mean_probability_inside"] is None
    else:
        assert summary["mean_probability_outside"] is None


@pytest.mark.parametrize("change", ["nan", "negative", "above_one", "mask", "shape"])
def test_statistics_reject_invalid_probabilities_and_masks(change):
    p = np.zeros((2, 2, 2)); mask = p.copy()
    if change == "nan": p[0, 0, 0] = np.nan
    elif change == "negative": p[0, 0, 0] = -.1
    elif change == "above_one": p[0, 0, 0] = 1.1
    elif change == "mask": mask[0, 0, 0] = .1
    else: mask = mask[:1]
    with pytest.raises(ValueError, match="matching 3D"):
        diagnosis.probability_statistics(p, mask)


def make_run(tmp_path, monkeypatch):
    config, root, reference, _ = setup_reference(tmp_path)
    config["model"]["medicalnet_weights"] = str(tmp_path / "missing_medicalnet.pth")
    write_json(reference / "effective_config.json", config)
    monkeypatch.setattr(debug, "build_segmenter", lambda config: TinySegmenter())
    source = tmp_path / "hard"
    hard.run_hard_crops(config, root, reference_dir=reference, output_dir=source)
    models = []

    def restore(config, *, state_dict):
        assert not Path(config["medicalnet_weights"]).exists()
        model = TinySegmenter()
        model.load_state_dict(state_dict, strict=True)
        models.append(model)
        return model

    monkeypatch.setattr(diagnosis, "build_segmenter", restore)
    return root, reference, source, models


def test_all_eight_checkpoint_measurements_preserve_sources_and_restore_without_pretrained_file(tmp_path, monkeypatch):
    root, reference, source, models = make_run(tmp_path, monkeypatch)
    before = {p: file_digest(p) for p in source.rglob("*") if p.is_file()}
    monkeypatch.setattr(debug, "make_optimizer", lambda *a: pytest.fail("diagnosis must never train"))
    monkeypatch.setattr(debug, "train_volume", lambda *a, **kw: pytest.fail("diagnosis must never train"))
    torch.manual_seed(999)  # Restoring a full state must not depend on the original seed/runtime initialization.
    output = tmp_path / "probabilities"
    result = diagnosis.run_diagnosis(root, run_dir=source, reference_dir=reference, output_dir=output)
    assert result["complete"] and result["active_checkpoint"] is None and "passed" not in result
    assert result["all_reproduced_within_1e_4"] and len(result["checkpoints"]) == 8
    assert before == {p: file_digest(p) for p in source.rglob("*") if p.is_file()}
    assert len(models) == 8 and all(not m.training for m in models)
    assert all(p.grad is None for m in models for p in m.parameters())
    for key, report in result["checkpoints"].items():
        count = 2 if "positive_negative" in key else 1
        with np.load(output / key / "probabilities.npz") as arrays:
            assert arrays["probability"].shape == (count, 1, 8, 8, 8)
            assert arrays["probability"].dtype == np.float32 and arrays["mask"].dtype == np.uint8
            assert np.isfinite(arrays["probability"]).all()
        assert report["checkpoint_sha256"] == file_digest(source / (key + ".pt"))
        assert len(report["crops"]) == count
        assert len(list(csv.DictReader((output / key / "distributions.csv").open()))) == count * 15
        assert (output / key / "spatial_errors.csv").is_file()
        assert (output / key / "positive.png").is_file()
    assert json.loads((output / "report.json").read_text()) == result
    with pytest.raises(ValueError, match="already contains"):
        diagnosis.run_diagnosis(root, run_dir=source, reference_dir=reference, output_dir=output)
    with pytest.raises(ValueError, match="outside the source"):
        diagnosis.run_diagnosis(root, run_dir=source, reference_dir=reference, output_dir=source / "child")


@pytest.mark.parametrize("change", ["task", "initial_state", "fixed_batch", "updates", "state_dict", "config", "training_loss"])
def test_wrong_checkpoint_binding_is_reported_without_completing(tmp_path, monkeypatch, change):
    root, reference, source, _ = make_run(tmp_path, monkeypatch)
    path = source / "inj_2092/positive/best.pt"
    payload = torch.load(path, weights_only=False)
    if change == "task": payload["task"] = "segmentation"
    elif change == "initial_state": payload["initial_state_sha256"] = "changed"
    elif change == "fixed_batch": payload["fixed_batch"]["mask_sha256"] = "changed"
    elif change == "updates": payload["updates"] += 1
    elif change == "state_dict": payload["model_state_dict"].pop("output.weight")
    elif change == "training_loss": payload["training_loss"]["squared_pred"] = True
    else: payload["config"]["preprocessing"]["clip"] = True
    torch.save(payload, path)
    output = tmp_path / "rejected"
    with pytest.raises((ValueError, RuntimeError)):
        diagnosis.run_diagnosis(root, run_dir=source, reference_dir=reference, output_dir=output)
    report = json.loads((output / "report.json").read_text())
    assert not report["complete"] and report["active_checkpoint"] == "inj_2092/positive/best"
    assert "error" in report


def test_missing_checkpoints_fail_before_loading_ct_and_prediction_drift_is_visible(tmp_path, monkeypatch):
    root, reference, source, _ = make_run(tmp_path, monkeypatch)
    path = source / "inj_2092/positive/best.pt"
    original = path.read_bytes()
    path.unlink()
    output = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="laboratory PC"):
        diagnosis.run_diagnosis(root, run_dir=source, reference_dir=reference, output_dir=output)
    assert not output.exists()
    path.write_bytes(original)
    payload = torch.load(path, weights_only=False)
    payload["model_state_dict"]["output.bias"].fill_(9.)
    torch.save(payload, path)
    result = diagnosis.run_diagnosis(root, run_dir=source, reference_dir=reference, output_dir=output)
    assert result["complete"] and not result["all_reproduced_within_1e_4"]
    assert not result["checkpoints"]["inj_2092/positive/best"]["reproduced_within_1e_4"]


def test_legacy_checkpoints_without_training_loss_metadata_still_restore(tmp_path, monkeypatch):
    root, reference, source, _ = make_run(tmp_path, monkeypatch)
    saved = json.loads((source / "report.json").read_text())
    for stage in saved["stages"].values():
        for name in ("best", "last"):
            stage[name].pop("training_loss", None)
    write_json(source / "report.json", saved)
    for path in source.rglob("*.pt"):
        payload = torch.load(path, weights_only=False)
        payload.pop("training_loss", None)
        payload["metrics"].pop("training_loss", None)
        torch.save(payload, path)
    result = diagnosis.run_diagnosis(root, run_dir=source, reference_dir=reference, output_dir=tmp_path / "legacy")
    assert result["complete"] and result["all_reproduced_within_1e_4"]
    assert all(r["training_loss"] is None for r in result["checkpoints"].values())


def test_nonfinite_logits_are_rejected(tmp_path, monkeypatch):
    root, reference, source, _ = make_run(tmp_path, monkeypatch)

    class Nonfinite(TinySegmenter):
        def forward(self, image):
            return image * float("nan")

    monkeypatch.setattr(diagnosis, "build_segmenter", lambda *a, **kw: Nonfinite())
    with pytest.raises(ValueError, match="non-finite"):
        diagnosis.run_diagnosis(root, run_dir=source, reference_dir=reference, output_dir=tmp_path / "nan")


def test_cli_runs_diagnosis_without_a_training_config_argument(tmp_path, monkeypatch, capsys):
    import sys
    monkeypatch.setattr(diagnosis, "run_diagnosis", lambda *a, **kw: {
        "checkpoints": dict.fromkeys(range(8)), "all_reproduced_within_1e_4": True})
    monkeypatch.setattr(sys, "argv", ["diagnose_segmentation_probabilities", "--data-root", str(tmp_path)])
    diagnosis.main()
    output = capsys.readouterr().out
    assert "8 checkpoints" in output and "within 1e-4: True" in output
