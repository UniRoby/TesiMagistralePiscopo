"""Late LR change, exact reference prefix and preserved overfit budgets/audits."""
import copy
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from test_segmentation_batch_comparison import FakeDataset
from test_segmentation_padding_comparison import reference, TinyPaddingSegmenter
from tesi_m3d import compare_segmentation_batches as batches
from tesi_m3d import compare_segmentation_padding as comparison
from tesi_m3d.diagnose_localization import write_csv
from tesi_m3d.diagnose_segmentation_batch_probabilities import read_rows
from tesi_m3d.segmentation import file_digest
from tesi_m3d.train import load_yaml_config


def test_full_300_update_comparison_preserves_reference_and_exports_both_audits(reference, tmp_path):
    config, root, _ = reference
    config["training"].update(epochs=50, max_updates=300)
    source = tmp_path / "full_source"
    batches.run_comparison(config, root, output_dir=source)
    control = tmp_path / "constant"
    comparison.run_comparison(config, root, reference_dir=source, output_dir=control, deterministic_repeats=True)
    before = {p: file_digest(p) for p in control.rglob("*") if p.is_file()}
    original_config = copy.deepcopy(config)
    output = tmp_path / "decay"
    result = comparison.run_comparison(config, root, reference_dir=control, output_dir=output, lr_decay=True)
    assert result["complete"] and result["active_arm"] is result["active_checkpoint"] is None
    assert result["experiment"] == comparison.LR_EXPERIMENT
    assert set(result["arms"]) == {"decay"}
    assert config == original_config
    assert before == {p: file_digest(p) for p in control.rglob("*") if p.is_file()}
    assert result["reference_prefix"] == {"required_updates": 200, "matched_updates": 200,
                                         "passed": True, "first_divergence": None}
    assert result["reference_control"] == json.loads((control / "comparison.json").read_text())["arms"]["repeat_1"]
    arm = result["arms"]["decay"]
    assert arm["updates"] == 300 and arm["exposures_per_crop"] == 50
    assert result["passed"] == arm["last"]["passed"]
    trace = read_rows(output / "decay/update_metrics.csv")
    original = read_rows(control / "repeat_1/update_metrics.csv")
    for i, (row, prior) in enumerate(zip(trace, original)):
        assert all(row[k] == prior[k] for k in ("crop_indices", "positive_crops", "real_crops", "epoch"))
        if i < 200:
            assert {k: row[k] for k in prior} == prior
        factor = 1 if i < 200 else .1
        assert float(row["encoder_lr"]) == pytest.approx(config["training"]["encoder_lr"] * factor)
        assert float(row["decoder_lr"]) == pytest.approx(config["training"]["decoder_lr"] * factor)
    assert trace[200]["model_state_sha256"] != original[200]["model_state_sha256"]
    for cp in ("best", "last"):
        payload = torch.load(output / "decay" / f"{cp}.pt", weights_only=False)
        assert payload["lr_schedule"] == {"after_updates": 200, "factor": .1, "groups": ["encoder", "decoder"]}
        assert payload["execution_policy"]["strict_determinism"]
        expected_factor = .1 if payload["updates"] > 200 else 1
        for group, key in zip(payload["optimizer_state_dict"]["param_groups"], ("encoder_lr", "decoder_lr")):
            assert group["lr"] == pytest.approx(config["training"][key] * expected_factor)
        folder = output / "decay/probabilities" / cp
        assert (folder / "metrics.npz").is_file() and (folder / "boundary_metrics.csv").is_file()
        assert len(read_rows(folder / "crops.csv")) == 48 and len(read_rows(folder / "thresholds.csv")) == 1001
        assert arm["probability_diagnostics"][cp]["reproduced_within_1e_4"]
        for key in ("positive_crop_dice", "negative_crop_fp_fraction"):
            assert result["delta_decay_minus_constant"][cp][key] == arm[cp][key] - result["reference_control"][cp][key]
    assert json.loads((output / "comparison.json").read_text()) == result
    with pytest.raises(ValueError, match="already contains"):
        comparison.run_comparison(config, root, reference_dir=control, output_dir=output, lr_decay=True)

    before_decay = {p: file_digest(p) for p in output.rglob("*") if p.is_file()}
    recovered_dir = tmp_path / "full_pool"
    recovered = comparison.run_comparison(config, root, reference_dir=output, output_dir=recovered_dir, full_pool=True)
    assert recovered["complete"] and recovered["experiment"] == comparison.RECOVERY_EXPERIMENT
    assert recovered["compute_ratio_crop_presentations"] == 6
    assert before_decay == {p: file_digest(p) for p in output.rglob("*") if p.is_file()}
    arm = recovered["arms"]["full_pool"]
    assert arm["updates"] == arm["epochs"] == arm["exposures_per_crop"] == 300
    assert arm["effective_batch_size"] == 48 and arm["fixed_foreground_normalizer"] == 24
    assert arm["accumulation_steps"] == 24
    assert recovered["passed"] == arm["last"]["passed"]
    joint_trace = read_rows(recovered_dir / "full_pool/update_metrics.csv")
    assert len(joint_trace) == 300
    assert {r["crop_indices"] for r in joint_trace} == {";".join(map(str, range(48)))}
    assert {r["positive_crops"] for r in joint_trace} == {"16"}
    assert {r["real_crops"] for r in joint_trace} == {"16"}
    assert all(float(r["encoder_lr"]) == config["training"]["encoder_lr"] for r in joint_trace[:200])
    assert float(joint_trace[200]["encoder_lr"]) == pytest.approx(3e-6)
    assert {r["exposures"] for r in read_rows(recovered_dir / "full_pool/exposures.csv")} == {"300"}
    for cp in ("best", "last"):
        payload = torch.load(recovered_dir / "full_pool" / f"{cp}.pt", weights_only=False)
        assert payload["config"]["overfit_protocol"] == "full_pool"
        assert payload["config"]["training"]["accumulation_steps"] == 24
        assert payload["config"]["training"]["epochs"] == 300
        assert payload["sampling"]["layout"] == "full_pool"
        assert payload["sampling"]["fixed_foreground_normalizer"] == 24
        assert arm["probability_diagnostics"][cp]["reproduced_within_1e_4"]
        assert (recovered_dir / "full_pool/probabilities" / cp / "metrics.npz").is_file()
    assert json.loads((recovered_dir / "comparison.json").read_text()) == recovered

    # Equal edits to both repeats evade a pairwise comparison, but not the new prefix guard.
    for name in ("repeat_1", "repeat_2"):
        path = control / name / "update_metrics.csv"
        rows = read_rows(path)
        rows[0]["loss"] = str(float(rows[0]["loss"]) + 1e-8)
        write_csv(path, rows)
    rejected = tmp_path / "rejected"
    with pytest.raises(ValueError, match="reference trace diverged at update 1"):
        comparison.run_comparison(config, root, reference_dir=control, output_dir=rejected, lr_decay=True)
    failed = json.loads((rejected / "comparison.json").read_text())
    assert not failed["complete"] and not failed["passed"] and failed["active_arm"] == "decay"
    assert failed["reference_prefix"]["matched_updates"] == 0
    assert set(failed["reference_prefix"]["first_divergence"]["differences"]) == {"loss"}
    assert len(read_rows(rejected / "decay/update_metrics.csv")) == 1
    assert not (rejected / "decay/probabilities").exists()
    path = control / "repeat_2/update_metrics.csv"
    rows[0]["loss"] = "different"
    write_csv(path, rows)
    with pytest.raises(ValueError, match="reproducible deterministic reference"):
        comparison.run_comparison(config, root, reference_dir=control, output_dir=tmp_path / "unequal", lr_decay=True)


@pytest.mark.parametrize("change", ["loss", "gradient_sha256", "crop_indices_after_decay"])
def test_prefix_drift_and_post_decay_sampling_change_stop_immediately(tmp_path, monkeypatch, change):
    pool = batches.FixedCropPool(FakeDataset())
    monkeypatch.setattr(batches, "build_segmenter", lambda cfg: TinyPaddingSegmenter("zeros"))
    monkeypatch.setattr(batches, "LR_DECAY_AFTER", 4)
    config = load_yaml_config("configs/segmentation_overfit_squared.yaml")
    config["training"].update(epochs=1, max_updates=6)
    seen = []
    train = batches.train_volume

    def capture(model, batch, optimizer, **kwargs):
        seen.append(tuple(group["lr"] for group in optimizer.param_groups))
        return train(model, batch, optimizer, **kwargs)

    monkeypatch.setattr(batches, "train_volume", capture)
    with comparison.deterministic_execution():
        batches.run_arm(config, {"manifest_id": "fixture"}, pool, tmp_path / "control", mixed=True, device="cpu", deterministic=True)
        reference_trace = read_rows(tmp_path / "control/update_metrics.csv")
        index = 4 if change == "crop_indices_after_decay" else 1
        key = "crop_indices" if change == "crop_indices_after_decay" else change
        reference_trace[index][key] = "0" * 64 if key == "gradient_sha256" else "changed"
        seen.clear()
        with pytest.raises(ValueError, match=f"diverged at update {index + 1}"):
            batches.run_arm(config, {"manifest_id": "fixture"}, pool, tmp_path / "new", mixed=True, device="cpu", deterministic=True,
                            lr_decay_reference=reference_trace)
    assert len(seen) == index + 1
    base = (config["training"]["encoder_lr"], config["training"]["decoder_lr"])
    assert all(rates == base for rates in seen[:4])
    if index == 4:
        assert seen[-1] == pytest.approx(tuple(rate * .1 for rate in base))
    else:
        assert all(rates == base for rates in seen)
    prefix = json.loads((tmp_path / "new/prefix_check.json").read_text())
    assert prefix["first_divergence"]["updates"] == index + 1


def test_lr_mode_rejects_different_budget_and_conflicting_modes(tmp_path):
    config = load_yaml_config("configs/segmentation_overfit_squared.yaml")
    config["training"].update(epochs=1, max_updates=6)
    with pytest.raises(ValueError, match="exactly 300 updates"):
        comparison.run_comparison(config, tmp_path, lr_decay=True)
    with pytest.raises(ValueError, match="not both"):
        comparison.run_comparison(config, tmp_path, lr_decay=True, deterministic_repeats=True)


@pytest.mark.parametrize("passed,code", [(True, None), (False, 2)])
def test_cli_selects_lr_mode_and_separate_output(tmp_path, monkeypatch, passed, code):
    import sys
    calls = []
    monkeypatch.setattr(comparison, "run_comparison", lambda *a, **kw: calls.append(kw) or {
        "passed": passed, "reference_prefix": {"passed": True}})
    monkeypatch.setattr(sys, "argv", ["compare_segmentation_padding", "--data-root", str(tmp_path), "--lr-decay"])
    if code is None:
        comparison.main()
    else:
        with pytest.raises(SystemExit) as stopped:
            comparison.main()
        assert stopped.value.code == code
    assert calls[0]["lr_decay"] and not calls[0]["deterministic_repeats"]
    assert calls[0]["output_dir"] == "outputs/segmentation_lr_decay_comparison"
