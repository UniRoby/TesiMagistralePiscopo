"""Change only the Dice denominator; keep paired probes and common evaluation."""
import copy
import csv
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from monai.losses import DiceLoss
from test_segmentation import TinySegmenter
from test_segmentation_hard_crops import setup_reference
from tesi_m3d import compare_segmentation_losses as comparison
from tesi_m3d import debug_segmentation as debug
from tesi_m3d import diagnose_segmentation_probabilities as probabilities
from tesi_m3d.diagnose_localization import write_json
from tesi_m3d.segmentation import file_digest, segmentation_loss, segmentation_loss_components
from tesi_m3d.train_segmentation import train_volume


@pytest.mark.parametrize("squared", [False, True])
def test_dice_denominator_matches_monai_for_positive_crops_and_preserves_bce(squared):
    torch.manual_seed(21)
    logits = torch.randn(4, 1, 8, 8, 8, requires_grad=True)
    mask = torch.zeros_like(logits)
    mask[:2, :, :2, :3, :4] = 1
    bce, dice = segmentation_loss_components(logits, mask, squared_pred=squared)
    expected = DiceLoss(sigmoid=True, squared_pred=squared)(logits[:2], mask[:2])
    torch.testing.assert_close(dice, expected)
    torch.testing.assert_close(bce, torch.nn.functional.binary_cross_entropy_with_logits(logits, mask))
    default = segmentation_loss_components(logits, mask)
    if not squared:
        assert torch.equal(default[0], bce) and torch.equal(default[1], dice)
    combined = segmentation_loss(logits, mask, squared_pred=squared)
    torch.testing.assert_close(combined, .5 * bce + .5 * expected)
    combined.backward()
    assert torch.isfinite(logits.grad).all()


def test_empty_targets_keep_exactly_bce_gradients_in_both_arms():
    values = torch.linspace(-100, 100, 512).reshape(1, 1, 8, 8, 8)
    results = []
    for squared in (False, True):
        logits = values.clone().requires_grad_()
        loss = segmentation_loss(logits, torch.zeros_like(logits), squared_pred=squared)
        loss.backward()
        assert torch.isfinite(logits.grad).all()
        results.append((loss.detach(), logits.grad))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])
    with pytest.raises(ValueError, match="squared_pred"):
        segmentation_loss(values, torch.zeros_like(values), squared_pred="true")


@pytest.mark.parametrize("microbatch,accumulation", [(1, 8), (2, 4)])
@pytest.mark.parametrize("normalizer", [None, 4])
def test_squared_loss_accumulation_matches_full_batch_parameter_updates(microbatch, accumulation, normalizer):
    torch.manual_seed(21)
    image = torch.rand(8, 1, 8, 8, 8)
    mask = torch.zeros_like(image)
    mask[:2, :, :2, :3, :4] = 1
    model = TinySegmenter()
    reference = copy.deepcopy(model)
    measured = train_volume(model, {"image": image, "mask": mask}, torch.optim.SGD(model.parameters(), lr=.01),
                             microbatch=microbatch, accumulation=accumulation,
                             foreground_normalizer=normalizer, squared_pred=True)
    loss = segmentation_loss(reference(image), mask, squared_pred=True, foreground_normalizer=normalizer)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1)
    torch.optim.SGD(reference.parameters(), lr=.01).step()
    assert measured == pytest.approx(float(loss.detach()), rel=1e-6)
    for a, b in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(a, b)


def setup_comparison(tmp_path, monkeypatch):
    config, root, reference, _ = setup_reference(tmp_path)
    gate = tmp_path / "original_gate.json"
    gate.write_text('{"passed":false}')
    config["overfit_report"] = str(gate)
    write_json(reference / "effective_config.json", config)
    initial_models = []

    def build(config):
        model = TinySegmenter()
        initial_models.append(copy.deepcopy(model.state_dict()))
        return model

    def restore(config, *, state_dict):
        model = TinySegmenter()
        model.load_state_dict(state_dict, strict=True)
        return model

    monkeypatch.setattr(debug, "build_segmenter", build)
    monkeypatch.setattr(probabilities, "build_segmenter", restore)
    return config, root, reference, gate, initial_models


def test_eight_fresh_paired_runs_export_both_checkpoints_and_preserve_previous_results(tmp_path, monkeypatch):
    config, root, reference, gate, models = setup_comparison(tmp_path, monkeypatch)
    before = {p: file_digest(p) for p in reference.iterdir() if p.is_file()}
    output = tmp_path / "loss_comparison"
    result = comparison.run_comparison(config, root, reference_dir=reference, output_dir=output)
    assert result["complete"] and result["active_stage"] is result["active_checkpoint"] is None
    assert result["all_reproduced_within_1e_4"]
    assert len(models) == 8
    for initial in models[1:]:
        for name, tensor in initial.items():
            assert torch.equal(tensor, models[0][name])
    assert before == {p: file_digest(p) for p in reference.iterdir() if p.is_file()}
    assert gate.read_text() == '{"passed":false}'
    assert len(list(output.rglob("probabilities.npz"))) == 16
    for arm, squared in (("linear", False), ("squared", True)):
        stages = result["arms"][arm]["stages"]
        assert len(stages) == 4
        for key, stage in stages.items():
            assert stage["last"]["updates"] == stage["last"]["exposures_per_crop"] == 2
            assert stage["last"]["initial_state_sha256"] == result["initial_state_sha256"]
            assert stage["update_check"]["passed"] and stage["last"]["training_loss"]["squared_pred"] == squared
            folder = output / arm / key
            updates = list(csv.DictReader((folder / "updates.csv").open()))
            assert len(updates) == 2 and {r["squared_pred"] for r in updates} == {str(squared)}
            for r in updates:
                assert float(r["loss"]) == pytest.approx(.5 * (float(r["bce"]) + float(r["soft_dice_loss"])))
            for checkpoint in ("best", "last"):
                diag = stage["probability_diagnostics"][checkpoint]
                assert diag["metrics"]["soft_dice_loss"] == pytest.approx(stage[checkpoint]["soft_dice_loss"], abs=1e-7)
                with np.load(folder / checkpoint / "probabilities.npz") as array:
                    assert array["probability"].dtype == np.float32
                    assert array["mask"].shape == array["probability"].shape
                assert (folder / checkpoint / "distributions.csv").is_file()
                assert (folder / checkpoint / "positive.png").is_file()
    assert result["passed"] == all(s["last"]["passed"] for s in result["arms"]["squared"]["stages"].values())
    for key, delta in result["deltas_squared_minus_linear"].items():
        for checkpoint in ("best", "last"):
            a, b = [torch.load(output / arm / key / f"{checkpoint}.pt", weights_only=False) for arm in ("linear", "squared")]
            assert a["config"] == b["config"] and a["fixed_batch"] == b["fixed_batch"]
            assert not a["training_loss"]["squared_pred"] and b["training_loss"]["squared_pred"]
            expected_delta = (result["arms"]["squared"]["stages"][key][checkpoint]["positive_crop_dice"]
                              - result["arms"]["linear"]["stages"][key][checkpoint]["positive_crop_dice"])
            assert delta[checkpoint]["hard_dice"] == pytest.approx(expected_delta, abs=1e-7)
    assert json.loads((output / "comparison.json").read_text()) == result
    with pytest.raises(ValueError, match="already contains"):
        comparison.run_comparison(config, root, reference_dir=reference, output_dir=output)
    with pytest.raises(ValueError, match="outside the reference"):
        comparison.run_comparison(config, root, reference_dir=reference, output_dir=reference / "new")


@pytest.mark.parametrize("failure", ["initial_state", "squared_training", "probability_evaluation"])
def test_technical_failure_marks_comparison_incomplete_and_preserves_partial_results(tmp_path, monkeypatch, failure):
    config, root, reference, _, _ = setup_comparison(tmp_path, monkeypatch)
    if failure == "initial_state":
        def wrong(config):
            model = TinySegmenter()
            model.encoder.bias.data.add_(1)
            return model
        monkeypatch.setattr(debug, "build_segmenter", wrong)
    elif failure == "squared_training":
        original = comparison.run_fixed_stage
        def fail(*args, **kwargs):
            if kwargs["squared_pred"]:
                raise FloatingPointError("non-finite squared gradient")
            return original(*args, **kwargs)
        monkeypatch.setattr(comparison, "run_fixed_stage", fail)
    else:
        def fail(*args, **kwargs):
            raise ValueError("invalid checkpoint binding")
        monkeypatch.setattr(comparison, "evaluate_checkpoint", fail)
    output = tmp_path / "failed"
    with pytest.raises((ValueError, FloatingPointError)):
        comparison.run_comparison(config, root, reference_dir=reference, output_dir=output)
    report = json.loads((output / "comparison.json").read_text())
    assert not report["complete"] and not report["passed"] and "error" in report
    assert report["active_stage"] == ("squared" if failure == "squared_training" else "linear") + "/inj_2092/positive"
    if failure == "squared_training":
        assert len(report["arms"]["linear"]["stages"]) == 4
    if failure == "probability_evaluation":
        assert report["active_checkpoint"] == "linear/inj_2092/positive/best"


@pytest.mark.parametrize("passed", [False, True])
def test_cli_separates_failed_scientific_gate_from_technical_error(tmp_path, monkeypatch, passed):
    import sys
    monkeypatch.setattr(comparison, "run_comparison", lambda *a, **kw: {
        "complete": True, "passed": passed, "all_reproduced_within_1e_4": True, "deltas_squared_minus_linear": {}})
    monkeypatch.setattr(sys, "argv", ["compare_segmentation_losses", "--data-root", str(tmp_path)])
    if passed:
        comparison.main()
    else:
        with pytest.raises(SystemExit) as stopped:
            comparison.main()
        assert stopped.value.code == 2
