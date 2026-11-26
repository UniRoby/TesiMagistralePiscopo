"""Joint capacity recovery preserves the previous epoch-mean loss and microbatch memory."""
import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from test_segmentation_batch_comparison import FakeDataset
from tesi_m3d import compare_segmentation_batches as batches
from tesi_m3d import compare_segmentation_padding as comparison
from tesi_m3d.segmentation import segmentation_loss
from tesi_m3d.train import load_yaml_config
from tesi_m3d.train_segmentation import train_volume, validate_config, run_training


@pytest.mark.parametrize("microbatch", [1, 2])
def test_joint_update_matches_the_mean_of_six_original_objectives(microbatch):
    torch.manual_seed(21)
    pool = batches.FixedCropPool(FakeDataset())
    expected = torch.nn.Conv3d(1, 1, 1)
    actual = copy.deepcopy(expected)
    optimizer = torch.optim.SGD(expected.parameters(), lr=.01)
    optimizer.zero_grad()
    objective = 0
    for indices in batches.epoch_batches(pool, 21, 0, mixed=True):
        loss = segmentation_loss(expected(pool.image[indices]), pool.mask[indices],
                                 foreground_normalizer=4, squared_pred=True) / 6
        objective += float(loss.detach())
        loss.backward()
    torch.nn.utils.clip_grad_norm_(expected.parameters(), 1)
    optimizer.step()
    physical_batches = []
    actual.register_forward_pre_hook(lambda model, inputs: physical_batches.append(len(inputs[0])))
    loss = train_volume(actual, {"image": pool.image, "mask": pool.mask},
                        torch.optim.SGD(actual.parameters(), lr=.01), microbatch=microbatch,
                        accumulation=48 // microbatch, foreground_normalizer=24, squared_pred=True)
    assert loss == pytest.approx(objective, rel=1e-6)
    assert physical_batches == [microbatch] * (48 // microbatch)
    for a, b in zip(actual.parameters(), expected.parameters()):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)


def test_recovery_effective_config_is_valid_but_cannot_use_the_volume_loader(tmp_path):
    config = load_yaml_config("configs/segmentation_overfit_squared.yaml")
    config["overfit_protocol"] = "full_pool"
    config["training"].update(epochs=300, accumulation_steps=24)
    validate_config(config)
    with pytest.raises(ValueError, match="full-pool-overfit"):
        run_training(config, tmp_path)
    config["stage"] = "pilot"
    with pytest.raises(ValueError, match="dedicated overfit"):
        validate_config(config)


def test_recovery_requires_strict_mode_and_full_pool_size(tmp_path):
    config = load_yaml_config("configs/segmentation_overfit_squared.yaml")
    pool = batches.FixedCropPool(FakeDataset())
    with pytest.raises(ValueError, match="one 48-crop update"):
        batches.run_arm(config, {}, pool, tmp_path, mixed=True, device="cpu", full_pool=True)
    with pytest.raises(ValueError, match="multiple modes"):
        comparison.run_comparison(config, tmp_path, full_pool=True, lr_decay=True)


def test_recovery_pilots_keep_the_clipping_comparison_and_bind_the_correct_gate():
    from tesi_m3d.segmentation import protocol_signature
    configs = [load_yaml_config(f"configs/segmentation_{name}.yaml")
               for name in ("pilot_recovery", "pilot_recovery_clip", "full_recovery")]
    first, second, full = configs
    assert protocol_signature(first) == protocol_signature(second)
    assert first["preprocessing"]["clip"] is False and second["preprocessing"]["clip"] is True
    assert first["output_dir"] != second["output_dir"]
    for config in configs:
        validate_config(config)
        assert config["training"]["squared_pred"] is True
        assert config["overfit_report"] == "outputs/segmentation_overfit_recovery/full_pool/gate.json"
        assert config["training"]["batch_size"] * config["training"]["accumulation_steps"] == 8
        assert config["training"]["num_workers"] == 1
        assert config["training"]["volume_cache_size"] == 1
    assert full["pilot_reports"] == [c["output_dir"] + "/evaluation/metrics.json" for c in (first, second)]


@pytest.mark.parametrize("passed", [True, False])
def test_recovery_cli_selects_new_output_and_gate_code(tmp_path, monkeypatch, passed):
    import sys
    calls = []
    monkeypatch.setattr(comparison, "run_comparison", lambda *a, **kw: calls.append(kw) or {"passed": passed})
    monkeypatch.setattr(sys, "argv", ["compare_segmentation_padding", "--data-root", str(tmp_path), "--full-pool-overfit"])
    if passed:
        comparison.main()
    else:
        with pytest.raises(SystemExit) as stopped:
            comparison.main()
        assert stopped.value.code == 2
    assert calls[0]["full_pool"] and not calls[0]["lr_decay"] and not calls[0]["deterministic_repeats"]
    assert calls[0]["output_dir"] == "outputs/segmentation_overfit_recovery"
