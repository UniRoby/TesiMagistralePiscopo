"""Offline checks for mandatory MedicalNet initialization and checkpoint restore."""

from dataclasses import asdict
import hashlib
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")
from monai.networks.nets import resnet18
from tesi_m3d.model import Patch3DModelConfig, build_patch3d_classifier


@pytest.fixture(scope="module")
def encoder_weights():
    # Simulate a pretrained checkpoint; no network or large binary fixture in git.
    encoder = resnet18(spatial_dims=3, n_input_channels=1, shortcut_type="A", feed_forward=False)
    with torch.no_grad():
        encoder.conv1.weight.fill_(0.125)
    return {"module." + key: value for key, value in encoder.state_dict().items()}


@pytest.fixture
def initialized_model(tmp_path, monkeypatch, encoder_weights):
    import tesi_m3d.model as module

    weight_file = tmp_path / "resnet_18_23dataset.pth"
    weight_file.write_bytes(b"offline weight fixture")
    monkeypatch.setattr(module, "MEDICALNET_SHA256", hashlib.sha256(weight_file.read_bytes()).hexdigest())
    config = Patch3DModelConfig(medicalnet_weights=str(weight_file), dropout=0.0)
    with patch.object(torch, "load", return_value={"state_dict": encoder_weights}) as loader:
        model = build_patch3d_classifier(config)
    loader.assert_called_once_with(weight_file, map_location="cpu", weights_only=True)
    return model, config


def test_loads_every_encoder_weight_and_trains_on_32_cubed(initialized_model, encoder_weights):
    model, _ = initialized_model
    actual = model.state_dict()
    for key, expected in encoder_weights.items():
        assert torch.equal(actual[key.removeprefix("module.")], expected)
    assert model.conv1.stride == (2, 2, 2)
    for layer, dilation in ((model.layer3, 2), (model.layer4, 4)):
        assert layer[0].downsample.keywords["stride"] == 1
        assert layer[0].conv1.stride == (1, 1, 1)
        assert all(block.conv2.dilation == (dilation,) * 3 for block in layer)
    logits = model(torch.randn(2, 1, 32, 32, 32))
    assert logits.shape == (2, 1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([[0.0], [1.0]]))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.conv1.weight.grad is not None
    assert torch.isfinite(model.conv1.weight.grad).all()
    assert model.fc[1].weight.grad.abs().sum() > 0


def test_missing_medicalnet_file_never_returns_random_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="random initialization is disabled"):
        build_patch3d_classifier(Patch3DModelConfig(medicalnet_weights=str(tmp_path / "missing.pth")))


def test_corrupt_or_different_weights_are_rejected(tmp_path):
    path = tmp_path / "wrong.pth"
    path.write_bytes(b"not the official checkpoint")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        build_patch3d_classifier(Patch3DModelConfig(medicalnet_weights=str(path)))


def test_incomplete_medicalnet_state_is_rejected(tmp_path, monkeypatch):
    import tesi_m3d.model as module

    path = tmp_path / "weights.pth"
    path.write_bytes(b"fixture")
    monkeypatch.setattr(module, "MEDICALNET_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    with patch.object(torch, "load", return_value={"state_dict": {}}):
        with pytest.raises(RuntimeError, match="Missing key"):
            build_patch3d_classifier(Patch3DModelConfig(medicalnet_weights=str(path)))


def test_finetuned_checkpoint_restores_offline_without_medicalnet_file(initialized_model, tmp_path):
    from tesi_m3d.evaluate_patch_level import _load_model

    model, config = initialized_model
    cfg = asdict(config)
    cfg["medicalnet_weights"] = str(tmp_path / "not-present.pth")
    checkpoint = tmp_path / "best.pt"
    torch.save({"config": {"model": cfg}, "model_state_dict": model.state_dict()}, checkpoint)
    restored = _load_model(checkpoint, "cpu")
    for key, expected in model.state_dict().items():
        assert torch.equal(restored.state_dict()[key], expected)
    model.eval()
    restored.eval()
    with torch.no_grad():
        image = torch.randn(1, 1, 32, 32, 32)
        torch.testing.assert_close(model(image), restored(image))
    bad_state = dict(model.state_dict())
    del bad_state["fc.1.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        build_patch3d_classifier(config, state_dict=bad_state)


@pytest.mark.parametrize("config", [Patch3DModelConfig(name="simple_3d_cnn"), Patch3DModelConfig(in_channels=3)])
def test_unsupported_model_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        build_patch3d_classifier(config)
