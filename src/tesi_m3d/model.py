"""MedicalNet-initialized ResNet-18 for binary 3D patch classification."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping, Any


MEDICALNET_WEIGHTS = "checkpoints/medicalnet/resnet_18_23dataset.pth"
MEDICALNET_SHA256 = "61224f9317fcce873366deb3703183e92cc47325b726b69691b33536244e10f4"


@dataclass(frozen=True)
class Patch3DModelConfig:
    """MedicalNet encoder and a new binary classification head."""

    name: str = "resnet18_medicalnet"
    in_channels: int = 1
    num_classes: int = 1
    dropout: float = 0.2
    medicalnet_weights: str = MEDICALNET_WEIGHTS


class TorchDependencyError(RuntimeError):
    """Raised when the model dependencies are unavailable."""


def build_patch3d_classifier(
    config: Patch3DModelConfig | None = None,
    *,
    state_dict: Mapping[str, Any] | None = None,
):
    """Load MedicalNet for a new run, or restore a complete fine-tuned model.

    A missing or incompatible weight file is an error, never a random-weight
    fallback. Restoring a full state dict needs no original MedicalNet file.
    Input is ``(B, 1, D, H, W)``; output is one raw logit per patch.
    """

    config = config or Patch3DModelConfig()
    if config.name != "resnet18_medicalnet":
        raise ValueError("This branch only supports model.name=resnet18_medicalnet; old CNN checkpoints are incompatible")
    if config.in_channels != 1 or config.num_classes != 1:
        raise ValueError("MedicalNet patch classification requires in_channels=1 and num_classes=1")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")

    if state_dict is None:
        path = Path(config.medicalnet_weights)
        if not path.is_file():
            raise FileNotFoundError(
                f"MedicalNet weights not found: {path}. Download resnet_18_23dataset.pth "
                "as described in docs/LAB_WORKFLOW.md; random initialization is disabled."
            )
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != MEDICALNET_SHA256:
            raise ValueError(f"MedicalNet SHA256 mismatch: {path}; download the official resnet_18_23dataset.pth again")

    try:
        import torch
        from torch import nn
        from monai.networks.nets import resnet18
    except ImportError as exc:  # pragma: no cover - depends on training env
        raise TorchDependencyError("PyTorch and MONAI are required. Install with `pip install -e '.[train]'`.") from exc

    model = resnet18(
        pretrained=False, spatial_dims=3, n_input_channels=1,
        shortcut_type="A", conv1_t_stride=2, feed_forward=False,
    )
    # MedicalNet uses dilated stages 3/4 instead of MONAI's extra downsampling.
    for layer, dilation in ((model.layer3, 2), (model.layer4, 4)):
        layer[0].downsample.keywords["stride"] = 1
        for block in layer:
            block.stride = 1
            for conv in (block.conv1, block.conv2):
                conv.stride = (1, 1, 1)
                conv.dilation = conv.padding = (dilation,) * 3

    if state_dict is None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        weights = {key.removeprefix("module."): value for key, value in payload["state_dict"].items()}
        model.load_state_dict(weights, strict=True)

    model.fc = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(512, 1))
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model
