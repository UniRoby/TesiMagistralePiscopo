"""Patch-level 3D classifiers for M3Dsynth manipulation detection.

The main thesis model is a patch classifier: input is a CT patch with shape
``(B, 1, D, H, W)`` and output is one logit per patch with shape ``(B, 1)``.
``sigmoid(logit)`` is interpreted as probability that the patch is synthetic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Patch3DModelConfig:
    """Configuration for the compact 3D CNN patch classifier."""

    in_channels: int = 1
    num_classes: int = 1
    base_channels: int = 16
    dropout: float = 0.2


class TorchDependencyError(RuntimeError):
    """Raised when PyTorch-dependent code runs without PyTorch installed."""



def build_patch3d_classifier(config: Patch3DModelConfig | None = None):
    """Build a compact 3D CNN for patch classification.

    Args:
        config: Model hyperparameters. Default returns one logit per patch.

    Returns:
        ``torch.nn.Module`` mapping ``(B, C, D, H, W)`` to ``(B, num_classes)``.
    """

    config = config or Patch3DModelConfig()
    try:
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on training env
        raise TorchDependencyError(
            "PyTorch is required to build the 3D classifier. Install with `pip install -e '.[train]'`."
        ) from exc

    class Simple3DCNN(nn.Module):
        """Small 3D CNN encoder with global pooling and patch-level head."""

        def __init__(self) -> None:
            """Create convolutional feature extractor and linear classifier."""

            super().__init__()
            c = config.base_channels
            self.features = nn.Sequential(
                nn.Conv3d(config.in_channels, c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(2),
                nn.Conv3d(c, c * 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(c * 2),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(2),
                nn.Conv3d(c * 2, c * 4, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(c * 4),
                nn.ReLU(inplace=True),
                # Adaptive pooling removes dependence on exact patch size.
                nn.AdaptiveAvgPool3d(1),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(config.dropout),
                nn.Linear(c * 4, config.num_classes),
            )

        def forward(self, x):
            """Return raw logits for input patches with shape ``(B, C, D, H, W)``."""

            return self.classifier(self.features(x))

    return Simple3DCNN()
