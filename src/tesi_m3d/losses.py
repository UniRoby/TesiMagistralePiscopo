"""Loss functions for binary 3D patch manipulation detection."""

from __future__ import annotations


class TorchDependencyError(RuntimeError):
    """Raised when loss construction requires PyTorch but it is unavailable."""



def _torch_modules():
    """Import PyTorch lazily so non-training modules stay lightweight."""

    try:
        import torch
        from torch import nn
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - depends on training env
        raise TorchDependencyError(
            "PyTorch is required for training losses. Install with `pip install -e '.[train]'`."
        ) from exc
    return torch, nn, functional


def build_loss(name: str = "focal", *, alpha: float = 0.25, gamma: float = 2.0):
    """Create binary patch-level loss.

    Args:
        name: ``"bce"`` for BCEWithLogitsLoss or ``"focal"`` for BinaryFocalLoss.
        alpha: Positive-class weighting used by focal loss.
        gamma: Focusing parameter used by focal loss.

    Returns:
        PyTorch loss module accepting logits and targets with matching shape.
    """

    _, nn, _ = _torch_modules()
    normalized = name.lower().strip()
    if normalized == "bce":
        return nn.BCEWithLogitsLoss()
    if normalized == "focal":
        return BinaryFocalLoss(alpha=alpha, gamma=gamma)
    raise ValueError("loss name must be 'bce' or 'focal'")


class BinaryFocalLoss(_torch_modules()[1].Module):
    """Binary focal loss for imbalanced patch labels.

    Inputs are raw logits with shape ``(B, 1)`` or ``(B,)``. Targets are binary
    labels or soft labels in ``[0, 1]`` with compatible shape.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        """Store focal loss parameters."""

        _, nn, _ = _torch_modules()
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits, targets):
        """Compute focal loss from logits and binary/soft targets."""

        torch, _, functional = _torch_modules()
        targets = targets.to(dtype=logits.dtype).view_as(logits)
        bce = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        # p_t is model probability assigned to the target class for each sample.
        p_t = (prob * targets) + ((1.0 - prob) * (1.0 - targets))
        # alpha_t mirrors RetinaNet focal weighting for binary targets.
        alpha_t = (self.alpha * targets) + ((1.0 - self.alpha) * (1.0 - targets))
        loss = alpha_t * ((1.0 - p_t) ** self.gamma) * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
