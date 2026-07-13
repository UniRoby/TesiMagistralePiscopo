"""Evaluation metrics for patch-wise 3D manipulation localization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np


def threshold_heatmap(heatmap: np.ndarray, threshold: float) -> np.ndarray:
    """Threshold a heatmap into a boolean manipulation mask."""

    return np.asarray(heatmap) >= float(threshold)


@dataclass(frozen=True)
class BinaryLocalizationMetrics:
    """Voxel-level binary localization metrics after thresholding."""

    precision: float
    recall: float
    f1: float
    dice: float
    iou: float

    def to_dict(self) -> dict[str, float]:
        """Return metrics as a plain dictionary for logging or JSON export."""

        return asdict(self)


def binary_localization_metrics(mask: np.ndarray, prediction: np.ndarray) -> BinaryLocalizationMetrics:
    """Compute precision, recall, F1/Dice, and IoU from binary voxel masks."""

    mask = np.asarray(mask).astype(bool)
    prediction = np.asarray(prediction).astype(bool)
    if mask.shape != prediction.shape:
        raise ValueError("mask and prediction must have the same shape")

    tp = np.logical_and(mask, prediction).sum(dtype=np.float64)
    fp = np.logical_and(~mask, prediction).sum(dtype=np.float64)
    fn = np.logical_and(mask, ~prediction).sum(dtype=np.float64)

    precision = 1.0 if tp + fp == 0 else float(tp / (tp + fp))
    recall = 1.0 if tp + fn == 0 else float(tp / (tp + fn))
    f1 = 1.0 if 2 * tp + fp + fn == 0 else float((2 * tp) / (2 * tp + fp + fn))
    iou = 1.0 if tp + fp + fn == 0 else float(tp / (tp + fp + fn))
    return BinaryLocalizationMetrics(precision=precision, recall=recall, f1=f1, dice=f1, iou=iou)


def f1_iou(mask: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    """Backward-compatible helper returning only F1/Dice and IoU."""

    metrics = binary_localization_metrics(mask, prediction)
    return metrics.f1, metrics.iou


def voxel_auc_ap(mask: np.ndarray, heatmap: np.ndarray) -> tuple[float, float]:
    """Return voxel ROC-AUC and average precision from continuous heatmap scores."""

    mask_flat = np.asarray(mask).astype(bool).reshape(-1)
    scores_flat = np.asarray(heatmap, dtype=np.float32).reshape(-1)
    if mask_flat.shape != scores_flat.shape:
        raise ValueError("mask and heatmap must have the same number of voxels")
    if np.unique(mask_flat).size < 2:
        return float("nan"), float("nan")
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("scikit-learn is required for voxel_auc_ap") from exc
    return float(roc_auc_score(mask_flat, scores_flat)), float(average_precision_score(mask_flat, scores_flat))


def max_detection_score(heatmap: np.ndarray) -> float:
    """Return volume-level detection score as maximum heatmap value."""

    return float(np.max(heatmap))


def topk_detection_score(heatmap: np.ndarray, fraction: float = 0.01) -> float:
    """Return volume-level detection score as mean of top-k heatmap voxels."""

    scores = np.asarray(heatmap, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        raise ValueError("heatmap cannot be empty")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    k = max(1, int(np.ceil(scores.size * fraction)))
    top = np.partition(scores, scores.size - k)[-k:]
    return float(np.mean(top))


def volume_detection_scores(heatmaps: Sequence[np.ndarray], mode: str = "max", topk_fraction: float = 0.01) -> np.ndarray:
    """Convert many heatmaps to one scalar detection score per volume."""

    if mode not in {"max", "topk_mean"}:
        raise ValueError("mode must be 'max' or 'topk_mean'")
    if mode == "max":
        return np.asarray([max_detection_score(heatmap) for heatmap in heatmaps], dtype=np.float32)
    return np.asarray([topk_detection_score(heatmap, fraction=topk_fraction) for heatmap in heatmaps], dtype=np.float32)


def balanced_accuracy(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    """Compute balanced accuracy for binary labels and continuous scores."""

    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_score) >= float(threshold)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_score must have the same shape")
    positives = y_true.sum(dtype=np.float64)
    negatives = (~y_true).sum(dtype=np.float64)
    tpr = 1.0 if positives == 0 else np.logical_and(y_true, y_pred).sum() / positives
    tnr = 1.0 if negatives == 0 else np.logical_and(~y_true, ~y_pred).sum() / negatives
    return float((tpr + tnr) / 2.0)


def volume_auc_ba(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> tuple[float, float]:
    """Return volume-level ROC-AUC and balanced accuracy."""

    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score, dtype=np.float32)
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape")
    auc = float("nan")
    if np.unique(y_true).size == 2:
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError as exc:  # pragma: no cover - depends on optional env
            raise RuntimeError("scikit-learn is required for volume_auc_ba") from exc
        auc = float(roc_auc_score(y_true, y_score))
    return auc, balanced_accuracy(y_true, y_score, threshold=threshold)


def best_threshold_by_f1(
    mask: np.ndarray,
    heatmap: np.ndarray,
    thresholds: Iterable[float] | None = None,
) -> tuple[float, BinaryLocalizationMetrics]:
    """Select heatmap threshold on validation data by voxel F1/Dice."""

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    best_threshold = 0.5
    best_metrics = binary_localization_metrics(mask, threshold_heatmap(heatmap, best_threshold))
    for threshold in thresholds:
        metrics = binary_localization_metrics(mask, threshold_heatmap(heatmap, float(threshold)))
        if metrics.f1 > best_metrics.f1:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics
