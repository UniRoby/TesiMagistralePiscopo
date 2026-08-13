"""Training entrypoint for patch-wise 3D M3Dsynth experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import warnings
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from .dataset import M3DSynthPatchDataset, cross_generator_records, label_dir, read_records, scan_dir
from .evaluation import (
    BinaryLocalizationMetrics,
    best_threshold_by_balanced_accuracy,
    max_detection_score,
    topk_detection_score,
    volume_auc_ba,
    voxel_auc_ap,
)
from .inference import infer_heatmap
from .losses import build_loss
from .model import Patch3DModelConfig, build_patch3d_classifier
from .patch_index import load_or_build_patch_index
from .sampling import SequentialVolumeBatchSampler, VolumeGroupedBatchSampler
from .volume_io import align_mask_to_scan, load_label_mask, load_normalized_scan

KNOWN_MODEL_NAMES = {"simple_3d_cnn"}


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load an experiment YAML config using PyYAML."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - optional train env
        raise RuntimeError("PyYAML is required for training configs. Install pyyaml or use train extra.") from exc

    with Path(config_path).open() as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("config must be a YAML mapping")
    return loaded


def set_global_seed(seed: int) -> None:
    """Seed Python, numpy and torch so a run can be reproduced.

    ``training.seed`` was present in every config but never read, which meant no
    thesis run was reproducible. cuDNN autotuning is left enabled: it keeps
    results reproducible to about 1e-4, which is worth the throughput here.
    """

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - optional train env
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def resolve_cache_dir(
    config: dict[str, Any],
    cli_override: str | None,
    output_dir: Path,
) -> Path:
    """Resolve where to store the patch index cache.

    Order: CLI, then config, then ``TESI_M3D_CACHE_DIR``, then a directory under
    the run's output dir. The explicit options matter because on this machine the
    dataset drive has only a few GB free and the cache belongs elsewhere.
    """

    candidate = cli_override or config.get("cache_dir") or os.environ.get("TESI_M3D_CACHE_DIR")
    cache_dir = Path(candidate) if candidate else output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def subset_records(records, max_records: int | None, seed: int = 21):
    """Return a deterministic random subset of ``records``.

    The shuffle matters: ``metadata/m3dsynth/data.csv`` is grouped by manipulation
    type, so a plain head slice would yield only ``rem`` records and no ``inj``
    ones, and a "quick test" would silently train on half the problem.
    """

    records = list(records)
    if max_records is None or max_records >= len(records):
        return records
    shuffled = list(records)
    random.Random(int(seed)).shuffle(shuffled)
    return shuffled[: int(max_records)]


def make_balanced_sampler(dataset: M3DSynthPatchDataset):
    """Create a WeightedRandomSampler with equal total weight for pos/neg labels.

    Deprecated: this samples patches in fully random order across all volumes, so
    every item pays a full volume decode. Use :class:`VolumeGroupedBatchSampler`.
    Kept because external scripts may still import it.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional train env
        raise RuntimeError("PyTorch is required for training sampler") from exc

    labels = np.asarray(dataset.labels, dtype=np.int64)
    if labels.size == 0:
        raise ValueError("dataset has no patch examples")
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    counts[counts == 0] = 1.0
    # Each sample gets inverse-frequency weight so positives and negatives balance.
    weights = 1.0 / counts[labels]
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


class TrainingObjects(NamedTuple):
    """Everything ``main`` needs to run a training loop."""

    model: Any
    train_loader: Any
    train_sampler: Any
    train_dataset: Any
    valid_loader: Any
    valid_dataset: Any
    loss: Any
    optimizer: Any
    scaler: Any
    amp_dtype: Any
    use_amp: bool


def _amp_settings(training_cfg: dict[str, Any], device: str):
    """Return ``(use_amp, amp_dtype, scaler)`` for the requested device.

    ``mixed_precision`` was advertised in every config and never read. bfloat16 is
    the default because this GPU runs it at full rate and it needs no loss
    scaling, which removes the GradScaler/focal-loss interaction entirely.
    """

    import torch

    requested = bool(training_cfg.get("mixed_precision", False))
    use_amp = requested and str(device).startswith("cuda") and torch.cuda.is_available()
    name = str(training_cfg.get("amp_dtype", "bfloat16")).lower()
    if name not in {"bfloat16", "float16"}:
        raise ValueError(f"training.amp_dtype must be bfloat16 or float16, got {name!r}")
    amp_dtype = torch.bfloat16 if name == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16)
    return use_amp, amp_dtype, scaler


def _build_dataloader(
    dataset,
    batch_sampler,
    training_cfg: dict[str, Any],
    *,
    pin_memory: bool = False,
):
    """Wrap ``dataset`` in a DataLoader driven by a volume-grouped batch sampler."""

    import torch

    num_workers = int(training_cfg.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "batch_sampler": batch_sampler,
        "num_workers": num_workers,
        # Pinning makes the non_blocking CUDA copies in the training loop real
        # asynchronous transfers. It is deliberately configurable because it
        # costs some host RAM and is useless for CPU/MPS runs.
        "pin_memory": bool(pin_memory and training_cfg.get("pin_memory", True)),
    }
    if num_workers > 0:
        # Without persistent workers, Windows respawns and re-imports per epoch.
        kwargs["persistent_workers"] = bool(training_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(training_cfg.get("prefetch_factor", 2))
    return torch.utils.data.DataLoader(dataset, **kwargs)


def build_loaders(
    config: dict[str, Any],
    data_root_override: str | None = None,
    cache_dir: str | Path | None = None,
    device: str = "cpu",
    rebuild_index: bool = False,
) -> TrainingObjects:
    """Build model, train/valid loaders, loss, optimizer and AMP state."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional train env
        raise RuntimeError("PyTorch is required for training") from exc

    data_root = Path(data_root_override or config.get("data_root", "data/M3Dsynth"))
    metadata_dir = Path(config.get("metadata_dir", "metadata/m3dsynth"))
    if not metadata_dir.is_absolute():
        metadata_dir = Path.cwd() / metadata_dir
    if not metadata_dir.exists():
        raise FileNotFoundError(
            f"metadata directory not found: {metadata_dir}. "
            "The repository includes the official CSV files under metadata/m3dsynth. "
            "Set data_root to the downloaded TIFF dataset folder containing cycle/, pix2pix/, diffusion/, real/."
        )

    split_cfg = config.get("split", {})
    data_cfg = config.get("data", {}) or {}
    patch_cfg = config.get("patches", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})

    model_name = model_cfg.get("name", "simple_3d_cnn")
    if model_name not in KNOWN_MODEL_NAMES:
        raise ValueError(f"unknown model.name {model_name!r}; known names: {sorted(KNOWN_MODEL_NAMES)}")

    records = read_records(metadata_dir)
    test_mods = split_cfg.get("test_mods")
    if test_mods is None:
        # Keep old leave-generator-out configs working: heldout_mod becomes a one-item test list.
        test_mods = [split_cfg["heldout_mod"]]
    groups = cross_generator_records(
        records,
        train_mods=split_cfg["train_mods"],
        valid_mods=split_cfg.get("valid_mods", split_cfg["train_mods"]),
        test_mods=test_mods,
        train_splits=(split_cfg.get("train_set", "train"),),
        valid_splits=(split_cfg.get("valid_set", "valid"),),
        test_splits=(split_cfg.get("test_set", "test"),),
        include_real=bool(split_cfg.get("include_real", True)),
    )

    subset_seed = int(data_cfg.get("record_subset_seed", training_cfg.get("seed", 21)))
    train_records = subset_records(groups["train"], data_cfg.get("max_train_records"), subset_seed)
    valid_records = subset_records(groups["valid"], data_cfg.get("max_valid_records"), subset_seed)

    patch_shape = tuple(patch_cfg.get("patch_shape", (32, 32, 32)))
    train_stride = tuple(patch_cfg.get("train_stride", patch_shape))
    positive_overlap = float(patch_cfg.get("positive_overlap_fraction", 0.05))
    batch_size = int(training_cfg.get("batch_size", 8))
    volume_cache_size = int(training_cfg.get("volume_cache_size", 2))

    train_index = load_or_build_patch_index(
        train_records,
        data_root,
        cache_dir,
        patch_shape=patch_shape,
        stride=train_stride,
        positive_overlap_fraction=positive_overlap,
        rebuild=rebuild_index,
    )
    if train_index.n_positive == 0:
        raise ValueError(
            "the training patch index contains zero positive patches. "
            f"With patch_shape={patch_shape} and train_stride={train_stride} the grid samples "
            f"{np.prod([p / s for p, s in zip(patch_shape, train_stride)]):.1%} of each volume "
            "and misses every manipulation. Reduce train_stride (32 works) and subset the data "
            "with data.max_train_records instead."
        )

    train_dataset = M3DSynthPatchDataset(
        train_records,
        data_root=data_root,
        examples=train_index,
        patch_shape=patch_shape,
        stride=train_stride,
        volume_cache_size=volume_cache_size,
    )
    train_volume_keys = [train_dataset.volume_key(i) for i in range(len(train_dataset))]
    train_sampler = VolumeGroupedBatchSampler(
        train_volume_keys,
        train_dataset.labels,
        batch_size=batch_size,
        patches_per_volume=patch_cfg.get("max_patches_per_volume"),
        neg_per_pos=float(patch_cfg.get("neg_per_pos", 5.0)),
        max_positives_per_volume=patch_cfg.get("max_positives_per_volume"),
        positive_patches_per_volume=patch_cfg.get("positive_patches_per_volume"),
        max_patches_per_epoch=training_cfg.get("max_patches_per_epoch"),
        max_volumes_per_epoch=training_cfg.get("max_volumes_per_epoch"),
        positive_volume_fraction=patch_cfg.get("positive_volume_fraction"),
        seed=int(training_cfg.get("seed", 21)),
    )
    pin_memory = str(device).startswith("cuda") and torch.cuda.is_available()
    train_loader = _build_dataloader(
        train_dataset, train_sampler, training_cfg, pin_memory=pin_memory
    )

    valid_loader = None
    valid_dataset = None
    if valid_records:
        valid_index = load_or_build_patch_index(
            valid_records,
            data_root,
            cache_dir,
            patch_shape=patch_shape,
            stride=train_stride,
            positive_overlap_fraction=positive_overlap,
            rebuild=rebuild_index,
        )
        valid_dataset = M3DSynthPatchDataset(
            valid_records,
            data_root=data_root,
            examples=valid_index,
            patch_shape=patch_shape,
            stride=train_stride,
            volume_cache_size=volume_cache_size,
        )
        valid_keys = [valid_dataset.volume_key(i) for i in range(len(valid_dataset))]
        # Natural, unbalanced order: a rebalanced validation set would report an
        # AP against a distribution the model never sees at inference.
        valid_sampler = SequentialVolumeBatchSampler(
            valid_keys,
            batch_size=batch_size,
            labels=valid_dataset.labels,
            max_patches_per_volume=patch_cfg.get("max_valid_patches_per_volume", 256),
            max_volumes=training_cfg.get("max_valid_volumes"),
            seed=int(training_cfg.get("seed", 21)),
        )
        valid_loader = _build_dataloader(
            valid_dataset, valid_sampler, training_cfg, pin_memory=pin_memory
        )

    model = build_patch3d_classifier(
        Patch3DModelConfig(
            in_channels=int(model_cfg.get("in_channels", 1)),
            num_classes=int(model_cfg.get("num_classes", 1)),
            base_channels=int(model_cfg.get("base_channels", 16)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
    )
    loss = build_loss(training_cfg.get("loss", "focal"))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    use_amp, amp_dtype, scaler = _amp_settings(training_cfg, device)
    return TrainingObjects(
        model=model,
        train_loader=train_loader,
        train_sampler=train_sampler,
        train_dataset=train_dataset,
        valid_loader=valid_loader,
        valid_dataset=valid_dataset,
        loss=loss,
        optimizer=optimizer,
        scaler=scaler,
        amp_dtype=amp_dtype,
        use_amp=use_amp,
    )


def build_training_objects(config: dict[str, Any], data_root_override: str | None = None):
    """Build model, train dataset, loss, optimizer, and DataLoader from config.

    Backward-compatible 4-tuple wrapper over :func:`build_loaders`.
    """

    objects = build_loaders(config, data_root_override=data_root_override)
    return objects.model, objects.train_loader, objects.loss, objects.optimizer


def train_one_epoch(
    model,
    loader,
    loss_fn,
    optimizer,
    device: str = "cpu",
    scaler=None,
    amp_dtype=None,
    use_amp: bool = False,
    grad_clip_norm: float | None = None,
) -> float:
    """Train model for one epoch and return mean training loss."""

    try:
        import torch
        from tqdm import tqdm
    except ImportError as exc:  # pragma: no cover - optional train env
        raise RuntimeError("PyTorch and tqdm are required for training") from exc

    model.to(device)
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    model.train()
    losses: list[float] = []
    for batch in tqdm(loader, desc="    batch", leave=False, disable=False):
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=bool(use_amp)):
            logits = model(image)
            loss = loss_fn(logits, label)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def evaluate_patch_loader(
    model,
    loader,
    loss_fn,
    device: str = "cpu",
    amp_dtype=None,
    use_amp: bool = False,
) -> dict[str, float]:
    """Run one validation pass and return patch-level metrics."""

    try:
        import torch
        from tqdm import tqdm
    except ImportError as exc:  # pragma: no cover - optional train env
        raise RuntimeError("PyTorch and tqdm are required for evaluation") from exc

    model.to(device)
    model.eval()
    losses: list[float] = []
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="    valid", leave=False, disable=False):
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=bool(use_amp)):
                logits = model(image)
                loss = loss_fn(logits, label)
            losses.append(float(loss.detach().cpu()))
            all_scores.append(torch.sigmoid(logits.float()).detach().cpu().numpy().reshape(-1))
            all_labels.append(label.detach().cpu().numpy().reshape(-1))

    if not all_labels:
        raise ValueError("validation loader yielded no patches")

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    if np.unique(labels).size != 2:
        raise ValueError(
            "validation must contain both positive and negative patches; "
            f"received n={len(labels)}, n_pos={int(np.count_nonzero(labels))}"
        )
    auc, ba = volume_auc_ba(labels, scores, threshold=0.5)
    ap = float("nan")
    if np.unique(labels).size == 2:
        try:
            from sklearn.metrics import average_precision_score

            ap = float(average_precision_score(labels, scores))
        except ImportError:  # pragma: no cover - depends on optional env
            pass
    return {
        "loss": float(np.mean(losses)),
        "auc": float(auc),
        "ap": ap,
        "balanced_accuracy": float(ba),
        "n": float(len(labels)),
        "n_pos": float(np.count_nonzero(labels)),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for patch classifier training."""

    parser = argparse.ArgumentParser(description="Train a patch-wise 3D detector on M3Dsynth.")
    parser.add_argument("--config", required=True, help="Path to a cross-generator YAML config.")
    parser.add_argument("--data-root", help="Override dataset root from config.")
    parser.add_argument("--output-dir", help="Override output directory from config.")
    parser.add_argument("--cache-dir", help="Where to store the patch index cache.")
    parser.add_argument("--rebuild-index", action="store_true", help="Ignore any cached patch index.")
    parser.add_argument(
        "--resume",
        help="Resume from a checkpoint saved by this trainer (model, optimizer, AMP scaler, epoch).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the index and sampler, print dataset statistics, and exit without training.",
    )
    parser.add_argument("--device", default="cpu", help="PyTorch device, for example cpu, cuda, or mps.")
    return parser.parse_args()


def _write_metrics_row(path: Path, row: dict[str, Any]) -> None:
    """Append one epoch of metrics to a CSV, writing the header on first use."""

    is_new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _load_checkpoint(path: str | Path, objects: TrainingObjects) -> tuple[int, float, int]:
    """Restore training state and return epoch, best AP, and stop counter.

    Older checkpoints containing only model weights remain usable: their
    optimizer is freshly initialized and their next epoch follows the saved one.
    """

    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6 has no weights_only keyword.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"checkpoint {path} does not contain model_state_dict")
    objects.model.load_state_dict(payload["model_state_dict"])
    if "optimizer_state_dict" in payload:
        objects.optimizer.load_state_dict(payload["optimizer_state_dict"])
    else:
        warnings.warn(
            "checkpoint has no optimizer state; resuming with a freshly initialized optimizer",
            RuntimeWarning,
            stacklevel=2,
        )
    if objects.scaler is not None and "scaler_state_dict" in payload:
        objects.scaler.load_state_dict(payload["scaler_state_dict"])
    next_epoch = int(payload.get("epoch", 0))
    best_ap = float(payload.get("best_ap", -float("inf")))
    epochs_without_improvement = int(payload.get("epochs_without_improvement", 0))
    return next_epoch, best_ap, epochs_without_improvement


def _checkpoint_payload(
    objects: TrainingObjects,
    config: dict[str, Any],
    epoch: int,
    best_ap: float,
    epochs_without_improvement: int = 0,
) -> dict[str, Any]:
    """Return a complete, resumable checkpoint payload."""

    payload: dict[str, Any] = {
        "model_state_dict": objects.model.state_dict(),
        "optimizer_state_dict": objects.optimizer.state_dict(),
        "config": config,
        "epoch": int(epoch),
        "best_ap": float(best_ap),
        "epochs_without_improvement": int(epochs_without_improvement),
    }
    if objects.scaler is not None:
        payload["scaler_state_dict"] = objects.scaler.state_dict()
    return payload


def _load_torch_payload(path: Path) -> dict[str, Any]:
    """Load a checkpoint compatibly across supported PyTorch releases."""

    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"checkpoint {path} does not contain model_state_dict")
    return payload


def _unique_validation_records(dataset: M3DSynthPatchDataset):
    """Return one metadata record per physical validation scan directory."""

    unique = {}
    for record in dataset.records:
        unique.setdefault(str(scan_dir(dataset.data_root, record).resolve()), record)
    return list(unique.values())


def _best_localization_from_counts(
    thresholds: np.ndarray,
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
) -> tuple[float, BinaryLocalizationMetrics]:
    """Select highest threshold among equal micro-F1 values."""

    denominator = 2.0 * tp + fp + fn
    f1 = np.divide(2.0 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    index = int(np.flatnonzero(f1 == np.max(f1))[-1])
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    precision = 1.0 if precision_denominator[index] == 0 else float(tp[index] / precision_denominator[index])
    recall = 1.0 if recall_denominator[index] == 0 else float(tp[index] / recall_denominator[index])
    iou_denominator = tp[index] + fp[index] + fn[index]
    iou = 1.0 if iou_denominator == 0 else float(tp[index] / iou_denominator)
    metrics = BinaryLocalizationMetrics(precision, recall, float(f1[index]), float(f1[index]), iou)
    return float(thresholds[index]), metrics


def _detection_candidates(heatmap: np.ndarray, fractions: list[float]) -> dict[str, float]:
    """Return max and top-k volume scores from one heatmap."""

    scores = {"max": max_detection_score(heatmap)}
    scores.update({f"topk_{fraction:g}": topk_detection_score(heatmap, fraction) for fraction in fractions})
    return scores


def _render_validation_example(
    path: Path,
    scan: np.ndarray,
    heatmap: np.ndarray,
    truth: np.ndarray,
    localization_threshold: float,
    title: str,
) -> None:
    """Save suspicious and, when available, ground-truth axial slices."""

    from PIL import Image, ImageDraw

    labels = ["CT", "heatmap", "ground truth", "prediction"]
    suspicious_z = int(np.argmax(np.max(heatmap, axis=(1, 2))))
    rows = [("max-score slice", suspicious_z)]
    truth_slices = np.flatnonzero(np.any(truth, axis=(1, 2)))
    if len(truth_slices):
        truth_z = int(truth_slices[len(truth_slices) // 2])
        if truth_z != suspicious_z:
            rows.append(("ground-truth slice", truth_z))

    height, width = scan.shape[1:]
    header = 42
    canvas = Image.new("RGB", (width * 4, header + height * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 2), title, fill="black")
    for index, label in enumerate(labels):
        draw.text((index * width + 4, 22), label, fill="black")
    for row_index, (row_name, z) in enumerate(rows):
        gray = np.clip(scan[z] * 255.0, 0, 255).astype(np.uint8)
        base = np.repeat(gray[..., None], 3, axis=2)
        heat = np.clip(heatmap[z], 0.0, 1.0)
        heat_overlay = base.astype(np.float32)
        heat_overlay[..., 0] = np.maximum(heat_overlay[..., 0], heat * 255.0)
        truth_overlay = base.copy()
        truth_overlay[truth[z].astype(bool)] = (0, 255, 0)
        prediction_overlay = base.copy()
        prediction_overlay[heatmap[z] >= localization_threshold] = (255, 0, 0)
        panels = [base, np.clip(heat_overlay, 0, 255).astype(np.uint8), truth_overlay, prediction_overlay]
        y = header + row_index * height
        for column, panel in enumerate(panels):
            canvas.paste(Image.fromarray(panel), (column * width, y))
        draw.text((4, y + 4), f"{row_name} z={z}", fill="white", stroke_width=1, stroke_fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _save_validation_report(
    report_dir: Path,
    records,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    localization_threshold: float,
    valid_dataset: M3DSynthPatchDataset,
    model,
    patch_shape,
    stride,
    batch_size: int,
    aggregation: str,
    device: str,
    examples_per_class: int,
) -> None:
    """Save representative TP/FP/TN/FN heatmap overlays and a CSV index."""

    predictions = scores >= threshold
    categories = np.where(labels, np.where(predictions, "TP", "FN"), np.where(predictions, "FP", "TN"))
    rows = []
    for category in ("TP", "FP", "TN", "FN"):
        indices = np.flatnonzero(categories == category)
        if not len(indices):
            continue
        order = indices[np.argsort(scores[indices])]
        chosen = order[-examples_per_class:] if category in {"TP", "FP"} else order[:examples_per_class]
        for index in chosen:
            record = records[int(index)]
            scan = load_normalized_scan(valid_dataset.data_root, record)
            heatmap = infer_heatmap(
                model, scan, patch_shape=patch_shape, stride=stride,
                batch_size=batch_size, aggregation=aggregation, device=device,
            )
            truth = np.zeros(scan.shape, dtype=bool) if record.is_real else align_mask_to_scan(
                load_label_mask(label_dir(valid_dataset.data_root, record)), scan.shape, img_id=record.img_id
            )
            filename = f"{category}_{record.img_id}_{int(index):03d}.png".replace("/", "_").replace("\\", "_")
            _render_validation_example(
                report_dir / filename, scan, heatmap, truth, localization_threshold,
                f"{category} | {record.img_id} | score={scores[index]:.4f} | threshold={threshold:.4f}",
            )
            rows.append({"category": category, "img_id": record.img_id, "score": float(scores[index]), "file": filename})
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "img_id", "score", "file"])
        writer.writeheader()
        writer.writerows(rows)


def calibrate_best_checkpoint(
    best_path: Path,
    valid_dataset: M3DSynthPatchDataset,
    config: dict[str, Any],
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    """Calibrate volume and localization thresholds once on validation scans.

    Full-resolution heatmaps are handled one volume at a time. This keeps the
    calibration memory bounded while matching the exact inference path used by
    the CLI.
    """

    payload = _load_torch_payload(best_path)
    model_cfg = payload.get("config", config).get("model", {})
    model = build_patch3d_classifier(
        Patch3DModelConfig(
            in_channels=int(model_cfg.get("in_channels", 1)),
            num_classes=int(model_cfg.get("num_classes", 1)),
            base_channels=int(model_cfg.get("base_channels", 16)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)

    patch_cfg = config.get("patches", {})
    evaluation_cfg = config.get("evaluation", {})
    patch_shape = tuple(patch_cfg.get("patch_shape", (32, 32, 32)))
    stride = tuple(patch_cfg.get("inference_stride", (16, 16, 16)))
    aggregation = str(evaluation_cfg.get("heatmap_aggregation", "average"))
    thresholds = np.linspace(0.01, 0.99, int(evaluation_cfg.get("calibration_threshold_steps", 99)))
    tp = np.zeros(len(thresholds), dtype=np.float64)
    fp = np.zeros(len(thresholds), dtype=np.float64)
    fn = np.zeros(len(thresholds), dtype=np.float64)
    volume_labels: list[bool] = []
    topk_fractions = [float(value) for value in evaluation_cfg.get("topk_fractions", [0.001, 0.01])]
    volume_scores: dict[str, list[float]] = {"max": []}
    volume_scores.update({f"topk_{fraction:g}": [] for fraction in topk_fractions})
    voxel_auc: list[float] = []
    voxel_ap: list[float] = []

    records = _unique_validation_records(valid_dataset)
    for record in records:
        scan = load_normalized_scan(valid_dataset.data_root, record)
        heatmap = infer_heatmap(
            model, scan, patch_shape=patch_shape, stride=stride,
            batch_size=batch_size, aggregation=aggregation, device=device,
        )
        if record.is_real:
            mask = np.zeros(scan.shape, dtype=bool)
        else:
            mask = align_mask_to_scan(
                load_label_mask(label_dir(valid_dataset.data_root, record)), scan.shape, img_id=record.img_id
            )
        for index, threshold in enumerate(thresholds):
            prediction = heatmap >= threshold
            tp[index] += np.logical_and(mask, prediction).sum(dtype=np.float64)
            fp[index] += np.logical_and(~mask, prediction).sum(dtype=np.float64)
            fn[index] += np.logical_and(mask, ~prediction).sum(dtype=np.float64)
        if np.any(mask):
            auc, ap = voxel_auc_ap(mask, heatmap)
            voxel_auc.append(auc)
            voxel_ap.append(ap)
        volume_labels.append(record.is_manipulated)
        for name, score in _detection_candidates(heatmap, topk_fractions).items():
            volume_scores[name].append(score)

    volume_labels_array = np.asarray(volume_labels, dtype=bool)
    candidate_metrics = {}
    for name, values in volume_scores.items():
        scores_array = np.asarray(values, dtype=np.float32)
        candidate_threshold, candidate_ba = best_threshold_by_balanced_accuracy(volume_labels_array, scores_array)
        candidate_auc, _ = volume_auc_ba(volume_labels_array, scores_array, candidate_threshold)
        candidate_metrics[name] = {
            "threshold": candidate_threshold,
            "balanced_accuracy": candidate_ba,
            "auc": candidate_auc,
        }
    requested_score = str(evaluation_cfg.get("detection_score", "max_heatmap"))
    if requested_score == "auto":
        selected_name = max(candidate_metrics, key=lambda name: (candidate_metrics[name]["auc"], candidate_metrics[name]["balanced_accuracy"]))
    elif requested_score in {"max", "max_heatmap"}:
        selected_name = "max"
    elif requested_score == "topk_mean":
        selected_name = f"topk_{float(evaluation_cfg.get('topk_fraction', 0.01)):g}"
        if selected_name not in candidate_metrics:
            raise ValueError(f"topk_fraction is not present in topk_fractions: {selected_name}")
    else:
        raise ValueError("evaluation.detection_score must be auto, max_heatmap, or topk_mean")
    selected_scores = np.asarray(volume_scores[selected_name], dtype=np.float32)
    selected_metrics = candidate_metrics[selected_name]
    detection_threshold = float(selected_metrics["threshold"])
    detection_ba = float(selected_metrics["balanced_accuracy"])
    detection_auc = float(selected_metrics["auc"])
    localization_threshold, localization_metrics = _best_localization_from_counts(thresholds, tp, fp, fn)
    examples_per_class = int(evaluation_cfg.get("report_examples_per_class", 0))
    if examples_per_class > 0:
        _save_validation_report(
            best_path.parent / f"validation_report_{aggregation}", records, volume_labels_array, selected_scores,
            detection_threshold, localization_threshold, valid_dataset, model, patch_shape, stride,
            batch_size, aggregation, device, examples_per_class,
        )
    selected_fraction = float(selected_name.removeprefix("topk_")) if selected_name.startswith("topk_") else None
    return {
        "format_version": 1,
        "checkpoint": best_path.name,
        "inference": {
            "patch_shape": [int(value) for value in patch_shape],
            "stride": [int(value) for value in stride],
            "aggregation": aggregation,
        },
        "classification": {
            "score_mode": "topk_mean" if selected_fraction is not None else "max",
            "topk_fraction": selected_fraction,
            "threshold": detection_threshold,
            "balanced_accuracy": detection_ba,
            "auc": detection_auc,
            "n_volumes": int(len(records)),
            "n_positive_volumes": int(np.count_nonzero(volume_labels_array)),
            "candidates": candidate_metrics,
        },
        "localization": {
            "threshold": localization_threshold,
            **localization_metrics.to_dict(),
            "mean_voxel_auc_on_positive_volumes": float(np.nanmean(voxel_auc)),
            "mean_voxel_ap_on_positive_volumes": float(np.nanmean(voxel_ap)),
            "n_positive_volumes": int(np.count_nonzero(volume_labels_array)),
        },
    }


def main() -> None:
    """Run supervised training with per-epoch validation and checkpoints."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional train env
        raise RuntimeError("PyTorch is required for training") from exc

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    args = parse_args()
    config = load_yaml_config(args.config)
    training_cfg = config.get("training", {})
    seed = int(training_cfg.get("seed", 21))
    set_global_seed(seed)

    default_output = config.get("output_dir") or f"outputs/{config.get('experiment_name', 'train')}"
    output_dir = Path(args.output_dir or default_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = resolve_cache_dir(config, args.cache_dir, output_dir)

    print(f"Loading dataset and building model (cache: {cache_dir})...")
    objects = build_loaders(
        config,
        data_root_override=args.data_root,
        cache_dir=cache_dir,
        device=args.device,
        rebuild_index=args.rebuild_index,
    )

    stats = objects.train_sampler.describe()
    stats.update(
        {
            "train_patches_indexed": float(len(objects.train_dataset)),
            "train_positives_indexed": float(objects.train_dataset.examples.n_positive),
            "valid_patches_indexed": float(len(objects.valid_dataset)) if objects.valid_dataset else 0.0,
            "amp": float(objects.use_amp),
        }
    )
    print(json.dumps({k: round(v, 4) for k, v in stats.items()}, indent=2))
    if args.dry_run:
        (output_dir / "dry_run_stats.json").write_text(json.dumps(stats, indent=2))
        print("dry run complete; no training performed")
        return

    epochs = int(training_cfg.get("epochs", 1))
    grad_clip = training_cfg.get("grad_clip_norm")
    metrics_path = output_dir / "metrics.csv"
    best_path = output_dir / "best.pt"
    best_ap = -float("inf")
    epochs_without_improvement = 0
    start_epoch = 0
    if args.resume:
        start_epoch, best_ap, epochs_without_improvement = _load_checkpoint(args.resume, objects)
        if start_epoch >= epochs:
            raise ValueError(
                f"checkpoint is already at epoch {start_epoch}, but training.epochs is only {epochs}"
            )
        print(f"Resumed {args.resume} at epoch {start_epoch + 1}/{epochs}.")

    early_stopping_patience = training_cfg.get("early_stopping_patience")
    early_stopping_patience = (
        None if early_stopping_patience is None else int(early_stopping_patience)
    )
    epoch_iter = tqdm(range(start_epoch, epochs), desc="epoch") if tqdm else range(start_epoch, epochs)
    for epoch in epoch_iter:
        objects.train_sampler.set_epoch(epoch)
        mean_loss = train_one_epoch(
            objects.model,
            objects.train_loader,
            objects.loss,
            objects.optimizer,
            device=args.device,
            scaler=objects.scaler,
            amp_dtype=objects.amp_dtype,
            use_amp=objects.use_amp,
            grad_clip_norm=grad_clip,
        )
        valid_metrics: dict[str, float] = {}
        if objects.valid_loader is not None:
            valid_metrics = evaluate_patch_loader(
                objects.model,
                objects.valid_loader,
                objects.loss,
                device=args.device,
                amp_dtype=objects.amp_dtype,
                use_amp=objects.use_amp,
            )

        row = {"epoch": epoch + 1, "train_loss": mean_loss}
        row.update({f"valid_{k}": v for k, v in valid_metrics.items()})
        _write_metrics_row(metrics_path, row)
        if tqdm:
            epoch_iter.set_postfix({"loss": f"{mean_loss:.6f}", "val_ap": f"{valid_metrics.get('ap', float('nan')):.4f}"})
        else:
            print(f"epoch={epoch + 1}, train_loss={mean_loss:.6f}, valid={valid_metrics}")

        current_ap = valid_metrics.get("ap", float("nan"))
        # Keep a usable best checkpoint even when AP is unavailable (for example
        # when scikit-learn is absent), then replace it whenever AP improves.
        improved = not best_path.exists() or (
            current_ap == current_ap and current_ap > best_ap
        )
        if improved:
            best_ap = current_ap
            epochs_without_improvement = 0
        elif current_ap == current_ap:
            epochs_without_improvement += 1

        payload = _checkpoint_payload(
            objects, config, epoch + 1, best_ap, epochs_without_improvement
        )
        if bool(training_cfg.get("checkpoint_every_epoch", True)):
            torch.save(payload, output_dir / f"checkpoint_epoch{epoch + 1:03d}.pt")
        if improved:
            torch.save(payload, best_path)
        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                f"Early stopping after {epochs_without_improvement} epochs without validation AP improvement."
            )
            break

    checkpoint = output_dir / "patch3d_classifier.pt"
    torch.save(
        _checkpoint_payload(objects, config, epoch + 1, best_ap, epochs_without_improvement),
        checkpoint,
    )
    if objects.valid_dataset is not None and bool(config.get("evaluation", {}).get("calibrate_best", True)):
        calibration = calibrate_best_checkpoint(
            best_path,
            objects.valid_dataset,
            config,
            device=args.device,
            batch_size=int(config.get("evaluation", {}).get("calibration_batch_size", training_cfg.get("batch_size", 8))),
        )
        calibration_path = output_dir / "calibration.json"
        calibration_path.write_text(json.dumps(calibration, indent=2) + "\n")
        print(
            "Calibrated validation thresholds: "
            f"detection_score={calibration['classification']['score_mode']}, "
            f"detection={calibration['classification']['threshold']:.6f}, "
            f"localization={calibration['localization']['threshold']:.6f}"
        )
    print(f"\nSaved checkpoint to {checkpoint}")
    print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
