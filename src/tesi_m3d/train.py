"""Training entrypoint for patch-wise 3D M3Dsynth experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from .dataset import M3DSynthPatchDataset, cross_generator_records, read_records
from .evaluation import volume_auc_ba
from .losses import build_loss
from .model import Patch3DModelConfig, build_patch3d_classifier
from .patch_index import load_or_build_patch_index
from .sampling import SequentialVolumeBatchSampler, VolumeGroupedBatchSampler

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


def _build_dataloader(dataset, batch_sampler, training_cfg: dict[str, Any]):
    """Wrap ``dataset`` in a DataLoader driven by a volume-grouped batch sampler."""

    import torch

    num_workers = int(training_cfg.get("num_workers", 0))
    kwargs: dict[str, Any] = {"batch_sampler": batch_sampler, "num_workers": num_workers}
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
        max_patches_per_epoch=training_cfg.get("max_patches_per_epoch"),
        max_volumes_per_epoch=training_cfg.get("max_volumes_per_epoch"),
        seed=int(training_cfg.get("seed", 21)),
    )
    train_loader = _build_dataloader(train_dataset, train_sampler, training_cfg)

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
            max_patches_per_volume=patch_cfg.get("max_valid_patches_per_volume", 256),
            max_volumes=training_cfg.get("max_valid_volumes"),
            seed=int(training_cfg.get("seed", 21)),
        )
        valid_loader = _build_dataloader(valid_dataset, valid_sampler, training_cfg)

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
        return {"loss": float("nan"), "auc": float("nan"), "ap": float("nan"),
                "balanced_accuracy": float("nan"), "n": 0.0, "n_pos": 0.0}

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
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
    best_ap = -float("inf")

    epoch_iter = tqdm(range(epochs), desc="epoch") if tqdm else range(epochs)
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

        payload = {"model_state_dict": objects.model.state_dict(), "config": config, "epoch": epoch + 1}
        if bool(training_cfg.get("checkpoint_every_epoch", True)):
            torch.save(payload, output_dir / f"checkpoint_epoch{epoch + 1:03d}.pt")
        current_ap = valid_metrics.get("ap", float("nan"))
        if current_ap == current_ap and current_ap > best_ap:  # skips NaN
            best_ap = current_ap
            torch.save(payload, output_dir / "best.pt")

    checkpoint = output_dir / "patch3d_classifier.pt"
    torch.save({"model_state_dict": objects.model.state_dict(), "config": config}, checkpoint)
    print(f"\nSaved checkpoint to {checkpoint}")
    print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
