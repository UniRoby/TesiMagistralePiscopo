"""Voxel-level 3D U-Net baseline training on pix2pix manipulations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from tqdm import tqdm

from .dataset import M3DSynthSegmentationPatchDataset, cross_generator_records, read_records
from .losses import SegmentationBCEDiceLoss, SegmentationFocalDiceLoss
from .model import UNet3DModelConfig, build_unet3d
from .patch_index import load_or_build_patch_index
from .sampling import SequentialVolumeBatchSampler, VolumeGroupedBatchSampler
from .train import (
    _amp_settings,
    _build_dataloader,
    _write_metrics_row,
    load_yaml_config,
    resolve_cache_dir,
    set_global_seed,
    subset_records,
)

EARLY_STOPPING_PATIENCE = 5


class UNetTrainingObjects(NamedTuple):
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


def build_unet_loaders(
    config: dict[str, Any],
    data_root_override: str | None = None,
    cache_dir: str | Path | None = None,
    device: str = "cpu",
    rebuild_index: bool = False,
) -> UNetTrainingObjects:
    """Build pix2pix segmentation datasets, loaders, model and optimizer."""

    import torch

    data_root = Path(data_root_override or config.get("data_root", "data/M3Dsynth"))
    metadata_dir = Path(config.get("metadata_dir", "metadata/m3dsynth"))
    if not metadata_dir.is_absolute():
        metadata_dir = Path.cwd() / metadata_dir

    split_cfg = config["split"]
    if split_cfg.get("train_mods") != ["pix2pix"] or split_cfg.get("valid_mods") != ["pix2pix"]:
        raise ValueError("the baseline requires split.train_mods and split.valid_mods to be [pix2pix]")
    groups = cross_generator_records(
        read_records(metadata_dir),
        train_mods=["pix2pix"],
        valid_mods=["pix2pix"],
        test_mods=split_cfg.get("test_mods", ["cycle", "diffusion"]),
        train_splits=(split_cfg.get("train_set", "train"),),
        valid_splits=(split_cfg.get("valid_set", "valid"),),
        test_splits=(split_cfg.get("test_set", "test"),),
        include_real=bool(split_cfg.get("include_real", True)),
    )

    data_cfg = config.get("data", {})
    patch_cfg = config.get("patches", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    if model_cfg.get("name", "unet3d") != "unet3d":
        raise ValueError("model.name must be unet3d")
    seed = int(training_cfg.get("seed", 21))
    subset_seed = int(data_cfg.get("record_subset_seed", seed))
    train_records = subset_records(groups["train"], data_cfg.get("max_train_records"), subset_seed)
    valid_records = subset_records(groups["valid"], data_cfg.get("max_valid_records"), subset_seed)

    patch_shape = tuple(int(v) for v in patch_cfg.get("patch_shape", (64, 64, 64)))
    if any(size % 8 for size in patch_shape):
        raise ValueError("patches.patch_shape dimensions must be divisible by 8")
    stride = tuple(int(v) for v in patch_cfg.get("train_stride", (32, 32, 32)))
    overlap = float(patch_cfg.get("positive_overlap_fraction", 1e-6))
    batch_size = int(training_cfg.get("batch_size", 2))
    cache_size = int(training_cfg.get("volume_cache_size", 1))

    train_index = load_or_build_patch_index(
        train_records, data_root, cache_dir, patch_shape=patch_shape, stride=stride,
        positive_overlap_fraction=overlap,
        centered_positive_crops=int(patch_cfg.get("centered_positive_crops_per_volume", 0)),
        positive_crop_jitter=int(patch_cfg.get("positive_crop_jitter", 0)),
        positive_crop_seed=seed,
        rebuild=rebuild_index,
    )
    valid_index = load_or_build_patch_index(
        valid_records, data_root, cache_dir, patch_shape=patch_shape, stride=stride,
        positive_overlap_fraction=overlap, rebuild=rebuild_index,
    )
    if train_index.n_positive == 0 or valid_index.n_positive == 0:
        raise ValueError("train and validation indices must contain positive mask patches")

    dataset_kwargs = {
        "data_root": data_root,
        "patch_shape": patch_shape,
        "stride": stride,
        "volume_cache_size": cache_size,
    }
    train_dataset = M3DSynthSegmentationPatchDataset(train_records, examples=train_index, **dataset_kwargs)
    valid_dataset = M3DSynthSegmentationPatchDataset(valid_records, examples=valid_index, **dataset_kwargs)
    train_keys = [train_dataset.volume_key(index) for index in range(len(train_dataset))]
    valid_keys = [valid_dataset.volume_key(index) for index in range(len(valid_dataset))]

    train_sampler = VolumeGroupedBatchSampler(
        train_keys,
        train_dataset.labels,
        batch_size=batch_size,
        patches_per_volume=patch_cfg.get("max_patches_per_volume", 8),
        positive_patches_per_volume=patch_cfg.get("positive_patches_per_volume", 4),
        positive_volume_fraction=patch_cfg.get("positive_volume_fraction", 0.8),
        max_patches_per_epoch=training_cfg.get("max_patches_per_epoch"),
        seed=seed,
    )
    valid_sampler = SequentialVolumeBatchSampler(
        valid_keys,
        batch_size=batch_size,
        labels=valid_dataset.labels,
        max_patches_per_volume=patch_cfg.get("max_valid_patches_per_volume", 32),
        max_volumes=training_cfg.get("max_valid_volumes"),
        seed=seed,
    )
    pin_memory = str(device).startswith("cuda") and torch.cuda.is_available()
    train_loader = _build_dataloader(train_dataset, train_sampler, training_cfg, pin_memory=pin_memory)
    valid_loader = _build_dataloader(valid_dataset, valid_sampler, training_cfg, pin_memory=pin_memory)

    model = build_unet3d(
        UNet3DModelConfig(
            in_channels=int(model_cfg.get("in_channels", 1)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            base_channels=int(model_cfg.get("base_channels", 16)),
            input_mode=str(model_cfg.get("input_mode", "ct")),
        )
    )
    loss_name = str(training_cfg.get("loss", "bce_dice"))
    if loss_name == "bce_dice":
        loss = SegmentationBCEDiceLoss()
    elif loss_name == "focal_dice":
        loss = SegmentationFocalDiceLoss(alpha=0.75, gamma=2.0)
    else:
        raise ValueError("training.loss must be 'bce_dice' or 'focal_dice'")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-5)),
    )
    use_amp, amp_dtype, scaler = _amp_settings(training_cfg, device)
    return UNetTrainingObjects(
        model, train_loader, train_sampler, train_dataset, valid_loader,
        valid_dataset, loss, optimizer, scaler, amp_dtype, use_amp,
    )


def train_one_epoch(objects: UNetTrainingObjects, device: str) -> float:
    """Train for one epoch and return mean BCE+Dice loss."""

    import torch

    objects.model.train()
    losses: list[float] = []
    for batch in tqdm(objects.train_loader, desc="    train", unit="batch", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        objects.optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=objects.amp_dtype, enabled=objects.use_amp):
            loss = objects.loss(objects.model(image), target)
        if objects.scaler.is_enabled():
            objects.scaler.scale(loss).backward()
            objects.scaler.step(objects.optimizer)
            objects.scaler.update()
        else:
            loss.backward()
            objects.optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def evaluate(objects: UNetTrainingObjects, device: str) -> dict[str, float]:
    """Return validation loss and global voxel-level overlap metrics."""

    import torch

    objects.model.eval()
    losses: list[float] = []
    tp = fp = fn = 0
    soft_intersection = soft_denominator = 0.0
    positive_probability_sum = negative_probability_sum = 0.0
    positive_voxels = negative_voxels = 0
    with torch.no_grad():
        for batch in tqdm(objects.valid_loader, desc="    valid", unit="batch", leave=False):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=objects.amp_dtype, enabled=objects.use_amp):
                probabilities = objects.model(image)
                loss = objects.loss(probabilities, target)
            prediction = probabilities >= 0.5
            truth = target >= 0.5
            tp += int(torch.logical_and(prediction, truth).sum())
            fp += int(torch.logical_and(prediction, ~truth).sum())
            fn += int(torch.logical_and(~prediction, truth).sum())
            soft_intersection += float((probabilities * target).sum())
            soft_denominator += float(probabilities.sum() + target.sum())
            positive_probability_sum += float(probabilities[truth].sum())
            negative_probability_sum += float(probabilities[~truth].sum())
            positive_voxels += int(truth.sum())
            negative_voxels += int((~truth).sum())
            losses.append(float(loss.detach().cpu()))
    dice_denominator = 2 * tp + fp + fn
    iou_denominator = tp + fp + fn
    return {
        "loss": float(np.mean(losses)),
        "dice": 1.0 if dice_denominator == 0 else 2.0 * tp / dice_denominator,
        "iou": 1.0 if iou_denominator == 0 else tp / iou_denominator,
        "precision": 1.0 if tp + fp == 0 else tp / (tp + fp),
        "recall": 1.0 if tp + fn == 0 else tp / (tp + fn),
        "soft_dice": (2.0 * soft_intersection + 1e-6) / (soft_denominator + 1e-6),
        "mean_positive_probability": positive_probability_sum / max(positive_voxels, 1),
        "mean_negative_probability": negative_probability_sum / max(negative_voxels, 1),
    }


def checkpoint_payload(objects: UNetTrainingObjects, config: dict[str, Any], epoch: int, best_score: float, stale_epochs: int, selection_metric: str):
    """Return model and training state for one checkpoint."""

    return {
        "model_state_dict": objects.model.state_dict(),
        "optimizer_state_dict": objects.optimizer.state_dict(),
        "scaler_state_dict": objects.scaler.state_dict(),
        "config": config,
        "epoch": epoch,
        "best_score": best_score,
        "selection_metric": selection_metric,
        "epochs_without_improvement": stale_epochs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the pix2pix 3D U-Net segmentation baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--cache-dir")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--micro-overfit", action="store_true", help="Overfit two positive patches and exit.")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def micro_overfit(objects: UNetTrainingObjects, device: str, max_steps: int = 300) -> float:
    """Overfit two positive patches; return fixed-threshold Dice."""

    import torch

    positive = np.flatnonzero(objects.train_dataset.labels)
    if len(positive) < 2:
        raise ValueError("micro-overfit requires at least two positive patches")
    positive = positive[np.argsort(objects.train_dataset.examples.soft_score[positive])[-2:]]
    samples = [objects.train_dataset[int(index)] for index in positive]
    image = torch.stack([sample["image"] for sample in samples]).to(device)
    target = torch.stack([sample["mask"] for sample in samples]).to(device)
    dice = 0.0
    for group in objects.optimizer.param_groups:
        group["lr"] = max(float(group["lr"]), 1e-3)
    objects.model.train()
    for _ in tqdm(range(max_steps), desc="micro-overfit", unit="step"):
        objects.optimizer.zero_grad(set_to_none=True)
        probabilities = objects.model(image)
        loss = objects.loss(probabilities, target)
        loss.backward()
        objects.optimizer.step()
        prediction = probabilities.detach() >= 0.5
        truth = target >= 0.5
        tp = int(torch.logical_and(prediction, truth).sum())
        fp = int(torch.logical_and(prediction, ~truth).sum())
        fn = int(torch.logical_and(~prediction, truth).sum())
        dice = 2.0 * tp / max(2 * tp + fp + fn, 1)
        if dice > 0.90:
            break
    return dice


def main() -> None:
    """Train, validate and stop after five stale selection-metric epochs."""

    import torch

    args = parse_args()
    config = load_yaml_config(args.config)
    training_cfg = config.get("training", {})
    patience = int(training_cfg.get("early_stopping_patience", EARLY_STOPPING_PATIENCE))
    if patience != EARLY_STOPPING_PATIENCE:
        raise ValueError("training.early_stopping_patience must be 5 for this baseline")
    set_global_seed(int(training_cfg.get("seed", 21)))

    output_dir = Path(args.output_dir or config.get("output_dir", "outputs/train_pix2pix_unet_baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = resolve_cache_dir(config, args.cache_dir, output_dir)
    objects = build_unet_loaders(config, args.data_root, cache_dir, args.device, args.rebuild_index)
    objects.model.to(args.device)

    stats = {
        "train_patches": len(objects.train_dataset),
        "train_positive_patches": objects.train_dataset.examples.n_positive,
        "valid_patches": len(objects.valid_dataset),
        "valid_positive_patches": objects.valid_dataset.examples.n_positive,
    }
    positive_overlap = objects.train_dataset.examples.soft_score[objects.train_dataset.examples.label > 0]
    stats["train_positive_overlap_fraction"] = {
        name: float(value) for name, value in zip(
            ("min", "p25", "median", "p75", "max"),
            np.quantile(positive_overlap, (0.0, 0.25, 0.5, 0.75, 1.0)),
        )
    }
    stats.update(objects.train_sampler.describe())
    print(json.dumps(stats, indent=2))
    if args.dry_run:
        (output_dir / "dry_run_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        return
    if args.micro_overfit:
        dice = micro_overfit(objects, args.device)
        print(f"micro_overfit_dice={dice:.4f}")
        if dice <= 0.90:
            raise SystemExit("Micro-overfit failed: Dice did not exceed 0.90; do not start full training.")
        return

    selection_metric = "soft_dice" if training_cfg.get("loss") == "focal_dice" else "dice"
    best_score = -1.0
    stale_epochs = 0
    last_epoch = 0
    for epoch in range(int(training_cfg.get("epochs", 100))):
        last_epoch = epoch + 1
        objects.train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(objects, args.device)
        metrics = evaluate(objects, args.device)
        _write_metrics_row(
            output_dir / "metrics.csv",
            {"epoch": last_epoch, "train_loss": train_loss, **{f"valid_{key}": value for key, value in metrics.items()}},
        )
        improved = metrics[selection_metric] > best_score
        if improved:
            best_score = metrics[selection_metric]
            stale_epochs = 0
            torch.save(checkpoint_payload(objects, config, last_epoch, best_score, stale_epochs, selection_metric), output_dir / "best.pt")
        else:
            stale_epochs += 1
        print(
            f"epoch={last_epoch} train_loss={train_loss:.6f} "
            f"valid_soft_dice={metrics['soft_dice']:.4f} valid_dice={metrics['dice']:.4f}"
        )
        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping: validation {selection_metric} did not improve for 5 epochs.")
            break

    torch.save(
        checkpoint_payload(objects, config, last_epoch, best_score, stale_epochs, selection_metric),
        output_dir / "unet3d_last.pt",
    )


if __name__ == "__main__":
    main()
