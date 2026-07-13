"""Training entrypoint for patch-wise 3D M3Dsynth experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import M3DSynthPatchDataset, cross_generator_records, read_records
from .losses import build_loss
from .model import Patch3DModelConfig, build_patch3d_classifier


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


def make_balanced_sampler(dataset: M3DSynthPatchDataset):
    """Create a WeightedRandomSampler with equal total weight for pos/neg labels."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional train env
        raise RuntimeError("PyTorch is required for training sampler") from exc

    labels = np.asarray([example.label for example in dataset.examples], dtype=np.int64)
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


def build_training_objects(config: dict[str, Any], data_root_override: str | None = None):
    """Build model, train dataset, loss, optimizer, and DataLoader from config."""

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
    patch_cfg = config.get("patches", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})

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

    patch_shape = tuple(patch_cfg.get("patch_shape", (32, 32, 32)))
    train_stride = tuple(patch_cfg.get("train_stride", patch_shape))
    train_dataset = M3DSynthPatchDataset(
        groups["train"],
        data_root=data_root,
        patch_shape=patch_shape,
        stride=train_stride,
        positive_overlap_fraction=float(patch_cfg.get("positive_overlap_fraction", 0.05)),
    )
    sampler = make_balanced_sampler(train_dataset)
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(training_cfg.get("batch_size", 8)),
        sampler=sampler,
        num_workers=int(training_cfg.get("num_workers", 0)),
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
    return model, loader, loss, optimizer


def train_one_epoch(model, loader, loss_fn, optimizer, device: str = "cpu") -> float:
    """Train model for one epoch and return mean training loss."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional train env
        raise RuntimeError("PyTorch is required for training") from exc

    model.to(device)
    model.train()
    losses: list[float] = []
    for batch in loader:
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = loss_fn(logits, label)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for patch classifier training."""

    parser = argparse.ArgumentParser(description="Train a patch-wise 3D detector on M3Dsynth.")
    parser.add_argument("--config", required=True, help="Path to a cross-generator YAML config.")
    parser.add_argument("--data-root", help="Override dataset root from config.")
    parser.add_argument("--output-dir", help="Override output directory from config.")
    parser.add_argument("--device", default="cpu", help="PyTorch device, for example cpu, cuda, or mps.")
    return parser.parse_args()


def main() -> None:
    """Run minimal supervised training and save the final checkpoint."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional train env
        raise RuntimeError("PyTorch is required for training") from exc

    args = parse_args()
    config = load_yaml_config(args.config)
    output_dir = Path(args.output_dir or config.get("output_dir", "outputs/train"))
    output_dir.mkdir(parents=True, exist_ok=True)
    model, loader, loss_fn, optimizer = build_training_objects(config, data_root_override=args.data_root)
    epochs = int(config.get("training", {}).get("epochs", 1))
    for epoch in range(epochs):
        mean_loss = train_one_epoch(model, loader, loss_fn, optimizer, device=args.device)
        print(f"epoch={epoch + 1}, train_loss={mean_loss:.6f}")
    checkpoint = output_dir / "patch3d_classifier.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint)
    print(f"Saved checkpoint to {checkpoint}")


if __name__ == "__main__":
    main()
