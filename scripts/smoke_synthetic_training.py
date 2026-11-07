#!/usr/bin/env python3
"""Check MedicalNet fine-tuning and heatmap reconstruction on synthetic volumes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tesi_m3d.evaluation import binary_localization_metrics, threshold_heatmap
from tesi_m3d.inference import infer_heatmap
from tesi_m3d.losses import build_loss
from tesi_m3d.model import Patch3DModelConfig, build_patch3d_classifier
from tesi_m3d.synthetic import SyntheticPatchDataset, make_synthetic_volumes


def main() -> None:
    """Run two MedicalNet fine-tuning steps and reconstruct a synthetic heatmap."""

    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--medicalnet-weights", default=Patch3DModelConfig.medicalnet_weights)
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(21)
    out_dir = Path("outputs/synthetic_smoke_resnet18_medicalnet")
    out_dir.mkdir(parents=True, exist_ok=True)

    volumes = make_synthetic_volumes(generators=("pix2pix", "cycle", "diffusion"), volumes_per_generator=1)
    dataset = SyntheticPatchDataset(volumes, patch_shape=(32, 32, 32), stride=(16, 16, 16))
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    model_config = Patch3DModelConfig(medicalnet_weights=args.medicalnet_weights)
    model = build_patch3d_classifier(model_config).to(args.device)
    loss_fn = build_loss("bce")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    model.train()
    losses: list[float] = []
    for step, batch in zip(range(2), loader):
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["image"].to(args.device))
        loss = loss_fn(logits, batch["label"].to(args.device))
        loss.backward()
        assert model.conv1.weight.grad is not None and torch.isfinite(model.conv1.weight.grad).all()
        optimizer.step()
        losses.append(float(loss.detach()))
        print(f"step={step + 1}, train_loss={losses[-1]:.6f}")

    checkpoint = out_dir / "synthetic_patch3d.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": {"model": asdict(model_config)}, "losses": losses}, checkpoint)

    target = volumes[0]
    heatmap = infer_heatmap(
        model,
        target.volume,
        patch_shape=(32, 32, 32),
        stride=(16, 16, 16),
        batch_size=4,
        aggregation="average",
        device=args.device,
    )
    np.save(out_dir / "synthetic_heatmap.npy", heatmap)
    prediction = threshold_heatmap(heatmap, threshold=float(heatmap.mean()))
    metrics = binary_localization_metrics(target.mask, prediction)

    print(f"checkpoint={checkpoint}")
    print(f"heatmap_shape={heatmap.shape}, heatmap_min={heatmap.min():.6f}, heatmap_max={heatmap.max():.6f}")
    print(f"dice={metrics.dice:.6f}, iou={metrics.iou:.6f}")

    if not np.isfinite(losses).all() or not np.isfinite(heatmap).all():
        raise SystemExit("synthetic smoke failed: loss is not finite")


if __name__ == "__main__":
    main()
