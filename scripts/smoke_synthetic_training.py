#!/usr/bin/env python3
"""Run a tiny CPU training smoke test on synthetic 3D volumes."""

from __future__ import annotations

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
    """Train for two tiny epochs and save checkpoint/heatmap outputs."""

    import torch

    torch.manual_seed(21)
    out_dir = Path("outputs/synthetic_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)

    volumes = make_synthetic_volumes(generators=("pix2pix", "cycle", "diffusion"), volumes_per_generator=1)
    dataset = SyntheticPatchDataset(volumes, patch_shape=(32, 32, 32), stride=(16, 16, 16))
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    model = build_patch3d_classifier(Patch3DModelConfig(base_channels=4, num_classes=1))
    loss_fn = build_loss("bce")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    model.train()
    losses: list[float] = []
    for epoch in range(2):
        epoch_losses: list[float] = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["image"])
            loss = loss_fn(logits, batch["label"])
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        mean_loss = float(np.mean(epoch_losses))
        losses.append(mean_loss)
        print(f"epoch={epoch + 1}, train_loss={mean_loss:.6f}")

    checkpoint = out_dir / "synthetic_patch3d.pt"
    torch.save({"model_state_dict": model.state_dict(), "losses": losses}, checkpoint)

    target = volumes[0]
    heatmap = infer_heatmap(
        model,
        target.volume,
        patch_shape=(32, 32, 32),
        stride=(16, 16, 16),
        batch_size=4,
        aggregation="average",
        device="cpu",
    )
    np.save(out_dir / "synthetic_heatmap.npy", heatmap)
    prediction = threshold_heatmap(heatmap, threshold=float(heatmap.mean()))
    metrics = binary_localization_metrics(target.mask, prediction)

    print(f"checkpoint={checkpoint}")
    print(f"heatmap_shape={heatmap.shape}, heatmap_min={heatmap.min():.6f}, heatmap_max={heatmap.max():.6f}")
    print(f"dice={metrics.dice:.6f}, iou={metrics.iou:.6f}")

    if not np.isfinite(losses[-1]):
        raise SystemExit("synthetic smoke failed: loss is not finite")


if __name__ == "__main__":
    main()
