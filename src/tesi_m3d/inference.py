"""Inference helpers for patch-wise 3D heatmap reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from .patches import PatchGrid, extract_patches, reconstruct_heatmap


def predict_patch_scores(
    model,
    volume: np.ndarray,
    grid: PatchGrid,
    batch_size: int = 8,
    device: str | None = None,
) -> np.ndarray:
    """Run a patch classifier over a volume and return synthetic probabilities.

    Args:
        model: PyTorch module mapping ``(B,1,D,H,W)`` to one logit per patch.
        volume: Normalized CT volume with shape ``(D,H,W)``.
        grid: Sliding-window grid used to extract patches.
        batch_size: Number of patches processed per forward pass.
        device: Optional PyTorch device string such as ``"cuda"`` or ``"cpu"``.

    Returns:
        ``float32`` vector with one probability per patch in grid order.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on train env
        raise RuntimeError("PyTorch is required for model inference") from exc

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    patches = extract_patches(np.asarray(volume, dtype=np.float32), grid)
    target_device = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            batch_np = patches[start : start + batch_size]
            # Add channel dimension: (B,D,H,W) -> (B,1,D,H,W).
            batch = torch.from_numpy(batch_np[:, None]).to(target_device)
            logits = model(batch)
            # Classifier outputs logits; sigmoid converts them to synthetic probabilities.
            probs = torch.sigmoid(logits).reshape(-1).detach().cpu().numpy()
            scores.append(probs.astype(np.float32, copy=False))
    return np.concatenate(scores, axis=0)


def infer_heatmap(
    model,
    volume: np.ndarray,
    patch_shape: Sequence[int] = (32, 32, 32),
    stride: Sequence[int] = (16, 16, 16),
    batch_size: int = 8,
    aggregation: str = "average",
    device: str | None = None,
) -> np.ndarray:
    """Run patch-wise inference and reconstruct a 3D tampering heatmap."""

    volume = np.asarray(volume, dtype=np.float32)
    grid = PatchGrid(tuple(volume.shape), tuple(patch_shape), tuple(stride))
    scores = predict_patch_scores(model, volume, grid, batch_size=batch_size, device=device)
    return reconstruct_heatmap(scores, grid, mode=aggregation)


def reconstruct_from_scores_file(
    scores_path: str | Path,
    volume_shape: tuple[int, int, int],
    patch_shape: tuple[int, int, int] = (32, 32, 32),
    stride: tuple[int, int, int] = (16, 16, 16),
    aggregation: str = "average",
) -> np.ndarray:
    """Reconstruct a heatmap from saved patch scores for smoke tests."""

    loaded = np.load(scores_path)
    scores = loaded["scores"] if isinstance(loaded, np.lib.npyio.NpzFile) else loaded
    grid = PatchGrid(volume_shape=volume_shape, patch_shape=patch_shape, stride=stride)
    return reconstruct_heatmap(scores, grid, mode=aggregation)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for score-file reconstruction."""

    parser = argparse.ArgumentParser(description="Run patch-wise 3D inference or reconstruct a heatmap.")
    parser.add_argument("--scores", help="Optional .npy/.npz patch-score file for reconstruction smoke tests.")
    parser.add_argument("--volume-shape", nargs=3, type=int, metavar=("Z", "Y", "X"), default=(64, 64, 64))
    parser.add_argument("--patch-shape", nargs=3, type=int, metavar=("DZ", "DY", "DX"), default=(32, 32, 32))
    parser.add_argument("--stride", nargs=3, type=int, metavar=("SZ", "SY", "SX"), default=(16, 16, 16))
    parser.add_argument("--aggregation", choices=("average", "gaussian"), default="average")
    parser.add_argument("--out", default="outputs/heatmap.npy")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for reconstructing heatmaps from patch scores."""

    args = parse_args()
    if args.scores is None:
        raise SystemExit("Full model inference needs a checkpoint; pass --scores to test reconstruction.")
    heatmap = reconstruct_from_scores_file(
        args.scores,
        tuple(args.volume_shape),
        patch_shape=tuple(args.patch_shape),
        stride=tuple(args.stride),
        aggregation=args.aggregation,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, heatmap)
    print(f"Saved heatmap to {out}")


if __name__ == "__main__":
    main()
