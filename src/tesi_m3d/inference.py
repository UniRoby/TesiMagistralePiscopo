"""Inference helpers for patch-wise 3D heatmap reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .evaluation import max_detection_score, topk_detection_score
from .model import Patch3DModelConfig, build_patch3d_classifier
from .patches import PatchGrid, reconstruct_heatmap
from .dataset import load_tiff_stack, normalize_percentile


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
    volume = np.asarray(volume, dtype=np.float32)
    target_device = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        # Do not materialize every overlapping inference patch at once: a
        # 512x512 CT with stride 16 can otherwise consume multiple GB of RAM.
        slices = grid.iter_slices()
        while True:
            batch_slices = []
            try:
                for _ in range(batch_size):
                    batch_slices.append(next(slices))
            except StopIteration:
                pass
            if not batch_slices:
                break
            batch_np = np.stack([volume[slc] for slc in batch_slices], axis=0)
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
    parser.add_argument("--checkpoint", help="Trained .pt checkpoint produced by tesi_m3d.train.")
    parser.add_argument("--volume-dir", help="Directory containing slide0000.tiff, slide0001.tiff, ...")
    parser.add_argument("--volume-shape", nargs=3, type=int, metavar=("Z", "Y", "X"), default=(64, 64, 64))
    parser.add_argument("--patch-shape", nargs=3, type=int, metavar=("DZ", "DY", "DX"), default=(32, 32, 32))
    parser.add_argument("--stride", nargs=3, type=int, metavar=("SZ", "SY", "SX"), default=(16, 16, 16))
    parser.add_argument("--aggregation", choices=("average", "gaussian"), default="average")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu", help="PyTorch device, for example cuda or cpu.")
    parser.add_argument("--threshold", type=float, help="Override calibrated detection threshold.")
    parser.add_argument("--mask-out", help="Where to save the thresholded localization mask (.npy).")
    parser.add_argument("--out", default="outputs/heatmap.npy")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for reconstructing heatmaps from patch scores."""

    args = parse_args()
    detection_threshold = 0.5
    localization_threshold = 0.5
    detection_score_mode = "max"
    topk_fraction = None
    threshold_source = "default"
    if args.scores is not None:
        heatmap = reconstruct_from_scores_file(
            args.scores,
            tuple(args.volume_shape),
            patch_shape=tuple(args.patch_shape),
            stride=tuple(args.stride),
            aggregation=args.aggregation,
        )
    else:
        if not args.checkpoint or not args.volume_dir:
            raise SystemExit("Pass --scores, or both --checkpoint and --volume-dir for model inference.")
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on train env
            raise RuntimeError("PyTorch is required for model inference") from exc
        try:
            payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch < 2.6
            payload = torch.load(args.checkpoint, map_location="cpu")
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise SystemExit(f"Invalid checkpoint: {args.checkpoint}")
        calibration_path = Path(args.checkpoint).with_name("calibration.json")
        if calibration_path.exists():
            calibration = json.loads(calibration_path.read_text())
            try:
                detection_threshold = float(calibration["classification"]["threshold"])
                localization_threshold = float(calibration["localization"]["threshold"])
                detection_score_mode = str(calibration["classification"].get("score_mode", "max"))
                topk_fraction = calibration["classification"].get("topk_fraction")
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit(f"Invalid calibration file: {calibration_path}") from exc
            threshold_source = str(calibration_path)
        if args.threshold is not None:
            detection_threshold = float(args.threshold)
            threshold_source = "--threshold"
        checkpoint_config = payload.get("config", {})
        model_cfg = checkpoint_config.get("model", {})
        model = build_patch3d_classifier(
            Patch3DModelConfig(
                in_channels=int(model_cfg.get("in_channels", 1)),
                num_classes=int(model_cfg.get("num_classes", 1)),
                base_channels=int(model_cfg.get("base_channels", 16)),
                dropout=float(model_cfg.get("dropout", 0.2)),
            )
        )
        model.load_state_dict(payload["model_state_dict"])
        model.to(args.device)
        volume = normalize_percentile(load_tiff_stack(args.volume_dir))
        heatmap = infer_heatmap(
            model,
            volume,
            patch_shape=tuple(args.patch_shape),
            stride=tuple(args.stride),
            batch_size=args.batch_size,
            aggregation=args.aggregation,
            device=args.device,
        )
        score = (
            topk_detection_score(heatmap, float(topk_fraction))
            if detection_score_mode == "topk_mean" and topk_fraction is not None
            else max_detection_score(heatmap)
        )
        print(
            f"detection_score={score:.6f}, detection_score_mode={detection_score_mode}, "
            f"detection_threshold={detection_threshold:.6f}, "
            f"predicted_manipulated={score >= detection_threshold}, threshold_source={threshold_source}"
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, heatmap)
    print(f"Saved heatmap to {out}")
    mask_out = Path(args.mask_out) if args.mask_out else out.with_name(f"{out.stem}_mask.npy")
    np.save(mask_out, heatmap >= localization_threshold)
    print(f"Saved localization mask to {mask_out} (threshold={localization_threshold:.6f})")


if __name__ == "__main__":
    main()
