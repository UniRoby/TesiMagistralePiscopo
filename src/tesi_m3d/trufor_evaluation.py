"""Whole-volume inference and fixed-threshold, voxel-mask evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .trufor_data import atomic_json, digest, normalized_slice, read_slice, slice_input

THRESHOLDS = (np.arange(1, 100) / 100).astype(np.float32)


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def autocast(device):
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda")


@torch.inference_mode()
def predict_slices(model, entry, device):
    model.eval()
    for z in range(entry["shape"][0]):
        arrays = slice_input(entry, z, model.context)
        tensors = [torch.from_numpy(a).unsqueeze(0).to(device) for a in arrays]
        with autocast(device):
            logits = model(*tensors)
        probability = logits.float().softmax(dim=1)[0, 1].cpu().numpy()
        if not np.isfinite(probability).all():
            raise ValueError(f"Non-finite prediction in {entry['scan']}, slice {z}")
        yield z, probability


def threshold_counts(probability, mask):
    buckets = np.searchsorted(THRESHOLDS, probability, side="right")
    pos = np.bincount(buckets[mask], minlength=100)
    neg = np.bincount(buckets[~mask], minlength=100)
    tp = pos[::-1].cumsum()[::-1][1:]
    fp = neg[::-1].cumsum()[::-1][1:]
    return np.stack((tp, fp, mask.sum() - tp))


def metrics(counts):
    tp, fp, fn = np.asarray(counts, dtype=np.float64)
    dice_denom = 2 * tp + fp + fn
    iou_denom = tp + fp + fn
    return (np.divide(2 * tp, dice_denom, out=np.ones_like(tp), where=dice_denom != 0),
            np.divide(tp, iou_denom, out=np.ones_like(tp), where=iou_denom != 0))


def save_preview(path, entry, z, probability, mask):
    gray = np.uint8(normalized_slice(entry, z) * 255)
    original = np.repeat(gray[..., None], 3, axis=-1)
    truth = original.copy()
    truth[mask] = (0.5 * truth[mask] + np.array([0, 127, 0])).astype(np.uint8)
    predicted = original.copy()
    positive = probability >= 0.5
    predicted[positive] = (0.5 * predicted[positive] + np.array([127, 0, 0])).astype(np.uint8)
    Image.fromarray(np.concatenate([original, truth, predicted], axis=1)).save(path)


def evaluate_volumes(model, entries, device, output, identity, expired=lambda: False):
    """Cache completed volumes; an interrupted volume is recomputed on resume."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in entries:
        key = digest([identity, entry["signature"], entry["id"]])
        destination = output / f"{key}.json"
        if destination.exists():
            row = json.loads(destination.read_text())
        else:
            if expired():
                return None
            counts = np.zeros((3, 99), dtype=np.int64)
            best_preview, best_rank = None, -1
            for z, probability in predict_slices(model, entry, device):
                mask = (read_slice(entry["mask"], z) != 0 if entry["mask"] else
                        np.zeros(probability.shape, dtype=bool))
                counts += threshold_counts(probability, mask)
                rank = int(mask.sum()) if not entry["real"] else int((probability >= 0.5).sum())
                if rank > best_rank:
                    best_rank, best_preview = rank, (z, probability.copy(), mask.copy())
            z, probability, mask = best_preview
            save_preview(output / f"{key}.png", entry, z, probability, mask)
            row = dict(id=entry["id"], orig_id=entry["orig_id"], kind=entry["kind"],
                       real=entry["real"], counts=counts.tolist(),
                       voxels=int(np.prod(entry["shape"])), preview_z=z,
                       preview=str(output / f"{key}.png"))
            atomic_json(destination, row)
            print(f"Evaluated {entry['id']}", flush=True)
        rows.append(row)
    return rows


def select_threshold(rows):
    curves = [metrics(r["counts"])[0] for r in rows if not r["real"]]
    if not curves:
        raise ValueError("Calibration requires manipulated validation volumes")
    mean = np.mean(curves, axis=0)
    index = int(np.flatnonzero(mean == mean.max())[-1])
    return float(THRESHOLDS[index])


def summarize(rows, threshold):
    index = int(np.argmin(np.abs(THRESHOLDS - threshold)))
    if not np.isclose(THRESHOLDS[index], threshold, rtol=0, atol=1e-7):
        raise ValueError("Threshold must belong to the validation grid")
    groups = {}
    for kind in ("all_fake", "inj", "rem"):
        selected = [r for r in rows if not r["real"] and (kind == "all_fake" or r["kind"] == kind)]
        if not selected:
            continue
        counts = np.array([r["counts"] for r in selected])[:, :, index]
        dice, iou = metrics(counts.T)
        micro = metrics(counts.sum(axis=0))
        groups[kind] = dict(n=len(selected), macro_f1=float(dice.mean()), macro_iou=float(iou.mean()),
                            micro_f1=float(micro[0]), micro_iou=float(micro[1]))
    real = [r for r in rows if r["real"]]
    if real:
        fps = [r["counts"][1][index] for r in real]
        groups["real"] = dict(n=len(real), false_positive_voxels=sum(fps),
                              positive_voxel_fraction=sum(fps) / sum(r["voxels"] for r in real),
                              volumes_with_any_false_positive=sum(v > 0 for v in fps))
    volumes = []
    for r in rows:
        counts = np.asarray(r["counts"])[:, index]
        dice, iou = metrics(counts)
        volumes.append(dict(id=r["id"], kind=r["kind"], tp=int(counts[0]), fp=int(counts[1]),
                            fn=int(counts[2]), f1=None if r["real"] else float(dice),
                            iou=None if r["real"] else float(iou), preview=r["preview"],
                            preview_z=r["preview_z"]))
    return dict(threshold=float(THRESHOLDS[index]), groups=groups, volumes=volumes)


def export_prediction(model, entry, device, output, threshold, checkpoint_hash):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Prediction output must be empty: {output}")
    probabilities = np.lib.format.open_memmap(output / "probabilities.npy", mode="w+",
                                              dtype=np.float32, shape=tuple(entry["shape"]))
    masks = np.lib.format.open_memmap(output / "mask.npy", mode="w+", dtype=np.bool_,
                                     shape=tuple(entry["shape"]))
    for z, p in predict_slices(model, entry, device):
        probabilities[z] = p
        masks[z] = p >= threshold
    probabilities.flush()
    masks.flush()
    atomic_json(output / "prediction.json", dict(complete=True, shape=entry["shape"],
                spacing_zyx=entry["spacing"], bounds=entry["bounds"], threshold=threshold,
                checkpoint_sha256=checkpoint_hash, axes="z,y,x", scan=entry["scan"]))
