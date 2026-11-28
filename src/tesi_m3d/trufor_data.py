"""Slice IO and reproducible sampling for the Pix2Pix TruFor experiments."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from .dataset import read_records, cross_generator_records, scan_dir, label_dir


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, allow_nan=False)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def slice_paths(directory):
    directory = Path(directory)
    paths = sorted(directory.glob("slide*.tiff"))
    if not paths:
        raise FileNotFoundError(f"No TIFF slices in {directory}")
    expected = [f"slide{i:04d}.tiff" for i in range(len(paths))]
    if [p.name for p in paths] != expected:
        raise ValueError(f"Non-contiguous or invalid slice names: {directory}")
    return paths


def read_slice(directory, z):
    with Image.open(Path(directory) / f"slide{z:04d}.tiff") as image:
        return np.array(image)


def file_signature(paths):
    return [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns,
             p.stat().st_ctime_ns) for p in paths]


def histogram_percentiles(histogram):
    """Match numpy's linear percentile interpolation without loading a volume."""
    hist = histogram.copy()
    hist[0] = 0
    if not hist.sum():
        hist = histogram
    cumulative = hist.cumsum()
    n = int(cumulative[-1])
    result = []
    for q in (0.01, 0.99):
        rank = (n - 1) * q
        lo, hi = int(np.floor(rank)), int(np.ceil(rank))
        a, b = np.searchsorted(cumulative, [lo, hi], side="right")
        result.append(float(a + (b - a) * (rank - lo)))
    return result


def inspect_scan(directory, cache_dir, mask_dir=None):
    paths = slice_paths(directory)
    masks = slice_paths(mask_dir) if mask_dir else []
    signature = digest({"version": 1, "scan": file_signature(paths),
                        "mask": file_signature(masks), "percentiles": [1, 99]})
    cache = Path(cache_dir) / "percentiles" / f"{signature}.json"
    if cache.exists():
        try:
            result = json.loads(cache.read_text())
            bounds = np.asarray(result["bounds"], dtype=float)
            if (result["signature"] == signature and bounds.shape == (2,)
                    and np.isfinite(bounds).all() and 0 <= bounds[0] <= bounds[1] <= 65535
                    and len(result["shape"]) == 3 and result["shape"][0] == len(paths)
                    and all(0 <= z < len(paths) for z in result["positives"])
                    and "trimmed_empty_masks" in result):
                return result
        except (ValueError, KeyError, TypeError):
            pass
    hist = np.zeros(65536, dtype=np.int64)
    shape = None
    for path in paths:
        with Image.open(path) as image:
            array = np.array(image)
        if array.ndim != 2 or array.dtype.kind not in "uib" or array.min() < 0 or array.max() > 65535:
            raise ValueError(f"Expected unsigned 2D CT in [0, 65535]: {path}")
        if shape is not None and array.shape != shape:
            raise ValueError(f"Inconsistent CT slice dimensions: {path}")
        shape = array.shape
        hist += np.bincount(array.ravel().astype(np.int64), minlength=65536)
    positives = []
    if masks and len(masks) < len(paths):
        raise ValueError(f"Mask shorter than CT: {mask_dir}")
    for z, path in enumerate(masks):
        with Image.open(path) as image:
            mask = np.array(image)
        if mask.shape != shape:
            raise ValueError(f"CT/mask spatial mismatch: {path}")
        if mask.any():
            if z >= len(paths):
                raise ValueError(f"Alignment would discard positive mask: {path}")
            positives.append(z)
    if masks and not positives:
        raise ValueError(f"Manipulated CT has an empty mask: {mask_dir}")
    result = dict(signature=signature, shape=[len(paths), *shape],
                  bounds=histogram_percentiles(hist), positives=positives,
                  trimmed_empty_masks=max(0, len(masks) - len(paths)))
    atomic_json(cache, result)
    return result


def prepare(metadata_dir, data_root, real_scan_root, cache_dir, output):
    metadata_dir = Path(metadata_dir)
    records = read_records(metadata_dir)
    groups = cross_generator_records(records, ["pix2pix"], ["pix2pix"])
    with (metadata_dir / "LIDC.csv").open(newline="") as stream:
        spacing = {(r["orig_id"], r["sdir_id"]):
                   [float(r[f"spacing_{axis}"]) for axis in "zyx"]
                   for r in csv.DictReader(stream)}
    entries, seen = [], set()
    for split, rows in groups.items():
        for r in rows:
            identity = (r.orig_id, r.sdir_id)
            if r.is_real and identity in seen:
                continue
            if r.is_real:
                seen.add(identity)
            directory = scan_dir(data_root, r)
            if r.is_real and real_scan_root:
                directory = Path(real_scan_root) / r.img_id
                if not directory.is_dir():
                    directory = Path(real_scan_root) / f"{r.orig_id}__{r.sdir_id}"
            mask = None if r.is_real else label_dir(data_root, r)
            for source in (directory, mask):
                if source and any(Path(p).resolve().is_relative_to(source.resolve())
                                  for p in (cache_dir, output)):
                    raise ValueError(f"Output/cache cannot be inside input directory: {source}")
            physical = spacing[identity]
            if not all(np.isfinite(v) and v > 0 for v in physical):
                raise ValueError(f"Invalid spacing for {identity}")
            info = inspect_scan(directory, cache_dir, mask)
            entries.append(dict(id=f"{r.mod}/{r.img_id}", orig_id=r.orig_id,
                                sdir_id=r.sdir_id, split=split, real=r.is_real,
                                kind="real" if r.is_real else r.ty,
                                scan=str(directory.resolve()),
                                mask=str(mask.resolve()) if mask else None,
                                spacing=physical, **info))
            print(f"Prepared {len(entries)}: {split} {entries[-1]['id']}", flush=True)
    manifest = dict(version=1, entries=entries, fingerprint=digest(entries))
    atomic_json(output, manifest)
    print(f"Preparation complete: {len(entries)} volumes. Manifest: {Path(output).resolve()}", flush=True)
    return manifest


def load_manifest(path):
    path = Path(path)
    try:
        contents = path.read_text()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"TruFor manifest not found: {path.resolve()}\n"
            "Run python -m tesi_m3d.trufor prepare --data-root <dataset-root> "
            f'--real-scan-root <real-scan-root> --output "{path}" and wait for '
            "'Preparation complete' before profiling or training. "
            "If prepare saved the manifest elsewhere, pass --manifest <existing-file>. "
            "An interrupted preparation can be repeated: completed percentile caches are reused."
        ) from exc
    manifest = json.loads(contents)
    if manifest.get("version") != 1 or manifest.get("fingerprint") != digest(manifest["entries"]):
        raise ValueError("Invalid manifest; rerun prepare")
    patients = {}
    for e in manifest["entries"]:
        previous = patients.setdefault(e["orig_id"], e["split"])
        if previous != e["split"]:
            raise ValueError(f"Patient leakage: {e['orig_id']}")
    return manifest


def normalized_slice(entry, z):
    array = read_slice(entry["scan"], z).astype(np.float32)
    if tuple(array.shape) != tuple(entry["shape"][1:]):
        raise ValueError(f"CT changed since prepare: {entry['scan']}")
    lo, hi = entry["bounds"]
    return np.clip((array - lo) / max(hi - lo, 1e-6), 0, 1)


def slice_input(entry, z, context):
    indices = [z - 1, z, z + 1] if context else [z]
    images, valid, distances = [], [], []
    for i in indices:
        present = 0 <= i < entry["shape"][0]
        images.append(normalized_slice(entry, i if present else z))
        valid.append(present)
        distances.append((i - z) * entry["spacing"][0])
    return np.stack(images)[:, None], np.array(valid), np.array(distances, dtype=np.float32)


def epoch_samples(entries, epoch, seed):
    """Stateless sample/augmentation tokens survive worker prefetch and resume."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    real = [i for i, e in enumerate(entries) if e["real"]]
    tokens = []
    for i in rng.permutation(len(entries)):
        e = entries[i]
        positives = set(e["positives"])
        negatives = [z for z in range(e["shape"][0]) if z not in positives]
        for slot in range(4):
            target = int(i)
            if not e["real"] and slot < 2:
                z = int(rng.choice(sorted(positives)))
            elif negatives:
                z = int(rng.choice(negatives))
            else:
                if not real:
                    raise ValueError("No negative slices available")
                target = int(rng.choice(real))
                z = int(rng.integers(entries[target]["shape"][0]))
            tokens.append((target, z, int(rng.integers(4)), int(rng.integers(2)),
                           int(rng.integers(2))))
    return tokens


class SliceDataset:
    def __init__(self, entries, context):
        self.entries, self.context = entries, context

    def __len__(self):
        return len(self.entries) * 4

    def __getitem__(self, token):
        import torch
        index, z, rotation, flip_y, flip_x = token
        entry = self.entries[index]
        images, valid, distances = slice_input(entry, z, self.context)
        mask = (read_slice(entry["mask"], z) != 0 if entry["mask"] else
                np.zeros(entry["shape"][1:], dtype=bool))
        def transform(array):
            array = np.rot90(array, rotation, axes=(-2, -1))
            if flip_y:
                array = np.flip(array, -2)
            if flip_x:
                array = np.flip(array, -1)
            return torch.from_numpy(array.copy())
        return transform(images), transform(mask).long(), torch.from_numpy(valid), torch.from_numpy(distances)


def validation_panel(entries, seed=42):
    result = []
    for kind, count in (("inj", 20), ("rem", 20), ("real", 10)):
        candidates = [e for e in entries if e["split"] == "valid" and e["kind"] == kind]
        candidates.sort(key=lambda e: digest([seed, e["id"]]))
        result.extend(candidates[:count])
    if not any(not e["real"] for e in result):
        raise ValueError("Validation contains no manipulated volumes")
    return result
