"""Fixed cohorts, strict annotation checks and volume-grouped segmentation crops."""
from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .dataset import M3DSynthRecord, assert_no_orig_id_leakage, cross_generator_records, label_dir, load_tiff_stack, read_records, scan_dir
from .volume_io import VolumeCache, align_mask_to_scan, count_contiguous_slices, load_label_mask, probe_volume_shape


def category(record):
    return "real" if record.is_real else record.ty


def unique_records(records, data_root, real_scan_root=None):
    unique = {}
    for record in records:
        unique.setdefault(str(scan_dir(data_root, record, real_scan_root).resolve()), record)
    return list(unique.values())


def stratified(records, counts, seed=21):
    rng = np.random.default_rng(seed)
    result = []
    for kind, count in counts.items():
        pool = sorted((r for r in records if category(r) == kind), key=lambda r: (r.mod, r.img_id))
        if len(pool) < count:
            raise ValueError(f"cohort needs {count} {kind} scans, only {len(pool)} available")
        result.extend(pool[int(i)] for i in rng.permutation(len(pool))[:count])
    return result


def make_manifest(config, data_root, real_scan_root=None):
    from .train import subset_records
    seed = int(config["training"].get("seed", 21))
    records = read_records(config.get("metadata_dir", "metadata/m3dsynth"))
    split = config["split"]
    groups = cross_generator_records(records, train_mods=split["train_mods"],
                                     valid_mods=split.get("valid_mods", ["pix2pix"]),
                                     test_mods=split.get("test_mods", ["cycle", "diffusion"]), include_real=True)
    train = unique_records(groups["train"], data_root, real_scan_root)
    common = unique_records(subset_records(groups["valid"], 64, seed), data_root, real_scan_root)
    cohorts = {
        "overfit": stratified(train, {"inj": 2, "rem": 2, "real": 2}, seed),
        "pilot_train": stratified(train, {"inj": 24, "rem": 24, "real": 16}, seed),
        "pilot_valid": stratified(common, {"inj": 2, "rem": 2, "real": 4}, seed),
        "full_train": train,
        "full_valid": stratified(common, {"inj": 4, "rem": 4, "real": 8}, seed),
        "common_valid": common,
    }
    assert_no_orig_id_leakage({"train": train, "valid": common, "test": groups["test"]})
    spacing = {}
    path = Path(config.get("metadata_dir", "metadata/m3dsynth")) / "LIDC.csv"
    if path.exists():
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                spacing[f'{row["orig_id"]}/{row["sdir_id"]}'] = [float(row[f"spacing_{a}"]) for a in "zyx"]
    result = {"format_version": 1, "seed": seed, "cohorts": {k: [asdict(r) for r in v] for k, v in cohorts.items()},
              "spacing": spacing}
    result["manifest_id"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def manifest_records(manifest, cohort):
    return [M3DSynthRecord(**row) for row in manifest["cohorts"][cohort]]


def validate_stack(dirname, *, check_headers=False):
    """Reject gaps instead of silently accepting the prefix before a gap."""
    dirname = Path(dirname)
    depth = count_contiguous_slices(dirname)
    files = [p for p in dirname.iterdir() if p.name.startswith("slide") and p.suffix == ".tiff"]
    if depth == 0 or len(files) != depth:
        raise ValueError(f"empty, gappy or unexpected TIFF slice sequence: {dirname}")
    shape = probe_volume_shape(dirname)
    if check_headers:
        from PIL import Image
        for path in files:
            with Image.open(path) as image:
                if image.size != (shape[2], shape[1]) or image.mode not in {"I;16", "I;16L", "I;16B"}:
                    raise ValueError(f"inconsistent scan slice shape or non-uint16 dtype: {path}")
    return shape


def checked_mask(data_root, record, shape):
    if record.is_real:
        return np.zeros(shape, dtype=bool)
    path = label_dir(data_root, record)
    validate_stack(path)
    mask = align_mask_to_scan(load_label_mask(path), shape, record.img_id, strict=True)
    if not mask.any():
        raise ValueError(f"empty manipulation annotation: {record.img_id}")
    return mask


def audit_records(records, data_root, real_scan_root=None):
    rows = []
    for record in records:
        shape = validate_stack(scan_dir(data_root, record, real_scan_root), check_headers=True)
        mask = checked_mask(data_root, record, shape)
        points = np.argwhere(mask)
        coord_in_scan = all(0 <= c < n for c, n in zip(record.coord, shape))
        row = {"img_id": record.img_id, "mod": record.mod, "shape": shape,
               "positive_voxels": int(mask.sum()), "metadata_coord_in_scan": coord_in_scan,
               "metadata_coord_in_mask": bool(mask[record.coord]) if coord_in_scan else False,
               "bbox_fill_fraction": None}
        if points.size:
            box_size = points.max(0) - points.min(0) + 1
            row["bbox_fill_fraction"] = float(len(points) / np.prod(box_size))
        rows.append(row)
        print(f'audit {record.mod}/{record.img_id}', flush=True)
    return rows


def load_bundle(data_root, record, cache_dir=None, preprocessing=None, real_scan_root=None):
    from .lung import lung_mask_from_ct
    from .segmentation import preprocess_scan
    dirname = scan_dir(data_root, record, real_scan_root)
    validate_stack(dirname)
    raw = load_tiff_stack(dirname)
    if raw.dtype != np.uint16:
        raise ValueError(f"expected uint16 CT: {dirname}")
    mask = checked_mask(data_root, record, raw.shape)
    lung = lung_mask_from_ct(raw)
    image, background = preprocess_scan(raw, dirname, cache_dir, preprocessing)
    return image, mask, lung, background


def crop_plan(mask, lung, patch_shape, rng, count=8):
    """Positive positions vary inside crops; negatives have exactly zero overlap."""
    patch = np.asarray(patch_shape)
    if count != 8 or np.any(np.asarray(mask.shape) < patch):
        raise ValueError("crop sampling requires eight crops and a padded volume")
    last = np.asarray(mask.shape) - patch
    positive = np.flatnonzero(mask)
    lungs = np.flatnonzero(lung)
    result = []
    n_positive = 4 if positive.size else 0
    for _ in range(n_positive):
        point = np.asarray(np.unravel_index(rng.choice(positive), mask.shape))
        start = np.clip(point - rng.integers(0, patch), 0, last)
        result.append(tuple(int(v) for v in start))
    n_lung = 2 if positive.size else 4
    for index in range(8 - n_positive):
        use_lung = index < n_lung and lungs.size > 0
        found = None
        for attempt in range(512):
            if use_lung and attempt < 256:
                point = np.asarray(np.unravel_index(rng.choice(lungs), mask.shape))
                start = np.clip(point - patch // 2, 0, last)
            else:
                start = rng.integers(0, last + 1)
            slc = tuple(slice(int(a), int(a + p)) for a, p in zip(start, patch))
            if not mask[slc].any():
                found = tuple(int(v) for v in start)
                break
        if found is None:
            # Deterministic exhaustive fallback, including edges, before failing.
            from .patches import PatchGrid
            for slc in PatchGrid(mask.shape, tuple(patch), tuple(patch)).iter_slices():
                if not mask[slc].any():
                    found = tuple(s.start for s in slc)
                    break
        if found is None:
            raise ValueError("scan has no clean crop of the configured size")
        result.append(found)
    return result


class SegmentationDataset:
    """One loader item is one volume's eight crops, split into GPU microbatches later."""
    def __init__(self, records, data_root, *, patch_shape=(64, 64, 64), cache_dir=None,
                 preprocessing=None, seed=21, augment=True, fixed=False, volume_cache_size=2, real_scan_root=None):
        self.records, self.data_root = list(records), Path(data_root)
        self.patch_shape, self.cache_dir = tuple(patch_shape), cache_dir
        self.preprocessing, self.seed = preprocessing, int(seed)
        self.real_scan_root = real_scan_root
        self.augment, self.fixed = augment, fixed
        self.cache = VolumeCache(volume_cache_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, key):
        import torch
        epoch, index = key if isinstance(key, tuple) else (0, key)
        rng = np.random.default_rng([self.seed, 0 if self.fixed else epoch, index])
        record = self.records[index]
        bundle = self.cache.get(str(scan_dir(self.data_root, record, self.real_scan_root).resolve()),
                                lambda: load_bundle(self.data_root, record, self.cache_dir, self.preprocessing, self.real_scan_root))
        image, mask, lung, background = bundle
        padding = [(0, max(0, p - s)) for p, s in zip(self.patch_shape, image.shape)]
        if any(b for _, b in padding):
            image = np.pad(image, padding, constant_values=background)
            mask = np.pad(mask, padding)
            lung = np.pad(lung, padding)
        coords = crop_plan(mask, lung, self.patch_shape, rng)
        images, targets = [], []
        for coord in coords:
            slc = tuple(slice(a, a + p) for a, p in zip(coord, self.patch_shape))
            crop, target = image[slc], mask[slc]
            if self.augment:
                for axis in (1, 2):
                    if rng.random() < 0.5:
                        crop, target = np.flip(crop, axis), np.flip(target, axis)
            images.append(np.array(crop, dtype=np.float32, copy=True))
            targets.append(np.array(target, dtype=np.float32, copy=True))
        return {"image": torch.from_numpy(np.stack(images)[:, None]),
                "mask": torch.from_numpy(np.stack(targets)[:, None]), "img_id": record.img_id}


class VolumeOrder:
    def __init__(self, length, seed=21):
        self.length, self.seed, self.epoch = length, seed, 0

    def __len__(self):
        return self.length

    def __iter__(self):
        for index in np.random.default_rng([self.seed, self.epoch]).permutation(self.length):
            yield (self.epoch, int(index))
