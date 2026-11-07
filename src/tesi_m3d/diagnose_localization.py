"""Compare fixed validation sampling, patch ranking and reconstructed localization."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .dataset import label_dir, scan_dir
from .evaluate_patch_level import (
    _load_model, _load_validation_records, patch_center, patch_overlap_fractions,
    topk_hits, highest_coverage_rank, score_overlap_spearman,
)
from .evaluation import binary_localization_metrics, voxel_auc_ap, voxel_auc_ap_in_roi
from .inference import predict_patch_scores
from .patch_index import load_or_build_patch_index
from .patches import PatchGrid, reconstruct_heatmap
from .sampling import SequentialVolumeBatchSampler
from .train import load_yaml_config
from .volume_io import align_mask_to_scan, load_label_mask, load_normalized_scan_and_lung_mask


def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def write_json(path: Path, value) -> None:
    """Write valid JSON; undefined metrics are null, never NaN."""
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(val) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [clean(val) for val in item]
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n', encoding='utf-8')


def distribution_metrics(truth, scores) -> dict:
    truth, scores = np.asarray(truth, dtype=bool), np.asarray(scores)
    if truth.shape != scores.shape:
        raise ValueError('truth and scores must have identical shapes')
    auc, ap = voxel_auc_ap(truth, scores) if truth.size else (float('nan'), float('nan'))
    return dict(n=int(truth.size), n_positive=int(truth.sum()),
                prevalence=float(truth.mean()) if truth.size else None, auc=auc, ap=ap)


def overlap_audit(overlaps, fractions=(0.01, 0.05)) -> list[dict]:
    overlaps = np.asarray(overlaps)
    return [dict(positive_overlap_fraction=float(fraction), n_total=int(overlaps.size),
                 n_positive=int(np.count_nonzero(overlaps >= fraction)),
                 n_negative=int(np.count_nonzero(overlaps == 0)),
                 n_excluded=int(np.count_nonzero((overlaps > 0) & (overlaps < fraction))))
            for fraction in fractions]


def alignment_audit(scan_shape, raw_mask, mask, lung, coord) -> dict:
    points = np.argwhere(mask)
    inside = all(0 <= c < s for c, s in zip(coord, scan_shape))
    return dict(scan_shape=list(scan_shape), raw_mask_shape=list(raw_mask.shape),
                z_adjustment=int(scan_shape[0] - raw_mask.shape[0]),
                raw_positive_voxels=int(raw_mask.sum()), positive_voxels=int(mask.sum()),
                discarded_positive_voxels=int(raw_mask.sum() - mask.sum()),
                mask_bbox_zyx=[points.min(0).tolist(), points.max(0).tolist()] if points.size else None,
                metadata_coord_zyx=list(coord), metadata_coord_in_scan=inside,
                metadata_coord_in_mask=bool(mask[tuple(coord)]) if inside else False,
                lung_voxels=int(lung.sum()), mask_outside_lung_voxels=int((mask & ~lung).sum()))


def inspect_patches(mask, grid, scores, overlaps, top_k) -> list[dict]:
    """Keep highest scores, best coverage and lowest-scored positives (FN candidates)."""
    order = np.argsort(-scores, kind='stable')
    rank = np.empty(len(scores), dtype=int)
    rank[order] = np.arange(1, len(scores) + 1)
    positive = np.flatnonzero(overlaps > 0)
    worst = positive[np.argsort(scores[positive], kind='stable')[:top_k]]
    coverage = np.argsort(-overlaps, kind='stable')[:top_k] if positive.size else []
    selected = sorted(set(order[:top_k]) | set(worst) | set(coverage))
    points = np.argwhere(mask)
    center = points.mean(0) if points.size else None
    rows = []
    slices = list(grid.iter_slices())
    for index in selected:
        slc = slices[index]
        start, stop = np.array([s.start for s in slc]), np.array([s.stop for s in slc])
        # Exact voxel-center distance to the box, including multiple disconnected lesions.
        distance = np.linalg.norm(np.maximum(np.maximum(start - points, points - (stop - 1)), 0), axis=1).min() if points.size else None
        rows.append(dict(patch_index=int(index), rank=int(rank[index]), score=float(scores[index]),
                         soft_score=float(overlaps[index]), start_zyx=start.tolist(), stop_zyx=stop.tolist(),
                         center_zyx=patch_center(slc).tolist(), distance_to_mask_voxels=float(distance) if distance is not None else None,
                         center_distance_voxels=float(np.linalg.norm(patch_center(slc) - center)) if center is not None else None,
                         selected_top_score=bool(index in order[:top_k]), selected_low_positive=bool(index in worst)))
    return rows


def topk_mask_coverage(mask, grid, scores, ks=(1, 3, 5)) -> dict:
    points = np.argwhere(mask)
    if not points.size:
        return {k: None for k in ks}
    slices = list(grid.iter_slices())
    order = np.argsort(-scores, kind='stable')
    result = {}
    for k in ks:
        covered = np.zeros(len(points), dtype=bool)
        for index in order[:k]:
            slc = slices[index]
            covered |= np.all((points >= [s.start for s in slc]) & (points < [s.stop for s in slc]), axis=1)
        result[k] = float(covered.mean())
    return result


def render_overlay(path, scan, mask, lung, heatmap, grid, patches, center, threshold):
    """Same voxel coordinates and fixed [0,1] color scale for every panel."""
    from PIL import Image, ImageDraw
    canvas = Image.new('RGB', (3 * 384, 3 * 416), 'black')
    for axis, name in enumerate(('axial z', 'coronal y', 'sagittal x')):
        section = int(np.clip(round(center[axis]), 0, scan.shape[axis] - 1))
        ct, truth, roi, hm = [np.take(a, section, axis=axis) for a in (scan, mask, lung, heatmap)]
        base = np.repeat((np.clip(ct, 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)
        other = [i for i in range(3) if i != axis]
        for col, title in enumerate(('CT / mask green / lung cyan', 'heatmap fixed 0..1 / patch grid', f'FP red / FN blue / TP green t={threshold:g}')):
            rgb = base.copy()
            if col == 0:
                from scipy.ndimage import binary_erosion
                rgb[roi & ~binary_erosion(roi)] = (0, 220, 255)
                rgb[truth] = (0, 255, 0)
            elif col == 1:
                color = np.zeros_like(rgb)
                color[..., 0] = np.uint8(np.clip(hm, 0, 1) * 255)
                rgb = np.uint8(0.5 * rgb + 0.5 * color)
            else:
                pred = hm >= threshold
                rgb[pred & ~truth] = (255, 0, 0)
                rgb[~pred & truth] = (0, 80, 255)
                rgb[pred & truth] = (0, 255, 0)
            scale = 384 / max(ct.shape)
            height, width = (max(1, round(size * scale)) for size in ct.shape)
            offset_x, offset_y = (384 - width) // 2, (384 - height) // 2
            image = Image.new('RGB', (384, 384), 'black')
            image.paste(Image.fromarray(rgb).resize((width, height), Image.Resampling.NEAREST), (offset_x, offset_y))
            draw = ImageDraw.Draw(image)
            if col == 1:
                for dim, length in enumerate(ct.shape):
                    for start in grid.starts[other[dim]]:
                        p = start * scale
                        draw.line((offset_x, offset_y + p, offset_x + width, offset_y + p) if dim == 0
                                  else (offset_x + p, offset_y, offset_x + p, offset_y + height), fill=(80, 80, 80))
                for patch in patches:
                    start, stop = patch['start_zyx'], patch['stop_zyx']
                    if start[axis] <= section < stop[axis]:
                        a, b = other
                        box = (offset_x + start[b] * scale, offset_y + start[a] * scale,
                               offset_x + (stop[b] - 1) * scale, offset_y + (stop[a] - 1) * scale)
                        draw.rectangle(box, outline=(255, 255, 0), width=2)
                        distance = patch['distance_to_mask_voxels']
                        label = f'{distance:.1f}' if distance is not None else 'n/a'
                        auc = f"{patch['auc']:.3f}" if np.isfinite(patch['auc']) else 'n/a'
                        ap = f"{patch['ap']:.3f}" if np.isfinite(patch['ap']) else 'n/a'
                        text = (f"#{patch['rank']} s={patch['score']:.2f} o={patch['soft_score']:.1%} d={label}\n"
                                f"3D patch HM AUC={auc} AP={ap}")
                        bounds = draw.multiline_textbbox((0, 0), text)
                        position = (min(box[0], 384 - bounds[2] - 2), min(box[1], 384 - bounds[3] - 2))
                        draw.rectangle(draw.multiline_textbbox(position, text), fill='black')
                        draw.multiline_text(position, text, fill='white')
            canvas.paste(image, (col * 384, axis * 416 + 32))
            ImageDraw.Draw(canvas).text((col * 384 + 4, axis * 416 + 4), f'{name}={section} | {title}', fill='white')
    canvas.save(path)


def run_diagnostics(config, checkpoints, data_root, output, *, device='cpu', img_ids=None,
                    max_report_images=4, top_k=5, thresholds=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)):
    if max_report_images < 0 or top_k < 1 or not thresholds or any(not 0 <= t <= 1 for t in thresholds):
        raise ValueError('invalid image count, top-k or probability thresholds')
    output, data_root = Path(output), Path(data_root)
    output.mkdir(parents=True, exist_ok=True)
    records = _load_validation_records(config, data_root, deduplicate=False)
    if img_ids:
        wanted = set(img_ids)
        missing = wanted - {r.img_id for r in records}
        if missing:
            raise ValueError(f'IDs outside the configured validation: {sorted(missing)}')
        records = [r for r in records if r.img_id in wanted]
    if not records:
        raise ValueError('empty validation selection')
    pc, tc = config.get('patches', {}), config.get('training', {})
    shape = tuple(pc.get('patch_shape', (32, 32, 32)))
    stride = tuple(pc.get('train_stride', shape))
    dense_stride = tuple(pc.get('inference_stride', (16, 16, 16)))
    fraction = float(pc.get('positive_overlap_fraction', 0.05))
    if not 0 < fraction <= 1:
        raise ValueError('positive_overlap_fraction must be in (0,1]')
    batch_size = int(tc.get('batch_size', 32))
    index = load_or_build_patch_index(records, data_root, output / 'cache', patch_shape=shape,
                                    stride=stride, positive_overlap_fraction=fraction)
    keys = [str(scan_dir(data_root, records[int(i)]).resolve()) for i in index.record_index]
    sampler = SequentialVolumeBatchSampler(keys, batch_size, labels=index.label,
                max_patches_per_volume=pc.get('max_valid_patches_per_volume', 256),
                max_volumes=tc.get('max_valid_volumes'), seed=int(tc.get('seed', 21)))
    selected = np.asarray([i for batch in sampler for i in batch], dtype=int)
    if not selected.size:
        raise ValueError('configured validation sampler yielded no patches')
    fixed = index.subset(selected)
    selected_keys = {keys[i] for i in selected}
    fixed.save(output / 'sampled_validation.npz')
    manifest = dict(split='valid', config=config, checkpoints=[str(p) for p in checkpoints],
                    data_root=str(data_root.resolve()), device=device, top_k=top_k,
                    records=[vars(r) for r in records],
                    sampled_indices=selected.tolist(), thresholds=list(thresholds),
                    distance_units='voxel centers in z,y,x; no physical spacing inferred',
                    notes='Fixed thresholds, no model selection or calibration on test. Dense volumes deduplicated by scan path.')
    write_json(output / 'manifest.json', manifest)
    comparison = []
    for run_number, checkpoint in enumerate(checkpoints):
        checkpoint = Path(checkpoint)
        destination = output / f'{run_number:02d}_{checkpoint.parent.name}_{checkpoint.stem}'
        destination.mkdir(exist_ok=True)
        model = _load_model(checkpoint, device)
        sampled_scores = np.empty(len(fixed), dtype=np.float32)
        dense_truth, dense_scores, all_truth, all_scores = [], [], [], []
        volume_rows, patch_rows, heatmap_rows, local_rows = [], [], [], []
        seen, images = set(), 0
        for rec_idx, record in enumerate(records):
            key = str(scan_dir(data_root, record).resolve())
            if key in seen or key not in selected_keys:
                continue
            seen.add(key)
            print(f'{checkpoint.name}: {record.mod}/{record.img_id}', flush=True)
            started = perf_counter()
            scan, lung = load_normalized_scan_and_lung_mask(data_root, record)
            raw_mask = np.zeros(scan.shape, dtype=bool) if record.is_real else load_label_mask(label_dir(data_root, record))
            mask = align_mask_to_scan(raw_mask, scan.shape, record.img_id)
            grid = PatchGrid(scan.shape, shape, dense_stride)
            inference_start = perf_counter()
            scores = predict_patch_scores(model, scan, grid, batch_size=batch_size, device=device)
            dense_inference_seconds = perf_counter() - inference_start
            overlaps = patch_overlap_fractions(mask, grid)
            coords = np.asarray(list(grid.iter_coords()), dtype=np.int32)
            np.savez_compressed(destination / f'{rec_idx:04d}_scores.npz', scores=scores, overlaps=overlaps, coords=coords,
                                volume_shape=scan.shape, patch_shape=shape, stride=dense_stride)
            matching_records = [i for i, r in enumerate(records) if str(scan_dir(data_root, r).resolve()) == key]
            positions = np.flatnonzero(np.isin(fixed.record_index, matching_records))
            if positions.size:
                train_grid = grid if stride == dense_stride else PatchGrid(scan.shape, shape, stride)
                train_scores = scores if train_grid is grid else predict_patch_scores(model, scan, train_grid, batch_size=batch_size, device=device)
                lookup = {c: i for i, c in enumerate(train_grid.iter_coords())}
                sampled_scores[positions] = [train_scores[lookup[tuple(c)]] for c in fixed.coord[positions]]
            supervised = (overlaps == 0) | (overlaps >= fraction)
            dense_truth.append(overlaps[supervised] >= fraction)
            dense_scores.append(scores[supervised])
            all_truth.append(overlaps >= fraction)
            all_scores.append(scores)
            inspected = inspect_patches(mask, grid, scores, overlaps, top_k)
            for patch in inspected:
                patch_rows.append(dict(record_index=rec_idx, mod=record.mod, img_id=record.img_id, **patch))
            alignment = alignment_audit(scan.shape, raw_mask, mask, lung, record.coord)
            write_json(destination / f'{rec_idx:04d}_alignment.json', alignment)
            audits = overlap_audit(overlaps, tuple(sorted(set((0.01, 0.05, fraction)))))
            train_overlaps = overlaps if stride == dense_stride else patch_overlap_fractions(mask, PatchGrid(scan.shape, shape, stride))
            train_audits = overlap_audit(train_overlaps, tuple(sorted(set((0.01, 0.05, fraction)))))
            volume_rows.append(dict(record_index=rec_idx, mod=record.mod, img_id=record.img_id,
                **distribution_metrics(overlaps[supervised] >= fraction, scores[supervised]),
                score_overlap_spearman=score_overlap_spearman(scores, overlaps),
                best_coverage_rank=highest_coverage_rank(overlaps, scores)[0] if mask.any() else None,
                topk_hits=topk_hits(overlaps, scores) if mask.any() else None,
                topk_mask_coverage=topk_mask_coverage(mask, grid, scores),
                dense_inference_seconds=dense_inference_seconds,
                dense_overlap_audit=audits, train_overlap_audit=train_audits,
                discarded_positive_voxels=alignment['discarded_positive_voxels']))
            roi = lung | mask
            for aggregation in ('average', 'gaussian'):
                heatmap = reconstruct_heatmap(scores, grid, mode=aggregation)
                for scope, region in (('global', np.ones(mask.shape, dtype=bool)), ('lung_union_mask', roi)):
                    auc, ap = voxel_auc_ap_in_roi(mask, heatmap, region)
                    for threshold in thresholds:
                        metrics = binary_localization_metrics(mask[region], heatmap[region] >= threshold).to_dict()
                        heatmap_rows.append(dict(record_index=rec_idx, img_id=record.img_id, aggregation=aggregation,
                            scope=scope, threshold=threshold, auc=auc, ap=ap, n_voxels=int(region.sum()),
                            prevalence=float(mask[region].mean()) if region.any() else None, **metrics))
                annotated_patches = []
                for patch in inspected:
                    slc = tuple(slice(a, b) for a, b in zip(patch['start_zyx'], patch['stop_zyx']))
                    auc, ap = voxel_auc_ap(mask[slc], heatmap[slc])
                    annotated_patches.append(dict(patch, auc=auc, ap=ap))
                    local_rows.append(dict(record_index=rec_idx, img_id=record.img_id, aggregation=aggregation,
                        patch_index=patch['patch_index'], score=patch['score'], soft_score=patch['soft_score'],
                        heatmap_min=float(heatmap[slc].min()), heatmap_max=float(heatmap[slc].max()), auc=auc, ap=ap))
                if images < max_report_images:
                    centers = [('metadata', record.coord)]
                    if mask.any():
                        centers.append(('mask', np.argwhere(mask).mean(0)))
                    centers.append(('top_score', next(p['center_zyx'] for p in inspected if p['rank'] == 1)))
                    low = [p for p in inspected if p['selected_low_positive']]
                    if low:
                        centers.append(('low_positive', min(low, key=lambda p: p['score'])['center_zyx']))
                    for title, center in centers:
                        render_overlay(destination / f'{rec_idx:04d}_{aggregation}_{title}.png', scan, mask, lung,
                                       heatmap, grid, annotated_patches, center, thresholds[0])
            images += 1
            volume_rows[-1]['seconds'] = perf_counter() - started
        sampled = distribution_metrics(fixed.label, sampled_scores)
        dense = distribution_metrics(np.concatenate(dense_truth), np.concatenate(dense_scores))
        all_dense = distribution_metrics(np.concatenate(all_truth), np.concatenate(all_scores))
        np.savez_compressed(destination / 'sampled_scores.npz', scores=sampled_scores, labels=fixed.label)
        report = dict(checkpoint=str(checkpoint), split='valid', sampled=sampled, dense_supervised=dense,
                      dense_all_boundary_as_negative=all_dense, volumes=volume_rows,
                      coverage='Reconstruction raises if any voxel is uncovered; no fabricated zero scores.')
        write_json(destination / 'metrics.json', report)
        write_csv(destination / 'patches.csv', patch_rows)
        write_csv(destination / 'heatmaps.csv', heatmap_rows)
        write_csv(destination / 'local_heatmaps.csv', local_rows)
        comparison.append(dict(checkpoint=str(checkpoint), sampled_ap=sampled['ap'], dense_ap=dense['ap'],
                               sampled_prevalence=sampled['prevalence'], dense_prevalence=dense['prevalence']))
    for metric in ('sampled_ap', 'dense_ap'):
        values = sorted({row[metric] for row in comparison if np.isfinite(row[metric])}, reverse=True)
        for row in comparison:
            row[f'{metric}_rank'] = values.index(row[metric]) + 1 if np.isfinite(row[metric]) else None
    write_json(output / 'comparison.json', comparison)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Common evaluation definition for every checkpoint.')
    parser.add_argument('--checkpoint', required=True, nargs='+')
    parser.add_argument('--data-root')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--img-ids', nargs='+', help='Restrict to explicit IDs within configured validation.')
    parser.add_argument('--max-report-images', type=int, default=4, help='Volumes with overlays, including real CTs.')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    run_diagnostics(config, args.checkpoint, args.data_root or config.get('data_root', 'data/M3Dsynth'), args.out_dir,
                    device=args.device, img_ids=args.img_ids, max_report_images=args.max_report_images,
                    top_k=args.top_k, thresholds=args.thresholds)


if __name__ == '__main__':
    main()
