import csv
import json

import numpy as np
import pytest
from PIL import Image

from tesi_m3d.diagnose_localization import (
    alignment_audit, distribution_metrics, inspect_patches, overlap_audit,
    run_diagnostics, write_json, topk_mask_coverage,
)
from tesi_m3d.patches import PatchGrid


def test_overlap_threshold_effect_and_exact_distance():
    assert overlap_audit(np.array([0, 0.005, 0.01, 0.049, 0.05])) == [
        dict(positive_overlap_fraction=0.01, n_total=5, n_positive=3, n_negative=1, n_excluded=1),
        dict(positive_overlap_fraction=0.05, n_total=5, n_positive=1, n_negative=1, n_excluded=3),
    ]
    mask = np.zeros((6, 2, 2), dtype=bool)
    mask[0, 0, 0] = True
    grid = PatchGrid(mask.shape, (2, 2, 2), (2, 2, 2))
    rows = inspect_patches(mask, grid, np.array([0.1, 0.2, 0.9]), np.array([0.125, 0, 0]), 1)
    assert {row['patch_index'] for row in rows} == {0, 2}
    assert rows[0]['distance_to_mask_voxels'] == 0
    assert rows[1]['distance_to_mask_voxels'] == 4
    assert rows[1]['rank'] == 1
    assert rows[0]['selected_low_positive']
    assert topk_mask_coverage(mask, grid, np.array([0.1, 0.2, 0.9])) == {1: 0.0, 3: 1.0, 5: 1.0}


def test_alignment_exposes_truncated_positives_and_roi_exclusion():
    raw = np.zeros((4, 2, 2), dtype=bool)
    raw[0, 0, 0] = raw[3, 1, 1] = True
    mask = raw[:3]
    result = alignment_audit(mask.shape, raw, mask, np.zeros_like(mask), (3, 1, 1))
    assert result['discarded_positive_voxels'] == 1
    assert result['mask_outside_lung_voxels'] == 1
    assert result['z_adjustment'] == -1
    assert not result['metadata_coord_in_scan']
    assert result['mask_bbox_zyx'] == [[0, 0, 0], [0, 0, 0]]


def test_undefined_metrics_are_null_in_json(tmp_path):
    metrics = distribution_metrics([False, False], [0.1, 0.9])
    write_json(tmp_path / 'metrics.json', metrics)
    result = json.loads((tmp_path / 'metrics.json').read_text())
    assert result['ap'] is None and result['auc'] is None
    assert result['prevalence'] == 0
    with pytest.raises(ValueError):
        distribution_metrics([True], [0.1, 0.2])


def test_overlay_labels_patch_heatmap_metrics_and_undefined_values(tmp_path, monkeypatch):
    from PIL import ImageDraw
    from tesi_m3d.diagnose_localization import render_overlay
    from tesi_m3d.evaluation import voxel_auc_ap

    mask = np.zeros((2, 2, 4), dtype=bool)
    mask[0, 0, 0] = True
    heatmap = mask.astype(np.float32)
    grid = PatchGrid(mask.shape, (2, 2, 2), (2, 2, 2))
    patches = inspect_patches(mask, grid, np.array([0.8, 0.9]), np.array([0.125, 0]), 2)
    for patch, slc in zip(patches, grid.iter_slices()):
        patch['auc'], patch['ap'] = voxel_auc_ap(mask[slc], heatmap[slc])
    labels = []
    original = ImageDraw.ImageDraw.multiline_text

    def capture(draw, position, text, **kwargs):
        labels.append(text)
        return original(draw, position, text, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, 'multiline_text', capture)
    render_overlay(tmp_path / 'metrics.png', np.zeros(mask.shape), mask, mask,
                   heatmap, grid, patches, (0, 0, 0), 0.5)
    assert any('3D patch HM AUC=1.000 AP=1.000' in label for label in labels)
    assert any('3D patch HM AUC=n/a AP=n/a' in label for label in labels)
    assert (tmp_path / 'metrics.png').exists()


def _write_stack(directory, stack):
    directory.mkdir(parents=True)
    for z, plane in enumerate(stack):
        Image.fromarray(plane.astype(np.uint16)).save(directory / f'slide{z:04d}.tiff')


def test_diagnostics_same_predictions_across_checkpoints(tmp_path, monkeypatch):
    import torch
    import tesi_m3d.diagnose_localization as module
    from tesi_m3d.dataset import M3DSynthRecord

    torch.set_num_threads(1)
    records = [M3DSynthRecord('fake', 'add', 'pix2pix', 'a', 'a', 1, 1, 1, 'valid'),
               M3DSynthRecord('real', 'add', 'real', 'b', 'b', 1, 1, 1, 'valid')]
    records.append(records[-1])  # The training sampler groups duplicate physical scans.
    monkeypatch.setattr(module, '_load_validation_records', lambda *a, **kw: records)

    class MeanModel(torch.nn.Module):
        def forward(self, image):
            return 5 * image.mean((1, 2, 3, 4)) - 2

    monkeypatch.setattr(module, '_load_model', lambda *a: MeanModel())
    scan = np.arange(64, dtype=np.uint16).reshape(4, 4, 4) + 100
    mask = np.zeros((5, 4, 4), dtype=np.uint16)
    mask[:2, :2, :2] = 1
    _write_stack(tmp_path / 'pix2pix/scan/fake', scan)
    _write_stack(tmp_path / 'pix2pix/label/fake', mask)
    _write_stack(tmp_path / 'real/scan/real', scan)
    config = dict(patches=dict(patch_shape=[2, 2, 2], train_stride=[2, 2, 2],
                              inference_stride=[1, 1, 1], positive_overlap_fraction=0.05,
                              max_valid_patches_per_volume=4), training=dict(batch_size=2, seed=21))
    output = tmp_path / 'diagnosis'
    result = run_diagnostics(config, ['a/best.pt', 'b/best.pt'], tmp_path, output,
                             max_report_images=1, top_k=1, thresholds=[0.5])
    assert result[0]['sampled_ap'] == result[1]['sampled_ap']
    assert result[0]['dense_ap'] == result[1]['dense_ap']
    assert result[0]['dense_ap_rank'] == result[1]['dense_ap_rank'] == 1
    fixed = np.load(output / 'sampled_validation.npz')
    assert len(fixed['label']) == 8
    report_dir = output / '00_a_best'
    report = json.loads((report_dir / 'metrics.json').read_text())
    assert report['sampled']['n'] == 8
    assert len(report['volumes']) == 2
    assert report['dense_supervised']['n'] > 8
    assert report['volumes'][0]['discarded_positive_voxels'] == 0
    with (report_dir / 'local_heatmaps.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert {r['aggregation'] for r in rows} == {'average', 'gaussian'}
    with Image.open(report_dir / '0000_average_mask.png') as image:
        assert image.size == (1152, 1248)
    assert (report_dir / '0000_scores.npz').exists()
    assert json.loads((output / 'manifest.json').read_text())['split'] == 'valid'


def test_ablation_configs_change_one_factor_only():
    from tesi_m3d.train import load_yaml_config
    baseline = load_yaml_config('configs/train_pix2pix_balanced_patches.yaml')
    for filename, section, key, value in [
        ('train_pix2pix_overlap_1pct.yaml', 'patches', 'positive_overlap_fraction', 0.01),
        ('train_pix2pix_soft_targets.yaml', 'training', 'target', 'soft_score'),
    ]:
        candidate = load_yaml_config(f'configs/{filename}')
        assert candidate[section][key] == value
        candidate[section][key] = baseline[section][key]
        for name in ('experiment_name', 'output_dir'):
            assert candidate.pop(name) != baseline[name]
        assert candidate == {k: v for k, v in baseline.items() if k not in ('experiment_name', 'output_dir')}
