import numpy as np
import pytest
from PIL import Image

from tesi_m3d.benchmark_loading import ProfiledDataset, batch_plan, measure_epochs
from tesi_m3d.dataset import M3DSynthPatchDataset, M3DSynthRecord, PatchExample
from tesi_m3d.sampling import SequentialVolumeBatchSampler


def _dataset(tmp_path):
    records = [M3DSynthRecord('sample', 'add', 'real', 'p', 's', 0, 0, 0)]
    directory = tmp_path / 'real/scan/sample'
    directory.mkdir(parents=True)
    for z in range(2):
        Image.fromarray(np.full((4, 4), 100 + z, dtype=np.uint16)).save(directory / f'slide{z:04d}.tiff')
    examples = [PatchExample(0, (0, y, x), 0, 0) for y in (0, 2) for x in (0, 2)]
    return M3DSynthPatchDataset(records, tmp_path, examples=examples, patch_shape=(2, 2, 2), volume_cache_size=1)


def test_shuffle_preserves_exact_samples_and_batch_sizes():
    sampler = SequentialVolumeBatchSampler(['a'] * 4 + ['b'] * 4, batch_size=2)
    grouped = batch_plan(sampler, 0, 'grouped', 21)
    shuffled = batch_plan(sampler, 0, 'shuffled', 21)
    assert sorted(i for b in grouped for i in b) == sorted(i for b in shuffled for i in b)
    assert list(map(len, grouped)) == list(map(len, shuffled))
    assert shuffled == batch_plan(sampler, 0, 'shuffled', 21)
    assert grouped != shuffled


def test_cache_counters_across_epochs(tmp_path):
    pytest.importorskip('torch')
    dataset = _dataset(tmp_path)
    sampler = SequentialVolumeBatchSampler(['sample'] * 4, batch_size=2)
    rows = measure_epochs(dataset, sampler, {'num_workers': 0}, epochs=2, order='grouped', seed=21)
    assert [r['volume_reads'] for r in rows] == [1, 0]
    assert [r['cache_hits'] for r in rows] == [3, 4]
    assert all(r['patches'] == 4 for r in rows)


def test_spawn_worker_counters_are_returned_to_parent(tmp_path):
    torch = pytest.importorskip('torch')
    dataset = _dataset(tmp_path)
    loader = torch.utils.data.DataLoader(ProfiledDataset(dataset), batch_size=2,
               num_workers=2, multiprocessing_context='spawn', persistent_workers=True, timeout=30)
    totals = []
    for epoch in range(2):
        batches = list(loader)
        totals.append(sum(int(batch['volume_reads'].sum()) for batch in batches))
        assert sum(int(batch['cache_hits'].sum()) for batch in batches) + totals[-1] == 4
    assert totals == [2, 0]
    assert dataset._volume_cache.misses == 0  # Counters are worker-local, not the parent's.


def test_benchmark_reuses_disk_index_and_never_rereads_masks_per_patch(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    import tesi_m3d.patch_index as module
    import tesi_m3d.train as train
    from tesi_m3d.benchmark_loading import run_benchmark
    from _tiff_fixtures import make_fake_corpus

    root, records = make_fake_corpus(tmp_path / 'data', n_records=1, scan_z=4, shape=(4, 4),
                                     cube_origin=(0, 0, 0), cube_size=(2, 2, 2))
    metadata = tmp_path / 'metadata'
    metadata.mkdir()
    monkeypatch.setattr(train, 'read_records', lambda *a: records)
    monkeypatch.setattr(train, 'build_patch3d_classifier', lambda *a, **kw: torch.nn.Linear(1, 1))
    loads = []
    original = module.load_label_mask

    def tracked(path):
        loads.append(path)
        return original(path)

    monkeypatch.setattr(module, 'load_label_mask', tracked)
    config = dict(metadata_dir=str(metadata), split=dict(train_mods=['pix2pix'], test_mods=['cycle']),
                  patches=dict(patch_shape=[2, 2, 2], train_stride=[2, 2, 2], max_patches_per_volume=4),
                  training=dict(batch_size=2, num_workers=0, seed=21))
    report = run_benchmark(config, root, tmp_path / 'benchmark', workers=[0], cache_sizes=[1], orders=['grouped'])
    assert len(loads) == 1
    assert report['n_index_patches'] == 8
    assert [r['volume_reads'] for r in report['epochs']] == [1, 0]
    assert (tmp_path / 'benchmark/epochs.csv').exists()
    assert (tmp_path / 'benchmark/benchmark.json').exists()
