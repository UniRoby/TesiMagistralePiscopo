"""Measure index startup and actual loader cache misses with 0/2 workers."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from time import perf_counter

import numpy as np

from .dataset import M3DSynthPatchDataset
from .diagnose_localization import write_csv, write_json
from .train import _build_dataloader, build_loaders, load_yaml_config


class ProfiledDataset:
    """Return worker-local cache deltas through each batch, including spawned workers."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        cache = self.dataset._volume_cache
        hits, misses = cache.hits, cache.misses
        started = perf_counter()
        item = self.dataset[index]
        item['io_seconds'] = perf_counter() - started
        item['volume_reads'] = cache.misses - misses
        item['cache_hits'] = cache.hits - hits
        return item


def batch_plan(sampler, epoch, order, seed):
    sampler.set_epoch(epoch)
    batches = list(sampler)
    if order == 'shuffled':
        sizes = [len(batch) for batch in batches]
        indices = np.asarray([i for batch in batches for i in batch])
        np.random.default_rng(seed + epoch).shuffle(indices)
        offsets = np.r_[0, np.cumsum(sizes)]
        batches = [indices[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]
    return batches


def measure_epochs(dataset, sampler, training, *, epochs, order, seed):
    """Drain complete epochs: prefetch cannot create uncounted reads after a cutoff."""
    rows = []
    loader = _build_dataloader(ProfiledDataset(dataset), batch_plan(sampler, 0, order, seed), training)
    for epoch in range(epochs):
        # Keep persistent workers and their caches alive between epoch iterations.
        loader.batch_sampler[:] = batch_plan(sampler, epoch, order, seed)
        started = perf_counter()
        patches = reads = hits = 0
        item_seconds = 0.0
        first_batch_seconds = None
        for batch in loader:
            if first_batch_seconds is None:
                first_batch_seconds = perf_counter() - started
            patches += len(batch['label'])
            reads += int(batch['volume_reads'].sum())
            hits += int(batch['cache_hits'].sum())
            item_seconds += float(batch['io_seconds'].sum())
        seconds = perf_counter() - started
        rows.append(dict(epoch=epoch + 1, workers=int(training.get('num_workers', 0)),
                         cache_size=dataset._volume_cache.maxsize, order=order, patches=patches,
                         seconds=seconds, first_batch_seconds=first_batch_seconds,
                         patches_per_second=patches / seconds, volume_reads=reads, cache_hits=hits,
                         worker_item_seconds=item_seconds))
    return rows


def run_benchmark(config, data_root, output, *, workers=(0, 2), cache_sizes=(1, 2),
                  orders=('grouped', 'shuffled'), epochs=2, rebuild_index=False):
    if epochs < 1 or any(w < 0 for w in workers) or any(c < 1 for c in cache_sizes):
        raise ValueError('epochs/cache sizes must be positive and workers non-negative')
    if any(o not in ('grouped', 'shuffled') for o in orders):
        raise ValueError('order must be grouped or shuffled')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    candidate = copy.deepcopy(config)
    candidate.setdefault('data', {})['max_valid_records'] = 0
    candidate.setdefault('training', {})['num_workers'] = 0
    cache_dir = output / 'index_cache'
    started = perf_counter()
    objects = build_loaders(candidate, data_root_override=str(data_root), cache_dir=cache_dir,
                            rebuild_index=rebuild_index)
    first_startup = perf_counter() - started
    started = perf_counter()
    warm = build_loaders(candidate, data_root_override=str(data_root), cache_dir=cache_dir)
    warm_startup = perf_counter() - started
    assert len(warm.train_dataset) == len(objects.train_dataset)
    del warm
    rows = []
    for worker_count in workers:
        for size in cache_sizes:
            for order in orders:
                source = objects.train_dataset
                dataset = M3DSynthPatchDataset(source.records, source.data_root, examples=source.examples,
                                               patch_shape=source.patch_shape, stride=source.stride,
                                               volume_cache_size=size)
                training = {**candidate['training'], 'num_workers': worker_count}
                print(f'workers={worker_count}, cache={size}, order={order}', flush=True)
                rows.extend(measure_epochs(dataset, objects.train_sampler, training, epochs=epochs,
                                           order=order, seed=int(training.get('seed', 21))))
                write_csv(output / 'epochs.csv', rows)
    report = dict(config=candidate, data_root=str(data_root), first_startup_seconds=first_startup,
                  first_startup_forced_rebuild=rebuild_index, warm_startup_seconds=warm_startup,
                  n_index_patches=len(objects.train_dataset), epochs=rows,
                  notes='Loader only: no GPU/forward/backward. Startup includes model/optimizer construction. '
                        'volume_reads counts normalized stack loader calls (cache misses), not OS physical disk reads. '
                        'worker_item_seconds sums concurrent item time; it is not wall time. '
                        'OS filesystem cache is not cleared. Every scenario starts with empty volume caches; '
                        'persistent workers follow the config. Index labels are reused, not loaded per patch.')
    write_json(output / 'benchmark.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-root')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--workers', type=int, nargs='+', default=[0, 2])
    parser.add_argument('--cache-sizes', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--orders', choices=['grouped', 'shuffled'], nargs='+', default=['grouped', 'shuffled'])
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--rebuild-index', action='store_true')
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    run_benchmark(config, args.data_root or config.get('data_root', 'data/M3Dsynth'), args.out_dir,
                  workers=args.workers, cache_sizes=args.cache_sizes, orders=args.orders,
                  epochs=args.epochs, rebuild_index=args.rebuild_index)


if __name__ == '__main__':
    main()
