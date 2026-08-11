"""Volume-grouped batch sampling for patch training.

The obvious sampler for an imbalanced patch dataset is a ``WeightedRandomSampler``
over all patches, and that is what this project used. It is catastrophic here:
patches are drawn in fully random order across ~2000 volumes, so two consecutive
items essentially never share a volume, and every item pays a full TIFF stack
decode. Adding a cache does not help, because the hit rate is ~0 by construction.

:class:`VolumeGroupedBatchSampler` fixes the *order* instead. Every batch draws
from exactly one volume, so one decode serves a whole batch.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np


class VolumeGroupedBatchSampler:
    """Yield batches of dataset indices that each come from a single volume.

    Class balance is handled per volume rather than globally. A pix2pix volume
    contains only 4-8 positive patches out of 1000-2800, so drawing a 50/50 batch
    would need each positive 3-4 times in the same batch, which corrupts
    BatchNorm statistics and invites memorization. Instead every selected volume
    contributes *all* of its positives once, plus ``neg_per_pos`` times as many
    distinct negatives. That yields ~16% positives per batch against a natural
    rate of ~0.25%, with no duplication.
    """

    def __init__(
        self,
        volume_keys: Sequence[str],
        labels: np.ndarray,
        batch_size: int,
        patches_per_volume: int | None = None,
        neg_per_pos: float = 5.0,
        max_positives_per_volume: int | None = None,
        max_patches_per_epoch: int | None = None,
        max_volumes_per_epoch: int | None = None,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        """Group example indices by volume and precompute the per-epoch budget."""

        labels = np.asarray(labels)
        if len(volume_keys) != len(labels):
            raise ValueError("volume_keys and labels must have the same length")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if len(labels) == 0:
            raise ValueError("cannot sample from an empty dataset")

        self.batch_size = int(batch_size)
        self.patches_per_volume = int(patches_per_volume or batch_size)
        self.neg_per_pos = float(neg_per_pos)
        self.max_positives_per_volume = max_positives_per_volume
        self.max_patches_per_epoch = max_patches_per_epoch
        self.max_volumes_per_epoch = max_volumes_per_epoch
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0

        if self.patches_per_volume % self.batch_size != 0:
            raise ValueError(
                f"patches_per_volume ({self.patches_per_volume}) must be a multiple of "
                f"batch_size ({self.batch_size}) so no batch spans two volumes"
            )

        # Group once. Storing plain ints and numpy arrays keeps the sampler
        # trivially picklable for spawned DataLoader workers on Windows.
        order = np.argsort(np.asarray(volume_keys, dtype=object), kind="stable")
        keys_sorted = [volume_keys[i] for i in order]
        boundaries = [0]
        for position in range(1, len(keys_sorted)):
            if keys_sorted[position] != keys_sorted[position - 1]:
                boundaries.append(position)
        boundaries.append(len(keys_sorted))

        self._positives: list[np.ndarray] = []
        self._negatives: list[np.ndarray] = []
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            member_indices = order[start:stop]
            member_labels = labels[member_indices]
            self._positives.append(member_indices[member_labels > 0].astype(np.int64))
            self._negatives.append(member_indices[member_labels == 0].astype(np.int64))

        self._n_volumes = len(self._positives)
        self._batches_per_epoch = self._compute_batches_per_epoch()

    @property
    def n_volumes(self) -> int:
        """Return the number of distinct volumes available."""

        return self._n_volumes

    def _volumes_this_epoch(self) -> int:
        """Return how many volumes one epoch visits."""

        if self.max_volumes_per_epoch is None:
            return self._n_volumes
        return min(self._n_volumes, int(self.max_volumes_per_epoch))

    def _compute_batches_per_epoch(self) -> int:
        """Return the exact number of batches ``__iter__`` will yield."""

        batches_per_volume = self.patches_per_volume // self.batch_size
        total = self._volumes_this_epoch() * batches_per_volume
        if self.max_patches_per_epoch is not None:
            total = min(total, int(self.max_patches_per_epoch) // self.batch_size)
        return max(total, 0)

    def set_epoch(self, epoch: int) -> None:
        """Reseed for ``epoch`` so runs are reproducible but epochs differ.

        The RNG is derived from ``[seed, epoch]`` rather than global state, so
        sampling is unaffected by whatever else consumes randomness.
        """

        self._epoch = int(epoch)

    def _sample_volume(self, volume: int, rng: np.random.Generator) -> np.ndarray:
        """Return the indices this volume contributes to one epoch."""

        positives = self._positives[volume]
        negatives = self._negatives[volume]

        n_pos = len(positives)
        if self.max_positives_per_volume is not None:
            n_pos = min(n_pos, int(self.max_positives_per_volume))
        # Cap positives so negatives never get squeezed out entirely.
        max_pos_by_ratio = int(self.patches_per_volume / (1.0 + self.neg_per_pos))
        n_pos = min(n_pos, max(max_pos_by_ratio, 0), self.patches_per_volume)

        chosen_pos = (
            rng.choice(positives, size=n_pos, replace=False)
            if n_pos > 0
            else np.empty(0, dtype=np.int64)
        )

        n_neg = self.patches_per_volume - n_pos
        if n_neg > 0 and len(negatives) == 0:
            # A volume that is entirely positive cannot contribute negatives;
            # top up with positives rather than emitting a short batch.
            chosen_neg = rng.choice(positives, size=n_neg, replace=n_neg > len(positives))
        elif n_neg > 0:
            chosen_neg = rng.choice(negatives, size=n_neg, replace=n_neg > len(negatives))
        else:
            chosen_neg = np.empty(0, dtype=np.int64)

        selected = np.concatenate([chosen_pos, chosen_neg]).astype(np.int64)
        rng.shuffle(selected)
        return selected

    def __iter__(self) -> Iterator[list[int]]:
        """Yield one epoch of single-volume batches."""

        rng = np.random.default_rng([self.seed, self._epoch])
        volume_order = rng.permutation(self._n_volumes)[: self._volumes_this_epoch()]

        emitted = 0
        for volume in volume_order:
            if emitted >= self._batches_per_epoch:
                return
            selected = self._sample_volume(int(volume), rng)
            for start in range(0, len(selected), self.batch_size):
                batch = selected[start : start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                if emitted >= self._batches_per_epoch:
                    return
                emitted += 1
                yield [int(i) for i in batch]

    def __len__(self) -> int:
        """Return the exact number of batches per epoch."""

        return self._batches_per_epoch

    def describe(self) -> dict[str, float]:
        """Return sampler statistics for logging and ``--dry-run``."""

        positives_per_volume = np.asarray([len(p) for p in self._positives], dtype=np.float64)
        max_pos_by_ratio = int(self.patches_per_volume / (1.0 + self.neg_per_pos))
        effective = np.minimum(positives_per_volume, max_pos_by_ratio)
        return {
            "n_volumes": float(self._n_volumes),
            "n_volumes_per_epoch": float(self._volumes_this_epoch()),
            "batches_per_epoch": float(self._batches_per_epoch),
            "patches_per_epoch": float(self._batches_per_epoch * self.batch_size),
            "volumes_with_positives": float(np.count_nonzero(positives_per_volume)),
            "mean_positives_per_volume": float(positives_per_volume.mean()),
            "mean_positives_per_batch": float(
                effective.mean() * self.batch_size / self.patches_per_volume
            ),
        }


class SequentialVolumeBatchSampler:
    """Yield every patch exactly once, grouped by volume, for evaluation.

    Validation must see the natural class distribution, otherwise reported AP is
    measured against a distribution the model will never face. This sampler only
    reorders: it does not resample or rebalance.
    """

    def __init__(
        self,
        volume_keys: Sequence[str],
        batch_size: int,
        max_patches_per_volume: int | None = None,
        max_volumes: int | None = None,
        seed: int = 0,
    ) -> None:
        """Group example indices by volume, preserving every patch."""

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.batch_size = int(batch_size)
        self.seed = int(seed)

        order = np.argsort(np.asarray(volume_keys, dtype=object), kind="stable")
        keys_sorted = [volume_keys[i] for i in order]
        boundaries = [0]
        for position in range(1, len(keys_sorted)):
            if keys_sorted[position] != keys_sorted[position - 1]:
                boundaries.append(position)
        boundaries.append(len(keys_sorted))

        rng = np.random.default_rng(self.seed)
        groups = [order[start:stop].astype(np.int64) for start, stop in zip(boundaries[:-1], boundaries[1:])]
        if max_volumes is not None and max_volumes < len(groups):
            groups = [groups[i] for i in sorted(rng.permutation(len(groups))[:max_volumes])]
        if max_patches_per_volume is not None:
            groups = [
                group if len(group) <= max_patches_per_volume
                else rng.choice(group, size=max_patches_per_volume, replace=False)
                for group in groups
            ]

        self._batches: list[list[int]] = []
        for group in groups:
            for start in range(0, len(group), self.batch_size):
                self._batches.append([int(i) for i in group[start : start + self.batch_size]])

    def set_epoch(self, epoch: int) -> None:
        """Accept the epoch hook; evaluation order is intentionally fixed."""

    def __iter__(self) -> Iterator[list[int]]:
        """Yield each batch once, in volume order."""

        return iter(self._batches)

    def __len__(self) -> int:
        """Return the number of evaluation batches."""

        return len(self._batches)
