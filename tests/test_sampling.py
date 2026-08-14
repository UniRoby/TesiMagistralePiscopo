"""Tests for volume-grouped batch sampling."""

from __future__ import annotations

import pickle
import unittest

import numpy as np

from tesi_m3d.sampling import SequentialVolumeBatchSampler, VolumeGroupedBatchSampler


def make_keys_and_labels(n_volumes: int = 6, per_volume: int = 200, n_pos: int = 5):
    """Return volume keys and labels mimicking the real ~0.25% positive rate."""

    keys: list[str] = []
    labels: list[int] = []
    for volume in range(n_volumes):
        keys.extend([f"vol{volume:03d}"] * per_volume)
        labels.extend([1] * n_pos + [0] * (per_volume - n_pos))
    return keys, np.asarray(labels, dtype=np.uint8)


class TestVolumeGrouping(unittest.TestCase):
    def test_every_batch_comes_from_one_volume(self) -> None:
        keys, labels = make_keys_and_labels()
        sampler = VolumeGroupedBatchSampler(keys, labels, batch_size=32, seed=7)
        for batch in sampler:
            self.assertEqual(len({keys[i] for i in batch}, ), 1)

    def test_len_matches_yielded_batches(self) -> None:
        keys, labels = make_keys_and_labels()
        sampler = VolumeGroupedBatchSampler(keys, labels, batch_size=32, seed=7)
        self.assertEqual(len(sampler), len(list(sampler)))

    def test_all_batches_are_full(self) -> None:
        keys, labels = make_keys_and_labels()
        sampler = VolumeGroupedBatchSampler(keys, labels, batch_size=16, seed=7)
        self.assertTrue(all(len(batch) == 16 for batch in sampler))

    def test_patches_per_volume_must_divide_batch_size(self) -> None:
        keys, labels = make_keys_and_labels()
        with self.assertRaises(ValueError):
            VolumeGroupedBatchSampler(keys, labels, batch_size=32, patches_per_volume=48)


class TestClassBalance(unittest.TestCase):
    def test_positives_are_boosted_without_duplication(self) -> None:
        keys, labels = make_keys_and_labels(n_volumes=4, per_volume=200, n_pos=5)
        sampler = VolumeGroupedBatchSampler(keys, labels, batch_size=32, neg_per_pos=5.0, seed=3)
        for batch in sampler:
            self.assertEqual(len(set(batch)), len(batch))  # no duplicated patches
            positives = int(labels[batch].sum())
            self.assertGreater(positives, 0)
            # Natural rate is 2.5%; grouping must lift it well above that.
            self.assertGreater(positives / len(batch), 0.10)

    def test_volume_without_positives_still_yields_batches(self) -> None:
        keys = ["a"] * 100 + ["b"] * 100
        labels = np.zeros(200, dtype=np.uint8)
        labels[:4] = 1  # only volume "a" has positives
        sampler = VolumeGroupedBatchSampler(keys, labels, batch_size=16, seed=1)
        batches = list(sampler)
        self.assertGreater(len(batches), 0)
        self.assertTrue(any(labels[b].sum() == 0 for b in batches))

    def test_stratifies_positive_volumes_without_duplicate_patches(self) -> None:
        keys = []
        labels = []
        for volume in range(10):
            keys.extend([str(volume)] * 40)
            labels.extend(([1] * 4 + [0] * 36) if volume < 5 else [0] * 40)
        labels = np.asarray(labels, dtype=np.uint8)
        sampler = VolumeGroupedBatchSampler(
            keys, labels, batch_size=16, max_patches_per_epoch=64,
            positive_volume_fraction=0.75, seed=2,
        )
        batches = list(sampler)
        self.assertEqual(sum(int(labels[batch].sum() > 0) for batch in batches), 3)
        self.assertTrue(all(len(batch) == len(set(batch)) for batch in batches))

    def test_reuses_positive_patches_only_when_target_requires_it(self) -> None:
        keys, labels = make_keys_and_labels(n_volumes=1, per_volume=40, n_pos=3)
        sampler = VolumeGroupedBatchSampler(
            keys, labels, batch_size=32, positive_patches_per_volume=8, seed=5,
        )
        batch = next(iter(sampler))
        self.assertEqual(int(labels[batch].sum()), 8)
        self.assertEqual(len(batch), 32)
        self.assertEqual(len(set(batch)), 27)  # five controlled positive reuses

    def test_mixed_replay_uses_exact_hard_and_random_ratios(self) -> None:
        keys = ["positive"] * 160 + ["negative"] * 160
        labels = np.zeros(320, dtype=np.uint8)
        labels[:8] = 1
        hard = np.zeros(320, dtype=bool)
        hard[8:72] = True
        hard[160:224] = True
        sampler = VolumeGroupedBatchSampler(
            keys, labels, batch_size=32, positive_patches_per_volume=8,
            hard_negative_flags=hard,
            hard_negatives_per_positive_volume=12,
            hard_negatives_per_negative_volume=16,
            seed=4,
        )

        batches = {keys[batch[0]]: batch for batch in sampler}
        positive_batch = batches["positive"]
        negative_batch = batches["negative"]
        self.assertEqual(int(labels[positive_batch].sum()), 8)
        self.assertEqual(int(hard[positive_batch].sum()), 12)
        self.assertEqual(int(hard[negative_batch].sum()), 16)


class TestReproducibility(unittest.TestCase):
    def test_same_seed_and_epoch_give_same_batches(self) -> None:
        keys, labels = make_keys_and_labels()
        first = VolumeGroupedBatchSampler(keys, labels, batch_size=32, seed=11)
        second = VolumeGroupedBatchSampler(keys, labels, batch_size=32, seed=11)
        first.set_epoch(2)
        second.set_epoch(2)
        self.assertEqual(list(first), list(second))

    def test_different_epochs_give_different_batches(self) -> None:
        keys, labels = make_keys_and_labels()
        sampler = VolumeGroupedBatchSampler(keys, labels, batch_size=32, seed=11)
        sampler.set_epoch(0)
        epoch0 = list(sampler)
        sampler.set_epoch(1)
        self.assertNotEqual(epoch0, list(sampler))

    def test_sampler_is_picklable(self) -> None:
        keys, labels = make_keys_and_labels()
        sampler = VolumeGroupedBatchSampler(keys, labels, batch_size=32, seed=5)
        restored = pickle.loads(pickle.dumps(sampler))
        self.assertEqual(len(restored), len(sampler))


class TestBudgets(unittest.TestCase):
    def test_max_patches_per_epoch_caps_total(self) -> None:
        keys, labels = make_keys_and_labels(n_volumes=10)
        sampler = VolumeGroupedBatchSampler(
            keys, labels, batch_size=32, max_patches_per_epoch=128, seed=2
        )
        batches = list(sampler)
        self.assertEqual(len(batches), 4)
        self.assertEqual(sum(len(b) for b in batches), 128)

    def test_max_volumes_per_epoch_caps_volumes(self) -> None:
        keys, labels = make_keys_and_labels(n_volumes=10)
        sampler = VolumeGroupedBatchSampler(
            keys, labels, batch_size=32, max_volumes_per_epoch=3, seed=2
        )
        self.assertEqual(len({keys[b[0]] for b in sampler}), 3)


class TestSequentialSampler(unittest.TestCase):
    def test_visits_every_patch_once(self) -> None:
        keys, _ = make_keys_and_labels(n_volumes=3, per_volume=50)
        sampler = SequentialVolumeBatchSampler(keys, batch_size=16)
        seen = [i for batch in sampler for i in batch]
        self.assertEqual(sorted(seen), list(range(len(keys))))

    def test_batches_do_not_span_volumes(self) -> None:
        keys, _ = make_keys_and_labels(n_volumes=3, per_volume=50)
        for batch in SequentialVolumeBatchSampler(keys, batch_size=16):
            self.assertEqual(len({keys[i] for i in batch}), 1)

    def test_caps_patches_per_volume(self) -> None:
        keys, _ = make_keys_and_labels(n_volumes=3, per_volume=50)
        sampler = SequentialVolumeBatchSampler(keys, batch_size=8, max_patches_per_volume=16)
        self.assertEqual(sum(len(b) for b in sampler), 48)

    def test_validation_cap_keeps_every_positive_patch(self) -> None:
        keys, labels = make_keys_and_labels(n_volumes=2, per_volume=100, n_pos=5)
        sampler = SequentialVolumeBatchSampler(
            keys, batch_size=8, labels=labels, max_patches_per_volume=16, seed=9
        )
        selected = [index for batch in sampler for index in batch]
        self.assertEqual(len(selected), 32)
        self.assertEqual(int(labels[selected].sum()), 10)
        self.assertTrue(all(index in selected for index in np.flatnonzero(labels)))

    def test_validation_selection_is_deterministic_with_labels(self) -> None:
        keys, labels = make_keys_and_labels(n_volumes=3, per_volume=100, n_pos=3)
        first = SequentialVolumeBatchSampler(keys, batch_size=8, labels=labels, max_patches_per_volume=16, seed=4)
        second = SequentialVolumeBatchSampler(keys, batch_size=8, labels=labels, max_patches_per_volume=16, seed=4)
        self.assertEqual(list(first), list(second))


if __name__ == "__main__":
    unittest.main()
