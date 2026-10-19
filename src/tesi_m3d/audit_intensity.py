"""Compare stored and model-visible intensity distributions of real/pix2pix CT pairs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .dataset import load_tiff_stack, read_records, scan_dir
from .lung import estimate_ct_offset_from_histogram


def _histogram_percentile(histogram: np.ndarray, percentile: float) -> float:
    cumulative = np.cumsum(histogram, dtype=np.int64)
    if cumulative[-1] == 0:
        return 0.0
    target = percentile / 100.0 * (cumulative[-1] - 1)
    return float(np.searchsorted(cumulative, target, side="right"))


def _volume_statistics(scan: np.ndarray) -> tuple[dict[str, float | int], np.ndarray]:
    """Calculate exact uint16 and post-normalization statistics from one histogram."""

    scan = np.asarray(scan)
    if scan.ndim != 3 or not np.issubdtype(scan.dtype, np.integer):
        raise ValueError("intensity audit requires a 3-D integer TIFF volume")
    histogram = np.bincount(scan.reshape(-1), minlength=65536).astype(np.float64)
    values = np.arange(histogram.size, dtype=np.float64)
    total = float(histogram.sum())
    nonzero_histogram = histogram.copy()
    nonzero_histogram[0] = 0
    nonzero_total = float(nonzero_histogram.sum())
    source = nonzero_histogram if nonzero_total else histogram
    low = _histogram_percentile(source, 1.0)
    median = _histogram_percentile(source, 50.0)
    high = _histogram_percentile(source, 99.0)
    scale = high - low
    normalized = np.zeros_like(values) if scale <= 0 else (values - low) / scale
    normalized_mean = float(np.sum(histogram * normalized) / total)
    normalized_variance = float(np.sum(histogram * (normalized - normalized_mean) ** 2) / total)
    clipped = np.clip(normalized, 0.0, 1.0)
    normalized_histogram, _ = np.histogram(
        clipped, bins=128, range=(0.0, 1.0), weights=histogram
    )
    normalized_histogram = normalized_histogram / normalized_histogram.sum()
    raw_mean = float(np.sum(histogram * values) / total)
    raw_variance = float(np.sum(histogram * (values - raw_mean) ** 2) / total)
    offset, air_peak = estimate_ct_offset_from_histogram(histogram)
    nonzero_values = np.flatnonzero(nonzero_histogram)
    stats: dict[str, float | int] = {
        "depth": int(scan.shape[0]), "height": int(scan.shape[1]), "width": int(scan.shape[2]),
        "stored_min": int(np.min(scan)), "stored_max": int(np.max(scan)),
        "stored_mean": raw_mean, "stored_std": float(np.sqrt(raw_variance)),
        "zero_fraction": float(histogram[0] / total),
        "nonzero_min": int(nonzero_values[0]) if nonzero_values.size else 0,
        "nonzero_p01": low, "nonzero_p50": median, "nonzero_p99": high,
        "estimated_offset": offset, "air_peak_stored": air_peak,
        "normalized_mean": normalized_mean,
        "normalized_std": float(np.sqrt(normalized_variance)),
        "normalized_below_zero_fraction": float(histogram[normalized < 0].sum() / total),
        "normalized_above_one_fraction": float(histogram[normalized > 1].sum() / total),
    }
    return stats, normalized_histogram


def _jensen_shannon(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    middle = (first + second) / 2
    first_term = np.sum(first[first > 0] * np.log(first[first > 0] / middle[first > 0]))
    second_term = np.sum(second[second > 0] * np.log(second[second > 0] / middle[second > 0]))
    return float((first_term + second_term) / 2)


class IntensityAuditor:
    """Audit matched real/pix2pix volumes without changing either dataset."""

    def __init__(
        self,
        data_root: str | Path,
        metadata_dir: str | Path = "metadata/m3dsynth",
        split: str | None = None,
        max_pairs: int | None = 32,
        seed: int = 21,
        real_scan_root: str | Path | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.metadata_dir = Path(metadata_dir)
        self.split = split
        if max_pairs is not None and max_pairs < 1:
            raise ValueError("max_pairs must be at least 1 or None")
        self.max_pairs = max_pairs
        self.seed = int(seed)
        self.real_scan_root = Path(real_scan_root) if real_scan_root else None

    def _pairs(self):
        records = read_records(self.metadata_dir)
        real = {record.img_id: record for record in records if record.is_real}
        pix2pix = {
            record.img_id: record for record in records
            if record.mod == "pix2pix" and (self.split is None or record.split == self.split)
        }
        pairs = [(real[img_id], pix2pix[img_id]) for img_id in sorted(pix2pix) if img_id in real]
        if not pairs:
            raise ValueError("no matched real/pix2pix img_id pairs found")
        if self.max_pairs is not None and len(pairs) > self.max_pairs:
            generator = np.random.default_rng(self.seed)
            chosen = np.sort(generator.choice(len(pairs), size=self.max_pairs, replace=False))
            pairs = [pairs[int(index)] for index in chosen]
        return pairs

    @staticmethod
    def _domain_summary(rows: list[dict]) -> dict:
        names = (
            "zero_fraction", "nonzero_p01", "nonzero_p50", "nonzero_p99",
            "estimated_offset", "normalized_mean", "normalized_std",
            "normalized_below_zero_fraction", "normalized_above_one_fraction",
        )
        return {
            name: {
                "mean": float(np.mean([row[name] for row in rows])),
                "std": float(np.std([row[name] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
            }
            for name in names
        }

    @staticmethod
    def _comparison(real_rows, fake_rows, real_histograms, fake_histograms) -> dict:
        real_distribution = np.mean(real_histograms, axis=0)
        fake_distribution = np.mean(fake_histograms, axis=0)
        ks = float(np.max(np.abs(np.cumsum(real_distribution) - np.cumsum(fake_distribution))))
        smd = {}
        zero_variance_shift = False
        for name in ("normalized_mean", "normalized_std"):
            first = np.asarray([row[name] for row in real_rows], dtype=np.float64)
            second = np.asarray([row[name] for row in fake_rows], dtype=np.float64)
            pooled = float(np.sqrt((np.var(first) + np.var(second)) / 2))
            difference = float(np.mean(second) - np.mean(first))
            zero_variance_shift |= pooled == 0 and difference != 0
            smd[name] = 0.0 if pooled == 0 else difference / pooled
        js = _jensen_shannon(real_distribution, fake_distribution)
        detected = (
            zero_variance_shift or js > 0.05 or ks > 0.10
            or any(abs(value) > 0.5 for value in smd.values())
        )
        return {
            "normalized_histogram_js_divergence": js,
            "normalized_histogram_ks_distance": ks,
            "standardized_mean_difference": smd,
            "offset_match_rate": float(np.mean([
                real_row["estimated_offset"] == fake_row["estimated_offset"]
                for real_row, fake_row in zip(real_rows, fake_rows)
            ])),
            "shape_match_rate": float(np.mean([
                all(real_row[axis] == fake_row[axis] for axis in ("depth", "height", "width"))
                for real_row, fake_row in zip(real_rows, fake_rows)
            ])),
            "heuristic_result": "distribution_shift_detected" if detected else "no_large_shift_detected",
            "heuristic_thresholds": {"js_divergence": 0.05, "ks_distance": 0.10, "abs_smd": 0.5},
        }

    def run(self, output_dir: str | Path) -> dict:
        """Run the paired audit and write ``volumes.csv``, ``pairs.csv`` and JSON."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        volume_rows, pair_rows = [], []
        rows_by_domain = {"real": [], "pix2pix": []}
        histograms = {"real": [], "pix2pix": []}
        for real_record, fake_record in self._pairs():
            pair_stats = {}
            for domain, record in (("real", real_record), ("pix2pix", fake_record)):
                directory = scan_dir(self.data_root, record, self.real_scan_root)
                stats, histogram = _volume_statistics(load_tiff_stack(directory))
                row = {"img_id": record.img_id, "domain": domain, "path": str(directory), **stats}
                volume_rows.append(row)
                rows_by_domain[domain].append(row)
                histograms[domain].append(histogram)
                pair_stats[domain] = stats
            pair_rows.append({
                "img_id": real_record.img_id,
                "shape_match": all(
                    pair_stats["real"][axis] == pair_stats["pix2pix"][axis]
                    for axis in ("depth", "height", "width")
                ),
                "real_offset": pair_stats["real"]["estimated_offset"],
                "pix2pix_offset": pair_stats["pix2pix"]["estimated_offset"],
                "delta_nonzero_p01": (
                    pair_stats["pix2pix"]["nonzero_p01"] - pair_stats["real"]["nonzero_p01"]
                ),
                "delta_nonzero_p50": (
                    pair_stats["pix2pix"]["nonzero_p50"] - pair_stats["real"]["nonzero_p50"]
                ),
                "delta_nonzero_p99": (
                    pair_stats["pix2pix"]["nonzero_p99"] - pair_stats["real"]["nonzero_p99"]
                ),
                "delta_normalized_mean": (
                    pair_stats["pix2pix"]["normalized_mean"] - pair_stats["real"]["normalized_mean"]
                ),
                "delta_normalized_std": (
                    pair_stats["pix2pix"]["normalized_std"] - pair_stats["real"]["normalized_std"]
                ),
                "delta_zero_fraction": (
                    pair_stats["pix2pix"]["zero_fraction"] - pair_stats["real"]["zero_fraction"]
                ),
            })

        comparison = self._comparison(
            rows_by_domain["real"], rows_by_domain["pix2pix"],
            histograms["real"], histograms["pix2pix"],
        )
        report = {
            "n_pairs": len(pair_rows), "split": self.split,
            "normalization": "same non-zero p01/p99 scaling used by the training dataset",
            "domains": {name: self._domain_summary(rows) for name, rows in rows_by_domain.items()},
            "comparison": comparison,
            "interpretation": "The heuristic is a screening test, not proof that the domains are indistinguishable.",
        }
        for filename, rows in (("volumes.csv", volume_rows), ("pairs.csv", pair_rows)):
            with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--metadata-dir", default="metadata/m3dsynth")
    parser.add_argument("--split", choices=("train", "valid", "test"))
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--real-scan-root")
    parser.add_argument("--out-dir", default="outputs/intensity_audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = IntensityAuditor(
        args.data_root, args.metadata_dir, args.split, args.max_pairs, args.seed, args.real_scan_root
    ).run(args.out_dir)
    print(json.dumps(report, indent=2))
    print(f"Intensity audit written to {args.out_dir}")


if __name__ == "__main__":
    main()
