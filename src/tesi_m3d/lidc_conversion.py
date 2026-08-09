"""Portable conversion of downloaded LIDC-IDRI DICOM series to M3Dsynth TIFF."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ConversionJob:
    """Describe one unique LIDC DICOM series and its TIFF destination."""

    orig_id: str
    sdir_id: str
    source_dir: Path
    output_dir: Path


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV file and return all rows as dictionaries."""

    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def real_series_output_dir(output_root: str | Path, orig_id: str, sdir_id: str) -> Path:
    """Return the deterministic output directory used for one real CT series."""

    return Path(output_root) / "real" / "scan" / f"{orig_id}__{sdir_id}"


def resolve_dataset_root(output_root: str | Path | None, require_pix2pix: bool = True) -> Path:
    """Resolve the dataset root that must hold both ``pix2pix/`` and ``real/``.

    The training loader reads ``<data_root>/pix2pix/scan`` and ``<data_root>/real/scan``
    from one root, so writing ``real/`` next to the source code instead of next to
    ``pix2pix/`` silently breaks training with ``no TIFF slices found``. The presence of
    a ``pix2pix`` directory is therefore used as the dataset-root marker.

    When ``output_root`` is ``None`` the current directory and its parents are searched
    for that marker, so the converter lands in the right place regardless of where it
    is launched from.
    """

    if output_root is None:
        start = Path.cwd().resolve()
        for candidate in (start, *start.parents):
            if (candidate / "pix2pix").is_dir():
                return candidate
        raise FileNotFoundError(
            f"no dataset root containing 'pix2pix' found in {start} or its parents; "
            "pass --output-root explicitly"
        )

    root = Path(output_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"output root does not exist: {root}")
    if require_pix2pix and not (root / "pix2pix").is_dir():
        present = sorted(child.name for child in root.iterdir() if child.is_dir())
        raise FileNotFoundError(
            f"{root} does not look like the dataset root: no 'pix2pix' directory here "
            f"(found: {', '.join(present) if present else 'no subdirectories'}). "
            "Point --output-root at the folder that already contains pix2pix, "
            "or pass --allow-any-root to write anyway."
        )
    return root


def dicom_series_suffix(dicom_reference: str) -> str:
    """Extract the final UID suffix embedded in an official LIDC path."""

    dirname = PurePosixPath(dicom_reference).name
    return dirname.rsplit("-", maxsplit=1)[-1]


def _manifest_relative_path(manifest_path: str, patient_id: str) -> Path:
    """Extract the path beginning at PatientID from an IDC Windows manifest path."""

    parts = PureWindowsPath(manifest_path).parts
    patient_index = next(
        (index for index, part in enumerate(parts) if part.casefold() == patient_id.casefold()),
        None,
    )
    if patient_index is None:
        raise ValueError(f"PatientID {patient_id} not found in manifest path: {manifest_path}")
    return Path(*parts[patient_index:])


def resolve_dicom_directory(
    dicom_root: str | Path,
    lidc_row: dict[str, str],
    download_rows: Sequence[dict[str, str]],
) -> Path:
    """Resolve an official M3Dsynth LIDC row against the downloaded IDC layout.

    The official converter first expects the historical TCIA directory stored
    in ``LIDC.csv``.  IDC's current downloader uses compact UID suffix folders,
    so this function falls back to ``metadata.csv`` and matches PatientID plus
    the final digits of SeriesInstanceUID.
    """

    root = Path(dicom_root)
    direct_path = root / Path(PurePosixPath(lidc_row["dicom"]))
    if direct_path.is_dir():
        return direct_path

    patient_id = lidc_row["orig_id"]
    suffix = dicom_series_suffix(lidc_row["dicom"])
    matching_rows = [
        row
        for row in download_rows
        if row.get("PatientID", "").casefold() == patient_id.casefold()
        and row.get("SeriesInstanceUID", "").endswith(suffix)
        and row.get("completion_status", "success").casefold() == "success"
    ]

    candidates: list[Path] = []
    for row in matching_rows:
        manifest_path = row.get("S5cmdManifestPath", "")
        if not manifest_path:
            continue
        candidate = root / _manifest_relative_path(manifest_path, patient_id)
        if candidate.is_dir():
            candidates.append(candidate)

    unique_candidates = list(dict.fromkeys(candidates))
    if len(unique_candidates) == 1:
        return unique_candidates[0]
    if len(unique_candidates) > 1:
        raise RuntimeError(
            f"ambiguous DICOM match for {patient_id}/{lidc_row['sdir_id']}: {unique_candidates}"
        )
    raise FileNotFoundError(
        f"DICOM series not found for {patient_id}/{lidc_row['sdir_id']} "
        f"(expected SeriesInstanceUID suffix {suffix})"
    )


def build_conversion_jobs(
    dicom_root: str | Path,
    output_root: str | Path,
    metadata_dir: str | Path,
    download_metadata: str | Path,
    limit: int | None = None,
) -> list[ConversionJob]:
    """Build one conversion job for every real series required by M3Dsynth."""

    metadata_dir = Path(metadata_dir)
    data_rows = read_csv_rows(metadata_dir / "data.csv")
    lidc_rows = read_csv_rows(metadata_dir / "LIDC.csv")
    download_rows = read_csv_rows(download_metadata)
    required_pairs = {
        (row["orig_id"], row["sdir_id"])
        for row in data_rows
        if row["mod"].casefold() == "real"
    }
    selected_rows = [
        row for row in lidc_rows if (row["orig_id"], row["sdir_id"]) in required_pairs
    ]
    if limit is not None:
        selected_rows = selected_rows[:limit]

    jobs: list[ConversionJob] = []
    for row in selected_rows:
        jobs.append(
            ConversionJob(
                orig_id=row["orig_id"],
                sdir_id=row["sdir_id"],
                source_dir=resolve_dicom_directory(dicom_root, row, download_rows),
                output_dir=real_series_output_dir(output_root, row["orig_id"], row["sdir_id"]),
            )
        )
    return jobs


def scan_to_uint16(scan: np.ndarray) -> np.ndarray:
    """Convert a raw signed LIDC scan to the uint16 convention used by M3Dsynth."""

    if scan.dtype == np.uint16:
        return scan
    if scan.dtype != np.int16:
        raise TypeError(f"expected int16 or uint16 DICOM pixels, got {scan.dtype}")

    allowed_offsets = [0, 512, 1024, 2000, 2048]
    offset = int(-np.min(scan))
    if offset not in allowed_offsets:
        edge = min(5, scan.shape[0])
        # Corners usually contain scanner background and reveal the stored-value offset.
        corners = np.stack(
            (
                scan[:edge, :5, :5], scan[:edge, :5, -5:],
                scan[:edge, -5:, :5], scan[:edge, -5:, -5:],
                scan[-edge:, :5, :5], scan[-edge:, :5, -5:],
                scan[-edge:, -5:, :5], scan[-edge:, -5:, -5:],
            ),
            axis=0,
        )
        offset = int(-np.median(corners))
        if offset not in allowed_offsets:
            counts = [np.count_nonzero(corners == -candidate) for candidate in allowed_offsets]
            offset = allowed_offsets[int(np.argmax(counts))]
    return np.asarray(np.clip(scan, a_min=-offset, a_max=None) + offset, dtype=np.uint16)


def load_dicom_stack(source_dir: str | Path) -> np.ndarray:
    """Load and sort one DICOM series in the same z order as M3Dsynth."""

    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - optional conversion extra
        raise RuntimeError("pydicom is required; install the project conversion extra") from exc

    slices: list[tuple[float, np.ndarray]] = []
    for path in Path(source_dir).iterdir():
        if path.suffix.casefold() != ".dcm":
            continue
        dataset = pydicom.dcmread(path)
        if hasattr(dataset, "ImagePositionPatient"):
            order = -float(dataset.ImagePositionPatient[2])
        else:
            order = float(getattr(dataset, "InstanceNumber", len(slices)))
        slices.append((order, np.asarray(dataset.pixel_array)))
    if not slices:
        raise FileNotFoundError(f"no .dcm files found in {source_dir}")
    return np.stack([pixels for _, pixels in sorted(slices, key=lambda item: item[0])], axis=0)


def save_tiff_stack(output_dir: str | Path, scan: np.ndarray) -> None:
    """Save a uint16 volume as numbered lossless TIFF slices plus completion marker."""

    from PIL import Image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_slice in output_dir.glob("slide*.tiff"):
        stale_slice.unlink()
    for index, pixels in enumerate(scan):
        Image.fromarray(pixels).save(
            output_dir / f"slide{index:04d}.tiff",
            compression="tiff_lzma",
        )
    (output_dir / ".complete").write_text(f"slices={scan.shape[0]}\n", encoding="ascii")


def convert_job(job: ConversionJob) -> tuple[str, str]:
    """Convert one job and return its series identifier and status."""

    marker = job.output_dir / ".complete"
    if marker.exists():
        return (f"{job.orig_id}/{job.sdir_id}", "skipped")
    scan = scan_to_uint16(load_dicom_stack(job.source_dir))
    save_tiff_stack(job.output_dir, scan)
    return (f"{job.orig_id}/{job.sdir_id}", f"converted {scan.shape[0]} slices")


def run_conversion(jobs: Sequence[ConversionJob], workers: int = 1) -> None:
    """Execute conversion jobs serially or with Windows-safe worker processes."""

    if workers <= 1:
        for index, job in enumerate(jobs, start=1):
            series, status = convert_job(job)
            print(f"[{index}/{len(jobs)}] {series}: {status}")
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(convert_job, job): job for job in jobs}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            series, status = future.result()
            print(f"[{completed}/{len(jobs)}] {series}: {status}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for LIDC conversion."""

    parser = argparse.ArgumentParser(description="Convert downloaded LIDC DICOM series to M3Dsynth TIFF.")
    parser.add_argument("--dicom-root", required=True, help="Directory containing LIDC-IDRI patient folders.")
    parser.add_argument(
        "--output-root",
        help="Dataset root containing pix2pix/ and future real/. "
        "Detected automatically from the current directory upwards when omitted.",
    )
    parser.add_argument("--download-metadata", required=True, help="IDC download metadata.csv path.")
    parser.add_argument("--metadata-dir", default="metadata/m3dsynth", help="Official M3Dsynth CSV directory.")
    parser.add_argument("--workers", type=int, default=2, help="Parallel conversion processes.")
    parser.add_argument("--limit", type=int, help="Convert only the first N series for a smoke test.")
    parser.add_argument(
        "--allow-any-root",
        action="store_true",
        help="Skip the pix2pix/ check on --output-root; the training loader will not "
        "find the converted series unless they sit next to pix2pix/.",
    )
    return parser.parse_args()


def main() -> None:
    """Resolve all required series and convert them to TIFF."""

    args = parse_args()
    output_root = resolve_dataset_root(args.output_root, require_pix2pix=not args.allow_any_root)
    jobs = build_conversion_jobs(
        dicom_root=args.dicom_root,
        output_root=output_root,
        metadata_dir=args.metadata_dir,
        download_metadata=args.download_metadata,
        limit=args.limit,
    )
    print(f"Dataset root: {output_root}")
    print(f"Resolved {len(jobs)} DICOM series; output={output_root / 'real' / 'scan'}")
    run_conversion(jobs, workers=args.workers)


if __name__ == "__main__":
    main()
