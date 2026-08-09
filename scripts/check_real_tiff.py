"""Validate converted real CT TIFF stacks: readability, content and orientation.

Checks each ``real/scan/<series>`` directory produced by ``convert_lidc_to_tiff.py``:

- every ``slide*.tiff`` opens, is uint16 and shares one shape;
- numbering is contiguous and matches the ``.complete`` marker;
- no slice is empty (constant) and the stack has plausible CT contrast;
- orientation heuristics: air background at the corners, body in the centre,
  spine in the posterior half, lung air peaking in the middle of the stack.

Optionally writes PNG previews (axial montage plus coronal/sagittal reslices)
for a quick visual confirmation.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image

# Stored uint16 values: HU + 1024 for the usual LIDC rescale (slope 1, intercept -1024).
AIR_HU = -1024
WATER_HU = 0
BONE_HU = 300
LUNG_HU = -500
SLICE_PATTERN = re.compile(r"^slide(\d{4})\.tiff$")


def hu_to_stored(hu: float, offset: int = 1024) -> float:
    return hu + offset


def load_stack(series_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Read one series into a (z, y, x) array, returning per-file problems."""

    problems: list[str] = []
    indices: list[int] = []
    for path in sorted(series_dir.glob("slide*.tiff")):
        match = SLICE_PATTERN.match(path.name)
        if match is None:
            problems.append(f"unexpected slice name: {path.name}")
            continue
        indices.append(int(match.group(1)))

    if not indices:
        problems.append("no slide*.tiff found")
        return np.empty((0, 0, 0), dtype=np.uint16), problems

    indices.sort()
    expected = list(range(len(indices)))
    if indices != expected:
        missing = sorted(set(expected) - set(indices))
        problems.append(f"non contiguous numbering, missing/extra: {missing[:10]}")

    planes: list[np.ndarray] = []
    for index in indices:
        path = series_dir / f"slide{index:04d}.tiff"
        try:
            with Image.open(path) as image:
                image.load()
                if image.mode != "I;16":
                    problems.append(f"{path.name}: unexpected PIL mode {image.mode}")
                planes.append(np.asarray(image))
        except Exception as exc:  # noqa: BLE001 - report any decoding failure
            problems.append(f"{path.name}: unreadable ({type(exc).__name__}: {exc})")

    shapes = {plane.shape for plane in planes}
    if len(shapes) > 1:
        problems.append(f"inconsistent slice shapes: {sorted(shapes)}")
        return np.empty((0, 0, 0), dtype=np.uint16), problems

    stack = np.stack(planes, axis=0)
    if stack.dtype != np.uint16:
        problems.append(f"unexpected dtype {stack.dtype}, expected uint16")
    return stack, problems


def check_marker(series_dir: Path, slice_count: int) -> list[str]:
    marker = series_dir / ".complete"
    if not marker.exists():
        return [".complete marker missing (conversion may be incomplete)"]
    text = marker.read_text(encoding="ascii").strip()
    declared = int(text.split("=")[1]) if "=" in text else -1
    if declared != slice_count:
        return [f".complete declares {declared} slices but {slice_count} files are present"]
    return []


ALLOWED_OFFSETS = (0, 512, 1024, 2000, 2048, 3024, 3048, 3072)
LUNG_PEAK_HU = -1000


def estimate_offset(stack: np.ndarray) -> tuple[int, int]:
    """Infer the stored-value offset from the lung-air peak.

    The converter shifts signed DICOM values by ``-min(scan)``, and that minimum is
    the out-of-FOV padding rather than air, so the offset is series dependent and
    cannot be read back from the TIFF. Padding collapses to 0, so the dominant peak
    among the non-zero voxels of a chest CT is lung air at roughly -1000 HU.

    Returns the snapped offset and the raw stored value of that peak.
    """

    body = stack[stack > 0]
    if body.size == 0:
        return 0, 0
    peak = int(np.argmax(np.bincount(body.ravel())))
    guess = peak - LUNG_PEAK_HU
    return min(ALLOWED_OFFSETS, key=lambda candidate: abs(candidate - guess)), peak


def check_content(stack: np.ndarray, offset: int) -> list[str]:
    """Flag empty slices and stacks without CT-like tissue contrast."""

    problems: list[str] = []
    flat = stack.reshape(stack.shape[0], -1)
    constant = np.flatnonzero(flat.max(axis=1) == flat.min(axis=1))
    if constant.size:
        problems.append(f"{constant.size} constant/black slices: {constant[:10].tolist()}")

    soft = np.count_nonzero(
        (stack > hu_to_stored(-200, offset)) & (stack < hu_to_stored(200, offset))
    ) / stack.size
    bone = np.count_nonzero(stack > hu_to_stored(BONE_HU, offset)) / stack.size
    if soft < 0.02:
        problems.append(f"almost no soft tissue voxels ({soft:.4%}), image may be empty")
    if bone < 0.001:
        problems.append(f"almost no bone voxels ({bone:.4%}), unexpected for a chest CT")
    return problems


def body_mask(stack: np.ndarray, offset: int) -> np.ndarray:
    return stack > hu_to_stored(-300, offset)


def check_orientation(stack: np.ndarray, offset: int) -> tuple[list[str], dict[str, float]]:
    """Heuristics for axial in-plane orientation and z ordering."""

    problems: list[str] = []
    depth, height, width = stack.shape
    mid = stack[depth // 2]
    body = body_mask(stack, offset)

    # 1. Air outside the patient: corners of the middle slice must stay near air.
    corner = np.concatenate(
        [mid[:20, :20].ravel(), mid[:20, -20:].ravel(), mid[-20:, :20].ravel(), mid[-20:, -20:].ravel()]
    )
    corner_hu = float(np.median(corner)) - offset
    if corner_hu > -500:
        problems.append(f"corners are not air-like (median {corner_hu:.0f} HU): possible padding/flip issue")

    # 2. Body occupies the centre of the frame.
    cy, cx = height // 2, width // 2
    centre_frac = float(body[depth // 2, cy - 40 : cy + 40, cx - 40 : cx + 40].mean())
    if centre_frac < 0.5:
        problems.append(f"centre of the frame is mostly air ({centre_frac:.1%} tissue): body not centred")

    # 3. Spine is posterior: bone centroid should sit toward the lower half of the array
    #    (row 0 = anterior for the standard LPS axial layout). Allow 45% (some anatomic
    #    variation and noise at the boundaries) but flag if it's significantly anterior.
    lung_slices = slice(max(0, depth // 2 - 10), min(depth, depth // 2 + 10))
    bone = stack[lung_slices] > hu_to_stored(BONE_HU, offset)
    if bone.any():
        rows = np.nonzero(bone)[1]
        spine_row = float(np.median(rows)) / height
    else:
        spine_row = float("nan")
        problems.append("no bone voxels near the mid stack, cannot verify anterior/posterior")
    if spine_row == spine_row and spine_row < 0.45:
        problems.append(
            f"dense bone significantly anterior (row {spine_row:.2f}): "
            "possible vertical flip"
        )

    # 4. Lung air inside the body: should peak in the middle of the stack, not at the ends.
    lung = (stack > hu_to_stored(-950, offset)) & (stack < hu_to_stored(LUNG_HU, offset))
    inside = lung & _fill_body(body)
    profile = inside.reshape(depth, -1).mean(axis=1)
    peak = int(np.argmax(profile))
    if profile.max() < 0.01:
        problems.append(f"no lung-like air inside the body (max {profile.max():.4%})")
    elif peak < depth * 0.15 or peak > depth * 0.85:
        problems.append(f"lung air peaks at slice {peak}/{depth}: possible z ordering problem")

    metrics = {
        "corner_hu": corner_hu,
        "centre_tissue_frac": centre_frac,
        "spine_row_frac": spine_row,
        "lung_peak_slice": float(peak),
        "lung_peak_frac": float(profile.max()),
    }
    return problems, metrics


def _fill_body(body: np.ndarray) -> np.ndarray:
    """Cheap convex-ish body interior: everything between the first and last tissue voxel."""

    forward = np.maximum.accumulate(body, axis=2)
    backward = np.maximum.accumulate(body[:, :, ::-1], axis=2)[:, :, ::-1]
    horizontal = forward & backward
    forward = np.maximum.accumulate(body, axis=1)
    backward = np.maximum.accumulate(body[:, ::-1, :], axis=1)[:, ::-1, :]
    return horizontal & (forward & backward)


def window(image: np.ndarray, offset: int, centre: int = -600, width_hu: int = 1500) -> np.ndarray:
    low, high = centre - width_hu / 2 + offset, centre + width_hu / 2 + offset
    scaled = (image.astype(np.float32) - low) / (high - low)
    return (np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8)


def write_previews(stack: np.ndarray, offset: int, out_dir: Path, name: str) -> list[Path]:
    """Write an axial montage plus coronal and sagittal reslices."""

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    depth, height, width = stack.shape

    picks = np.linspace(depth * 0.1, depth * 0.9, 9).astype(int)
    tile = 256
    montage = Image.new("L", (tile * 3, tile * 3))
    for position, index in enumerate(picks):
        frame = Image.fromarray(window(stack[index], offset)).resize((tile, tile))
        montage.paste(frame, ((position % 3) * tile, (position // 3) * tile))
    axial = out_dir / f"{name}_axial_montage.png"
    montage.save(axial)
    written.append(axial)

    coronal = window(stack[:, height // 2, :], offset)
    sagittal = window(stack[:, :, width // 2], offset)
    for label, plane in (("coronal", coronal), ("sagittal", sagittal)):
        path = out_dir / f"{name}_{label}.png"
        Image.fromarray(plane).resize((512, 512)).save(path)
        written.append(path)
    return written


def check_series(series_dir: Path, preview_dir: Path | None) -> bool:
    print(f"\n=== {series_dir.name} ===")
    stack, problems = load_stack(series_dir)
    if stack.size == 0:
        for problem in problems:
            print(f"  FAIL {problem}")
        return False

    problems += check_marker(series_dir, stack.shape[0])
    offset, air_peak = estimate_offset(stack)
    problems += check_content(stack, offset)
    orientation_problems, metrics = check_orientation(stack, offset)
    problems += orientation_problems

    print(f"  shape (z,y,x)      : {stack.shape}")
    print(f"  dtype              : {stack.dtype}")
    print(f"  stored range       : {stack.min()}..{stack.max()}")
    print(f"  air peak (stored)  : {air_peak}")
    print(f"  inferred offset    : {offset} (HU = stored - offset)")
    print(f"  HU range           : {int(stack.min()) - offset}..{int(stack.max()) - offset}")
    print(f"  corner median HU   : {metrics['corner_hu']:.0f} (air ~ {AIR_HU})")
    print(f"  centre tissue frac : {metrics['centre_tissue_frac']:.1%}")
    print(f"  spine row fraction : {metrics['spine_row_frac']:.2f} (>0.5 = posterior, correct)")
    print(f"  lung air peak      : slice {metrics['lung_peak_slice']:.0f}/{stack.shape[0]}"
          f" ({metrics['lung_peak_frac']:.1%} of volume)")

    if preview_dir is not None:
        for path in write_previews(stack, offset, preview_dir, series_dir.name):
            print(f"  preview            : {path}")

    if problems:
        for problem in problems:
            print(f"  FAIL {problem}")
        return False
    print("  OK all checks passed")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan-root",
        default="real/scan",
        help="directory containing one subdirectory per converted series",
    )
    parser.add_argument("--preview-dir", default=None, help="write PNG previews here")
    parser.add_argument("--limit", type=int, default=None, help="check only the first N series")
    parser.add_argument(
        "--log-file",
        default="outputs/check_real_tiff.log",
        help="write full log to this file (default: outputs/check_real_tiff.log)",
    )
    args = parser.parse_args()

    scan_root = Path(args.scan_root)
    if not scan_root.is_dir():
        print(f"scan root not found: {scan_root.resolve()}")
        return 2

    series = sorted(path for path in scan_root.iterdir() if path.is_dir())
    if args.limit is not None:
        series = series[: args.limit]
    if not series:
        print(f"no series directories under {scan_root.resolve()}")
        return 2

    # Redirect stdout to log file while keeping stderr visible for progress
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    import sys
    log_file = log_path.open("w", encoding="utf-8")
    old_stdout = sys.stdout
    sys.stdout = log_file

    try:
        preview_dir = Path(args.preview_dir) if args.preview_dir else None
        failures = []

        # Progress indicator on stderr (visible to user)
        print(f"Checking {len(series)} series...", file=sys.stderr)
        for idx, path in enumerate(series, start=1):
            if not check_series(path, preview_dir):
                failures.append(path.name)
            # Progress every 10 or at the end
            if idx % 10 == 0 or idx == len(series):
                print(f"  [{idx}/{len(series)}] {100*idx//len(series):3d}%", file=sys.stderr)

        print(f"\n{len(series) - len(failures)}/{len(series)} series passed")
        if failures:
            print("failed: " + ", ".join(failures))
            result = 1
        else:
            result = 0
    finally:
        sys.stdout = old_stdout
        log_file.close()

    # Print summary to stdout
    print(f"\n{'='*70}")
    print(f"Check completed. Results written to: {log_path.resolve()}")
    print(f"{'='*70}")

    # Also print the summary from the log
    with log_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[-5:]:
            if line.strip():
                print(line.rstrip())

    return result


if __name__ == "__main__":
    raise SystemExit(main())
