"""Helpers to build a fake M3Dsynth corpus on disk for tests.

Not a test module. It is a plain helper rather than a pytest ``conftest``
fixture because the suite is written with ``unittest.TestCase``, which cannot
receive pytest fixtures as arguments.

The Pillow modes match the real corpus exactly (verified against the dataset):
scan slices are mode ``I;16``, label slices are mode ``1``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tesi_m3d.dataset import M3DSynthRecord
from tesi_m3d.volume_io import SLICE_TEMPLATE


def write_tiff_stack(dirpath: str | Path, volume: np.ndarray) -> Path:
    """Write a 3D array as ``slide0000.tiff``, ``slide0001.tiff``, ..."""

    from PIL import Image

    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    if volume.ndim != 3:
        raise ValueError("volume must be 3D in z, y, x order")
    for index, plane in enumerate(volume):
        if plane.dtype == bool:
            image = Image.fromarray(plane.astype(np.uint8) * 255).convert("1")
        else:
            image = Image.fromarray(plane.astype(np.uint16))  # Pillow infers mode I;16
        image.save(dirpath / SLICE_TEMPLATE.format(index))
    return dirpath


def make_fake_record(
    img_id: str = "inj_1",
    mod: str = "pix2pix",
    orig_id: str = "LIDC-IDRI-0001",
    sdir_id: str = "3000001",
    split: str | None = "train",
) -> M3DSynthRecord:
    """Return a record with plausible coordinates for tests."""

    return M3DSynthRecord(
        img_id=img_id,
        ty="inj",
        mod=mod,
        orig_id=orig_id,
        sdir_id=sdir_id,
        coord_z=10,
        coord_y=20,
        coord_x=20,
        split=split,
    )


def make_fake_corpus(
    root: str | Path,
    n_records: int = 3,
    scan_z: int = 40,
    shape: tuple[int, int] = (64, 64),
    mask_z_offset: int = 1,
    cube_origin: tuple[int, int, int] = (8, 8, 8),
    cube_size: tuple[int, int, int] = (16, 16, 16),
    n_real: int = 0,
    real_shared_dirs: int = 1,
) -> tuple[Path, list[M3DSynthRecord]]:
    """Build ``<root>/pix2pix/{scan,label}/<img_id>/`` plus optional real scans.

    ``mask_z_offset=1`` reproduces the real pix2pix defect where every label
    stack carries one extra z-slice, which is what made the old mask-derived
    patch grid emit coordinates past the end of the scan.

    ``n_real`` records are spread over ``real_shared_dirs`` directories to
    reproduce the many-records-one-directory layout of the converted LIDC data.
    """

    root = Path(root)
    records: list[M3DSynthRecord] = []
    height, width = shape
    rng = np.random.default_rng(0)

    for i in range(n_records):
        img_id = f"inj_{i + 1}"
        scan = rng.integers(0, 4000, size=(scan_z, height, width), dtype=np.uint16)
        write_tiff_stack(root / "pix2pix" / "scan" / img_id, scan)

        mask = np.zeros((scan_z + mask_z_offset, height, width), dtype=bool)
        z0, y0, x0 = cube_origin
        dz, dy, dx = cube_size
        mask[z0 : z0 + dz, y0 : y0 + dy, x0 : x0 + dx] = True
        write_tiff_stack(root / "pix2pix" / "label" / img_id, mask)

        records.append(make_fake_record(img_id=img_id, orig_id=f"LIDC-IDRI-{i + 1:04d}"))

    for i in range(n_real):
        shared = i % real_shared_dirs
        orig_id = f"LIDC-REAL-{shared:04d}"
        sdir_id = f"900{shared}"
        real_dir = root / "real" / "scan" / f"{orig_id}__{sdir_id}"
        if not real_dir.exists():
            scan = rng.integers(0, 4000, size=(scan_z, height, width), dtype=np.uint16)
            write_tiff_stack(real_dir, scan)
        records.append(
            M3DSynthRecord(
                img_id=f"rem_{i + 1}",
                ty="rem",
                mod="real",
                orig_id=orig_id,
                sdir_id=sdir_id,
                coord_z=10,
                coord_y=20,
                coord_x=20,
                split="train",
            )
        )

    return root, records
