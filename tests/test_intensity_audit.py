import csv
import json

import numpy as np
from PIL import Image

from tesi_m3d.audit_intensity import IntensityAuditor, _volume_statistics
from tesi_m3d.lung import lung_mask_from_ct


def _write_stack(directory, stack):
    directory.mkdir(parents=True)
    for index, plane in enumerate(stack):
        Image.fromarray(plane.astype(np.uint16)).save(directory / f"slide{index:04d}.tiff")


def test_intensity_auditor_writes_matched_domain_report(tmp_path):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    fields = ["img_id", "ty", "mod", "orig_id", "sdir_id", "coord_z", "coord_y", "coord_x"]
    with (metadata / "data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        base = {"img_id": "sample", "ty": "add", "orig_id": "patient", "sdir_id": "series", "coord_z": 0, "coord_y": 0, "coord_x": 0}
        writer.writerow({**base, "mod": "real"})
        writer.writerow({**base, "mod": "pix2pix"})
    real = np.array([[[0, 100, 200], [300, 400, 500]]], dtype=np.uint16)
    fake = real.copy()
    fake[fake > 0] += 512
    _write_stack(tmp_path / "real" / "scan" / "sample", real)
    _write_stack(tmp_path / "pix2pix" / "scan" / "sample", fake)

    output = tmp_path / "report"
    report = IntensityAuditor(tmp_path, metadata, max_pairs=1).run(output)

    assert report["n_pairs"] == 1
    assert (output / "volumes.csv").exists()
    assert json.loads((output / "summary.json").read_text())["n_pairs"] == 1


def test_volume_statistics_match_training_percentile_normalization():
    scan = np.arange(100, dtype=np.uint16).reshape(4, 5, 5)

    stats, histogram = _volume_statistics(scan)

    assert stats["zero_fraction"] == 0.01
    assert np.isclose(histogram.sum(), 1.0)
    assert stats["normalized_std"] > 0


def test_lung_mask_excludes_exterior_air_and_keeps_both_lungs():
    scan = np.zeros((8, 32, 32), dtype=np.uint16)
    scan[:, 2:30, 2:30] = 3072
    scan[:, 8:18, 6:14] = 2072
    scan[:, 8:18, 18:26] = 2072

    lungs = lung_mask_from_ct(scan)

    assert lungs[:, 10:16, 8:12].all()
    assert lungs[:, 10:16, 20:24].all()
    assert not lungs[:, 0, 0].any()
