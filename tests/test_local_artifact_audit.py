import csv

import numpy as np
from PIL import Image

from tesi_m3d.audit_local_artifacts import LocalArtifactAuditor


def _write_stack(directory, stack):
    directory.mkdir(parents=True)
    for index, plane in enumerate(stack):
        Image.fromarray(plane.astype(np.uint16)).save(directory / f"slide{index:04d}.tiff")


def test_local_audit_separates_mask_near_and_far(tmp_path):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    with (metadata / "data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["img_id", "ty", "mod", "orig_id", "sdir_id", "coord_z", "coord_y", "coord_x"])
        writer.writeheader()
        for mod in ("real", "pix2pix"):
            writer.writerow({"img_id": "sample", "ty": "add", "mod": mod, "orig_id": "patient",
                             "sdir_id": "series", "coord_z": 1, "coord_y": 7, "coord_x": 7})

    real = np.full((3, 16, 16), 100, dtype=np.uint16)
    fake = real.copy()
    fake[1, 7, 7] += 20
    fake[1, 7, 8] += 5
    fake[0, 15, 15] += 50
    mask = np.zeros((4, 16, 16), dtype=np.uint16)  # Extra empty label slice.
    mask[1, 7, 7] = 1
    _write_stack(tmp_path / "real" / "scan" / "sample", real)
    _write_stack(tmp_path / "pix2pix" / "scan" / "sample", fake)
    _write_stack(tmp_path / "pix2pix" / "label" / "sample", mask)

    output = tmp_path / "report"
    report = LocalArtifactAuditor(tmp_path, metadata, max_pairs=1).run(output)
    with (output / "regions.csv").open(newline="") as handle:
        rows = {row["region"]: row for row in csv.DictReader(handle)}

    assert report["n_pairs"] == 1
    assert set(rows) == {"mask", "near", "far"}
    assert all(int(row["changed_voxels"]) == 1 for row in rows.values())
    assert float(rows["mask"]["mean_absolute_difference"]) == 20
    assert int(rows["far"]["max_absolute_difference"]) == 50
    assert report["regions"]["near"]["mean_gradient_difference_xy"] > 0
    assert (output / "previews" / "sample.png").exists()
    assert (output / "previews" / "sample_outside.png").exists()
