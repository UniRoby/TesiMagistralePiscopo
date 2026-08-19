from tesi_m3d.dataset import M3DSynthRecord
from pathlib import Path
from tempfile import TemporaryDirectory

from tesi_m3d.isotropic import completed_records, select_records


def test_isotropic_selection_filters_before_limit():
    records = [
        M3DSynthRecord("real", "rem", "real", "a", "1", 0, 0, 0),
        M3DSynthRecord("cycle", "rem", "cycle", "b", "1", 0, 0, 0),
        M3DSynthRecord("pix", "inj", "pix2pix", "c", "1", 0, 0, 0),
    ]

    assert [record.img_id for record in select_records(records, {"pix2pix"}, 1)] == ["pix"]


def test_completed_records_requires_scan_and_label_markers():
    pix = M3DSynthRecord("pix", "inj", "pix2pix", "a", "1", 0, 0, 0)
    real = M3DSynthRecord("real", "rem", "real", "b", "2", 0, 0, 0)
    with TemporaryDirectory() as temp:
        root = Path(temp)
        for path in (root / "pix2pix" / "scan" / "pix", root / "pix2pix" / "label" / "pix", root / "real" / "scan" / "b__2"):
            path.mkdir(parents=True)
            (path / ".complete").write_text("ok")

        assert completed_records([pix, real], root) == [pix, real]

        (root / "pix2pix" / "label" / "pix" / ".complete").unlink()
        assert completed_records([pix, real], root) == [real]
