from tesi_m3d.dataset import M3DSynthRecord
from tesi_m3d.isotropic import select_records


def test_isotropic_selection_filters_before_limit():
    records = [
        M3DSynthRecord("real", "rem", "real", "a", "1", 0, 0, 0),
        M3DSynthRecord("cycle", "rem", "cycle", "b", "1", 0, 0, 0),
        M3DSynthRecord("pix", "inj", "pix2pix", "c", "1", 0, 0, 0),
    ]

    assert [record.img_id for record in select_records(records, {"pix2pix"}, 1)] == ["pix"]
