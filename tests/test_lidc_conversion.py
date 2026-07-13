"""Tests for Windows IDC path matching used by the LIDC converter."""

from pathlib import Path

from tesi_m3d.lidc_conversion import (
    dicom_series_suffix,
    real_series_output_dir,
    resolve_dicom_directory,
)


def test_resolve_idc_manifest_path(tmp_path: Path) -> None:
    """Resolve an IDC compact folder using patient and SeriesInstanceUID suffix."""

    dicom_root = tmp_path / "lidc_idri"
    expected = dicom_root / "LIDC-IDRI-0014" / "38612" / "07402"
    expected.mkdir(parents=True)
    lidc_row = {
        "orig_id": "LIDC-IDRI-0014",
        "sdir_id": "3000562",
        "dicom": "LIDC-IDRI-0014/date/3000562.000000-NA-07402",
    }
    download_rows = [
        {
            "PatientID": "LIDC-IDRI-0014",
            "SeriesInstanceUID": "1.2.3.507402",
            "S5cmdManifestPath": r"C:\Tesi Magistrale Piscopo\lidc_idri\LIDC-IDRI-0014\38612\07402",
            "completion_status": "success",
        }
    ]

    assert resolve_dicom_directory(dicom_root, lidc_row, download_rows) == expected


def test_conversion_naming_is_stable(tmp_path: Path) -> None:
    """Use orig_id and sdir_id so one real source series is saved only once."""

    output = real_series_output_dir(tmp_path, "LIDC-IDRI-0014", "3000562")
    assert output == tmp_path / "real" / "scan" / "LIDC-IDRI-0014__3000562"
    assert dicom_series_suffix("patient/study/3000562.000000-NA-07402") == "07402"
