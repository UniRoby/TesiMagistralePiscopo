"""Tests for Windows IDC path matching used by the LIDC converter."""

from pathlib import Path

import pytest

from tesi_m3d.lidc_conversion import (
    dicom_series_suffix,
    real_series_output_dir,
    resolve_dataset_root,
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


def test_dataset_root_accepts_folder_holding_pix2pix(tmp_path: Path) -> None:
    """An explicit root is kept when it carries the pix2pix marker."""

    (tmp_path / "pix2pix").mkdir()

    assert resolve_dataset_root(tmp_path) == tmp_path.resolve()


def test_dataset_root_rejects_folder_without_pix2pix(tmp_path: Path) -> None:
    """Writing real/ away from pix2pix/ breaks the loader, so it must fail loudly."""

    project = tmp_path / "TesiMagistralePiscopo"
    (project / "src").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="does not look like the dataset root"):
        resolve_dataset_root(project)
    assert resolve_dataset_root(project, require_pix2pix=False) == project.resolve()


def test_dataset_root_is_detected_from_a_subdirectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omitting --output-root walks up from the cwd to the folder holding pix2pix."""

    (tmp_path / "pix2pix").mkdir()
    launch_dir = tmp_path / "TesiMagistralePiscopo" / "scripts"
    launch_dir.mkdir(parents=True)
    monkeypatch.chdir(launch_dir)

    assert resolve_dataset_root(None) == tmp_path.resolve()


def test_dataset_root_detection_fails_without_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Auto-detection must not guess a root when no pix2pix/ exists anywhere above."""

    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="no dataset root containing 'pix2pix'"):
        resolve_dataset_root(None)
