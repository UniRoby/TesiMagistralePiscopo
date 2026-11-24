"""Exact repeat traces and an explicit deterministic trilinear derivative."""
import copy
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from test_segmentation_padding_comparison import reference
from tesi_m3d import compare_segmentation_batches as batches
from tesi_m3d import compare_segmentation_padding as comparison
from tesi_m3d.diagnose_localization import write_csv
from tesi_m3d.segmentation import _TrilinearCPUBackward, file_digest
from tesi_m3d.train import set_global_seed


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("size", [(4, 6, 8), (3, 5, 7), (1, 1, 2), (2, 3, 4)])
def test_trilinear_forward_and_first_derivative_match_native_cpu(dtype, size):
    torch.manual_seed(21)
    x = torch.randn(1, 2, 2, 3, 4, dtype=dtype, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    expected = torch.nn.functional.interpolate(x, size=size, mode="trilinear", align_corners=False)
    actual = _TrilinearCPUBackward.apply(other, size)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn_like(expected)
    expected.backward(upstream); actual.backward(upstream)
    torch.testing.assert_close(other.grad, x.grad, rtol=0, atol=0)
    assert torch.isfinite(other.grad).all()


def test_trilinear_cpu_backward_passes_gradcheck():
    x = torch.randn(1, 1, 2, 2, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: _TrilinearCPUBackward.apply(t, (3, 4, 5)), (x,))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="laboratory CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_forward_is_native_and_strict_backward_matches_cpu(dtype, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with comparison.deterministic_execution():
        x = torch.randn(2, 3, 3, 4, 5, device="cuda", dtype=dtype, requires_grad=True)
        y = _TrilinearCPUBackward.apply(x, (7, 8, 9))
        native = torch.nn.functional.interpolate(x.detach(), size=(7, 8, 9), mode="trilinear", align_corners=False)
        torch.testing.assert_close(y, native, rtol=0, atol=0)
        gradient = torch.randn_like(y)
        expected = torch.zeros_like(x, device="cpu", requires_grad=True)
        torch.nn.functional.interpolate(expected, size=(7, 8, 9), mode="trilinear", align_corners=False).backward(gradient.cpu())
        y.backward(gradient)
        torch.testing.assert_close(x.grad.cpu(), expected.grad, rtol=0, atol=0)


def test_deterministic_context_restores_settings_even_on_error_and_seed_preserves_it(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    before = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
              torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        with comparison.deterministic_execution():
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
            monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
            monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
            set_global_seed(21)
            assert not torch.backends.cudnn.benchmark
            raise RuntimeError("synthetic failure")
    assert before == (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
                      torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    import os
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "invalid")
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        with comparison.deterministic_execution(): pytest.fail("invalid workspace must not run")


def test_two_identical_training_runs_reproduce_every_update_and_metric(reference, tmp_path, monkeypatch):
    config, root, source = reference
    before = {p: file_digest(p) for p in source.rglob("*") if p.is_file()}
    mode = torch.are_deterministic_algorithms_enabled()
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "outputs/segmentation_determinism_comparison"
    report = comparison.run_comparison(config, root, reference_dir=source, deterministic_repeats=True)
    assert report["complete"] and report["reproducible"] and report["first_divergence"] is None
    assert report["experiment"] == comparison.DETERMINISTIC_EXPERIMENT
    assert set(report["arms"]) == {"repeat_1", "repeat_2"}
    assert all(report["exact_file_matches"].values())
    assert report["runtime"]["deterministic_algorithms"] and not report["runtime"]["warn_only"]
    assert not report["runtime"]["cudnn_benchmark"] and report["runtime"]["trilinear_backward"] == "cpu_first_order"
    for name, arm in report["arms"].items():
        assert arm["deterministic"] and arm["decoder_padding_mode"] == "zeros"
        assert arm["updates"] == 6 and arm["exposures_per_crop"] == 1
        for row in comparison.read_rows(output / name / "update_metrics.csv"):
            assert len(row["model_state_sha256"]) == len(row["gradient_sha256"]) == 64
        for checkpoint in ("best", "last"):
            payload = torch.load(output / name / f"{checkpoint}.pt", weights_only=False)
            assert payload["execution_policy"] == {"strict_determinism": True, "trilinear_backward": "cpu_first_order"}
            assert arm["probability_diagnostics"][checkpoint]["reproduced_within_1e_4"]
            assert not arm["probability_diagnostics"][checkpoint]["cudnn_benchmark"]
    assert before == {p: file_digest(p) for p in source.rglob("*") if p.is_file()}
    assert torch.are_deterministic_algorithms_enabled() == mode
    assert json.loads((output / "comparison.json").read_text()) == report

    path = output / "repeat_2/update_metrics.csv"
    rows = comparison.read_rows(path)
    changed = copy.deepcopy(rows)
    changed[1]["loss"] = str(float(changed[1]["loss"]) + 1e-8)
    write_csv(path, changed)
    divergence = comparison.compare_repeats(output, 6)
    assert not divergence["reproducible"]
    assert divergence["first_divergence"]["updates"] == "2"
    assert set(divergence["first_divergence"]["differences"]) == {"loss"}
    changed = copy.deepcopy(rows)
    changed[0]["model_state_sha256"] = "different"
    write_csv(path, changed)
    assert comparison.compare_repeats(output, 6)["first_divergence"]["updates"] == "1"
    write_csv(path, rows[:-1])
    with pytest.raises(ValueError, match="full update trace"):
        comparison.compare_repeats(output, 6)


def test_unsupported_operation_is_reported_without_relaxing_determinism(reference, tmp_path, monkeypatch):
    config, root, source = reference
    mode = torch.are_deterministic_algorithms_enabled()

    def unsupported(*args, **kwargs):
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        raise RuntimeError("synthetic CUDA operation does not have a deterministic implementation")

    monkeypatch.setattr(batches, "train_volume", unsupported)
    output = tmp_path / "unsupported"
    with pytest.raises(RuntimeError, match="deterministic implementation"):
        comparison.run_comparison(config, root, reference_dir=source, output_dir=output, deterministic_repeats=True)
    report = json.loads((output / "comparison.json").read_text())
    assert not report["complete"] and report["reproducible"] is None and not report["passed"]
    assert report["active_arm"] == "repeat_1" and "synthetic CUDA operation" in report["error"]
    assert not (output / "repeat_2").exists()
    assert torch.are_deterministic_algorithms_enabled() == mode


@pytest.mark.parametrize("reproducible,passed,code", [(True, True, None), (True, False, 2), (False, False, 3)])
def test_deterministic_cli_distinguishes_reproducibility_from_localization(tmp_path, monkeypatch, reproducible, passed, code):
    import sys
    calls = []
    monkeypatch.setattr(comparison, "run_comparison", lambda *a, **kw: calls.append(kw) or {"reproducible": reproducible, "passed": passed})
    monkeypatch.setattr(sys, "argv", ["compare_segmentation_padding", "--data-root", str(tmp_path), "--deterministic-repeats"])
    if code is None:
        comparison.main()
    else:
        with pytest.raises(SystemExit) as stopped: comparison.main()
        assert stopped.value.code == code
    assert calls[0]["deterministic_repeats"] and calls[0]["output_dir"] == "outputs/segmentation_determinism_comparison"
