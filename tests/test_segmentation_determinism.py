"""Exact repeat traces and explicit deterministic spatial derivatives."""
import copy
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from test_segmentation_padding_comparison import reference
from test_model import encoder_weights
from tesi_m3d import compare_segmentation_batches as batches
from tesi_m3d import compare_segmentation_padding as comparison
from tesi_m3d.diagnose_localization import write_csv
from tesi_m3d.model import build_medicalnet_encoder
from tesi_m3d.segmentation import (_MaxPool3dCPUBackward, _TrilinearCPUBackward,
                                   MedicalNetSegmenter, file_digest, segmentation_loss)
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("pattern", ["random", "ties", "nonfinite"])
@pytest.mark.parametrize("pool", [torch.nn.MaxPool3d(3, 2, 1), torch.nn.MaxPool3d(2),
    torch.nn.MaxPool3d((2, 3, 2), (1, 2, 2), (1, 1, 0), dilation=(1, 2, 1), ceil_mode=True)])
def test_maxpool_forward_and_derivative_match_native_cpu(dtype, pattern, pool):
    torch.manual_seed(21)
    values = torch.randn(2, 3, 5, 7, 9, dtype=dtype)
    if pattern == "ties":
        values.zero_()
    elif pattern == "nonfinite":
        values[:, :, :2] = -float("inf")
        values[:, :, 2, 3, 4] = float("nan")
        values[:, :, -1, -1, -1] = float("inf")
    x = values.transpose(-1, -2).detach().requires_grad_()
    other = x.detach().clone().requires_grad_()
    with comparison.deterministic_execution():
        expected = pool(x)
        actual = _MaxPool3dCPUBackward.apply(other, pool)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        upstream = torch.randn_like(expected)
        expected.backward(upstream); actual.backward(upstream)
    torch.testing.assert_close(other.grad, x.grad, rtol=0, atol=0)
    assert torch.isfinite(other.grad).all()


def test_maxpool_cpu_backward_passes_gradcheck():
    x = torch.randn(1, 1, 3, 3, 3, dtype=torch.float64, requires_grad=True)
    pool = torch.nn.MaxPool3d(2, 1)
    assert torch.autograd.gradcheck(lambda t: _MaxPool3dCPUBackward.apply(t, pool), (x,))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="laboratory CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_maxpool_preserves_native_indices_and_has_strict_cpu_backward(dtype, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with comparison.deterministic_execution():
        x = torch.randn(2, 3, 5, 7, 9, device="cuda", dtype=dtype, requires_grad=True)
        pool = torch.nn.MaxPool3d(3, 2, 1)
        actual = _MaxPool3dCPUBackward.apply(x, pool)
        native, indices = torch.nn.functional.max_pool3d(x.detach(), 3, 2, 1, return_indices=True)
        torch.testing.assert_close(actual, native, rtol=0, atol=0)
        torch.testing.assert_close(actual, pool(x.detach()), rtol=0, atol=0)
        assert torch.equal(actual.grad_fn.saved_tensors[0], indices)
        cpu = x.detach().cpu().requires_grad_()
        expected = pool(cpu)
        gradient = torch.randn_like(actual)
        expected.backward(gradient.cpu()); actual.backward(gradient)
        torch.testing.assert_close(x.grad.cpu(), cpu.grad, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="laboratory CUDA required"))])
def test_actual_medicalnet_backward_is_strict_with_both_cpu_routes(encoder_weights, device, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with comparison.deterministic_execution():
        set_global_seed(21)
        encoder = build_medicalnet_encoder(initialize=False)
        encoder.load_state_dict({k.removeprefix("module."): v for k, v in encoder_weights.items()})
        model = MedicalNetSegmenter(encoder).to(device).train()
        model.deterministic_trilinear_backward = model.deterministic_maxpool_backward = True
        image = torch.randn(2, 1, 16, 16, 16, device=device)
        target = torch.zeros_like(image)
        target[0, :, 3:8, 4:9, 4:9] = 1
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(image)
            loss = segmentation_loss(logits, target)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert model.encoder.conv1.weight.grad.abs().sum() > 0
        assert model.output.weight.grad.abs().sum() > 0
        assert all(not m.training for m in model.encoder.modules() if isinstance(m, torch.nn.BatchNorm3d))
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()


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
    assert report["runtime"]["maxpool3d_backward"] == "cpu_first_order_native_indices"
    for name, arm in report["arms"].items():
        assert arm["deterministic"] and arm["decoder_padding_mode"] == "zeros"
        assert arm["updates"] == 6 and arm["exposures_per_crop"] == 1
        for row in comparison.read_rows(output / name / "update_metrics.csv"):
            assert len(row["model_state_sha256"]) == len(row["gradient_sha256"]) == 64
        for checkpoint in ("best", "last"):
            payload = torch.load(output / name / f"{checkpoint}.pt", weights_only=False)
            assert payload["execution_policy"] == {"strict_determinism": True, "trilinear_backward": "cpu_first_order",
                                                   "maxpool3d_backward": "cpu_first_order_native_indices"}
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
