"""Focused protocol tests; official weights integration is opt-in via env paths."""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.nn import functional as F

from tesi_m3d import trufor_data as data
from tesi_m3d import trufor_evaluation as evaluation
from tesi_m3d import trufor_training as training
from tesi_m3d.trufor_model import TruForCT, SliceFusion, localization_loss


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, rgb, noise):
        x = self.dropout(self.conv(rgb + noise))
        return [F.avg_pool2d(x, factor) for factor in (2, 4, 8, 16)]


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(4, 2, 1)

    def forward(self, features):
        x = sum(F.interpolate(f, size=features[0].shape[-2:]) for f in features)
        return self.head(x)


class TinyNetwork(nn.Module):
    channels = (4, 4, 4, 4)

    def __init__(self):
        super().__init__()
        self.dncnn = nn.Sequential(nn.Conv2d(3, 1, 1), nn.BatchNorm2d(1))
        self.backbone = TinyBackbone()
        self.decode_head = TinyDecoder()
        self.prepro = nn.Identity()


def tiny_model(_root=None, context=False, official_weights=None):
    return TruForCT(TinyNetwork(), context)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def write_stack(path, arrays):
    path.mkdir(parents=True)
    for i, a in enumerate(arrays):
        Image.fromarray(a).save(path / f"slide{i:04d}.tiff")


@pytest.fixture
def manifest(tmp_path):
    entries = []
    for split in ("train", "valid", "test"):
        for real in (False, True):
            root = tmp_path / split / str(real)
            scan, mask = root / "scan", root / "mask"
            image = np.arange(1024, dtype=np.uint16).reshape(32, 32)
            write_stack(scan, [image, image[::-1]])
            if not real:
                m = np.zeros((32, 32), dtype=bool)
                m[8:24, 8:24] = True
                write_stack(mask, [m, np.zeros_like(m), np.zeros_like(m)])
            info = data.inspect_scan(scan, tmp_path / "cache", None if real else mask)
            entries.append(dict(id=f"{split}/{real}", scan=str(scan), mask=None if real else str(mask),
                                orig_id=split, sdir_id=str(real), real=real, kind="real" if real else "inj",
                                split=split, spacing=[2.5, 1, 1], **info))
    return dict(version=1, entries=entries, fingerprint=data.digest(entries))


def test_percentile_cache_and_invalidation(tmp_path, monkeypatch):
    arrays = [np.array([[0, 2], [5, 65535]], dtype=np.uint16),
              np.array([[1, 2], [5, 8]], dtype=np.uint16)]
    scan = tmp_path / "scan"
    write_stack(scan, arrays)
    first = data.inspect_scan(scan, tmp_path / "cache")
    expected = np.percentile(np.concatenate([a[a > 0] for a in arrays]), [1, 99])
    np.testing.assert_allclose(first["bounds"], expected)
    original = Image.open
    monkeypatch.setattr(Image, "open", lambda *_: pytest.fail("Cached percentiles reread TIFF pixels"))
    assert data.inspect_scan(scan, tmp_path / "cache") == first
    monkeypatch.setattr(Image, "open", original)
    Image.fromarray(np.ones((2, 2), dtype=np.uint16)).save(scan / "slide0000.tiff")
    second = data.inspect_scan(scan, tmp_path / "cache")
    assert first["signature"] != second["signature"]


def test_mask_alignment_and_gaps(tmp_path):
    image = np.ones((4, 4), dtype=np.uint16)
    scan, mask = tmp_path / "scan", tmp_path / "mask"
    write_stack(scan, [image, image])
    write_stack(mask, [image > 0, image == 0, image == 0])
    assert data.inspect_scan(scan, tmp_path / "cache", mask)["trimmed_empty_masks"] == 1
    Image.fromarray(image > 0).save(mask / "slide0002.tiff")
    with pytest.raises(ValueError, match="discard positive"):
        data.inspect_scan(scan, tmp_path / "cache", mask)
    (scan / "slide0000.tiff").unlink()
    with pytest.raises(ValueError, match="Non-contiguous"):
        data.slice_paths(scan)


def test_sample_coverage_context_and_workers(manifest):
    entries = [e for e in manifest["entries"] if e["split"] == "train"]
    tokens = data.epoch_samples(entries, 0, 42)
    assert len(tokens) == 8 and {t[0] for t in tokens} == {0, 1}
    assert tokens == data.epoch_samples(entries, 0, 42)
    assert tokens != data.epoch_samples(entries, 1, 42)
    image, valid, distances = data.slice_input(entries[0], 0, True)
    assert valid.tolist() == [False, True, True]
    assert distances.tolist() == [-2.5, 0, 2.5]
    assert image.shape == (3, 1, 32, 32)
    dataset = data.SliceDataset(entries, True)
    # Exercise Windows-style spawning even when this test runs on Linux.
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=tokens,
               num_workers=4, multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=1,
               timeout=30)
    for batch, token in zip(loader, tokens):
        for expected, actual in zip(dataset[token], batch):
            torch.testing.assert_close(actual[0], expected)
    for batch, token in zip(loader, tokens):
        torch.testing.assert_close(batch[0][0], dataset[token][0])


def test_initial_context_identity_gradients_and_padding():
    baseline = tiny_model().eval()
    context = tiny_model(context=True).eval()
    context.network.load_state_dict(baseline.network.state_dict())
    images = torch.rand(1, 3, 1, 35, 37)
    valid, distance = torch.ones(1, 3, dtype=torch.bool), torch.tensor([[-2., 0., 2.]])
    a = baseline(images[:, 1:2], valid[:, 1:2], distance[:, 1:2])
    b = context(images, valid, distance)
    assert a.shape == (1, 2, 35, 37)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    context.train()
    assert not context.network.dncnn.training
    target = torch.zeros((1, 35, 37), dtype=torch.long)
    target[:, 10:20, 10:20] = 1
    localization_loss(context(images, valid, distance), target).backward()
    assert all(g.scale.grad is not None and torch.isfinite(g.scale.grad) for g in context.fusion)
    assert all(p.grad is None for p in context.network.dncnn.parameters())
    context.zero_grad()
    for g in context.fusion:
        g.scale.data.fill_(0.1)
    localization_loss(context(images, valid, distance), target).backward()
    assert any(g.gate.weight.grad.abs().sum() > 0 for g in context.fusion)
    gate = SliceFusion(4)
    gate.scale.data.fill_(1)
    center = torch.rand(1, 4, 3, 3)
    actual = gate(center, (center + 10, center - 3), torch.zeros(1, 2), torch.zeros(1, 2))
    torch.testing.assert_close(actual, center)


def test_counts_match_direct_thresholding_and_empty_loss():
    p = np.array([[0, .01, .50], [.99, 1., .497]], dtype=np.float32)
    mask = np.array([[False, True, True], [False, True, False]])
    counts = evaluation.threshold_counts(p, mask)
    for i, threshold in enumerate(evaluation.THRESHOLDS):
        prediction = p >= threshold
        assert counts[:, i].tolist() == [int((prediction & mask).sum()), int((prediction & ~mask).sum()),
                                         int((~prediction & mask).sum())]
    logits = torch.zeros(1, 2, 4, 4, requires_grad=True)
    loss = localization_loss(logits, torch.zeros(1, 4, 4, dtype=torch.long))
    loss.backward()
    assert torch.isfinite(loss) and logits.grad[:, 1].sum() > 0


class Budget:
    def __init__(self, stop_at=10000):
        self.calls, self.stop_at = 0, stop_at

    def expired(self):
        self.calls += 1
        return self.calls >= self.stop_at


def test_training_resume_exact_and_continuation(manifest, tmp_path, monkeypatch):
    monkeypatch.setattr(training, "build_model", tiny_model)
    config = {**training.DEFAULTS, "adapt_epochs": 1, "continuation_epochs": 1,
              "num_workers": 0, "accumulation_steps": 2}
    device = torch.device("cpu")
    def run(output, resume=False, budget=None, mode="2d", stage="adapt", parent=None):
        training.train(manifest, config, None, None, output, mode, stage, parent,
                       resume, device, budget or Budget())
    full, split = tmp_path / "full", tmp_path / "split"
    run(full)
    run(split, budget=Budget(stop_at=2))
    paused = training.load_checkpoint(split / "last.pt")
    assert paused["progress"]["cursor"] == 2
    run(split, resume=True)
    first, second = training.load_checkpoint(full / "last.pt"), training.load_checkpoint(split / "last.pt")
    assert second["progress"]["phase"] == "complete"
    for key in first["model"]:
        torch.testing.assert_close(first["model"][key], second["model"][key], atol=0, rtol=0)
    for mode in ("2d", "context"):
        run(tmp_path / mode, mode=mode, stage="continue", parent=full / "best.pt")
    a = json.loads((tmp_path / "2d/epoch-001.json").read_text())
    b = json.loads((tmp_path / "context/epoch-001.json").read_text())
    assert a["sample_order_sha256"] == b["sample_order_sha256"]
    assert a["presentations"] == b["presentations"] == 8
    with pytest.raises(ValueError, match="not empty"):
        run(full)


def test_prediction_without_ground_truth_and_calibration(manifest, tmp_path):
    entry = manifest["entries"][0].copy()
    entry.pop("mask")
    entry.pop("positives")
    model = tiny_model().eval()
    output = tmp_path / "prediction"
    evaluation.export_prediction(model, entry, torch.device("cpu"), output, .5, "test-hash")
    assert np.load(output / "mask.npy").shape == (2, 32, 32)
    assert np.load(output / "probabilities.npy").shape == (2, 32, 32)
    assert json.loads((output / "prediction.json").read_text())["complete"]
    row = dict(real=False, counts=np.tile([[1], [0], [0]], (1, 99)).tolist())
    assert evaluation.select_threshold([row]) == float(evaluation.THRESHOLDS[-1])


@pytest.mark.skipif(not os.getenv("TRUFOR_ROOT") or not os.getenv("TRUFOR_WEIGHTS"),
                    reason="Set TRUFOR_ROOT and TRUFOR_WEIGHTS for the upstream integration test")
def test_official_weights_forward_backward():
    from tesi_m3d.trufor_model import build_model
    baseline = build_model(os.environ["TRUFOR_ROOT"], False, os.environ["TRUFOR_WEIGHTS"]).eval()
    context = build_model(os.environ["TRUFOR_ROOT"], True, os.environ["TRUFOR_WEIGHTS"]).eval()
    images = torch.rand(1, 3, 1, 64, 64)
    valid, distance = torch.ones(1, 3, dtype=torch.bool), torch.tensor([[-2.5, 0., 2.5]])
    with torch.no_grad():
        a = baseline(images[:, 1:2], valid[:, 1:2], distance[:, 1:2])
        b = context(images, valid, distance)
        original = baseline.network(images[:, 1].expand(-1, 3, -1, -1).contiguous())[0]
    torch.testing.assert_close(a, original, atol=0, rtol=0)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    context.train()
    target = torch.zeros(1, 64, 64, dtype=torch.long)
    target[:, 20:40, 20:40] = 1
    loss = localization_loss(context(images, valid, distance), target)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(g.scale.grad.abs() > 0 for g in context.fusion)
