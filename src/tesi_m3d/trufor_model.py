"""Official TruFor localization network with optional residual slice context."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import hashlib

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

UPSTREAM_REVISION = "ae54475df6f41a491d7615100feb19263dec13f7"
OFFICIAL_WEIGHTS_SHA256 = "ac1d90e329a72e0d66e8665e123a19e94bfae3209c3ef8a4f9ca3b91578c7844"


def official_network(root):
    from yacs.config import CfgNode
    root = Path(root).resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"Expected TruFor revision {UPSTREAM_REVISION}, found {revision}")
    subprocess.run(["git", "-C", str(root), "diff", "--exit-code", "HEAD", "--",
                    "TruFor_train_test/lib"], check=True, capture_output=True)
    source = root / "TruFor_train_test"
    if "lib" in sys.modules:
        existing = Path(sys.modules["lib"].__file__).resolve()
        if not existing.is_relative_to(source):
            raise RuntimeError(f"Another package named lib is already imported: {existing}")
    sys.path.insert(0, str(source)) if str(source) not in sys.path else None
    from lib.models.cmx.builder_np_conf import EncoderDecoder
    cfg = CfgNode.load_cfg((source / "lib/config/trufor_ph3.yaml").read_text())
    cfg.MODEL.PRETRAINED = None
    cfg.MODEL.MODS = ["RGB", "NP++"]
    cfg.MODEL.EXTRA.NP_WEIGHTS = None
    cfg.MODEL.EXTRA.NP_OUT_CHANNELS = 1
    cfg.MODEL.EXTRA.MODULES = ["NP++", "backbone", "loc_head"]
    cfg.MODEL.EXTRA.FIX_MODULES = ["NP++"]
    cfg.MODEL.EXTRA.DETECTION = None
    return EncoderDecoder(cfg)


def load_official_weights(network, path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    if hasher.hexdigest() != OFFICIAL_WEIGHTS_SHA256:
        raise ValueError("Official checkpoint checksum mismatch; use the documented TruFor weights")
    # The verified upstream archive contains NumPy scalars in its training metadata.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    wanted = network.state_dict()
    extra = set(state) - set(wanted)
    forbidden = [k for k in extra if not k.startswith(("decode_head_conf.", "detection."))]
    missing = set(wanted) - set(state)
    if forbidden or missing:
        raise ValueError(f"Incompatible TruFor weights: missing={sorted(missing)}, extra={forbidden}")
    network.load_state_dict({k: state[k] for k in wanted}, strict=True)


class SliceFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Conv2d(3 * channels + 1, 1, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, central, neighbors, valid, distances):
        correction = torch.zeros_like(central)
        for i, neighbor in enumerate(neighbors):
            distance = distances[:, i, None, None, None].expand(-1, 1, *central.shape[-2:])
            gate = torch.sigmoid(self.gate(torch.cat(
                (central, neighbor, (neighbor - central).abs(), distance.to(central.dtype)), dim=1)))
            correction = correction + valid[:, i, None, None, None] * gate * (neighbor - central)
        count = valid.sum(dim=1).clamp_min(1)[:, None, None, None]
        return central + self.scale * correction / count


class TruForCT(nn.Module):
    def __init__(self, network, context=False, checkpoint_encoder=True):
        super().__init__()
        self.network = network
        self.context = context
        self.checkpoint_encoder = checkpoint_encoder
        self.fusion = nn.ModuleList([SliceFusion(c) for c in network.channels]) if context else nn.ModuleList()
        self.network.dncnn.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.network.dncnn.eval()
        # Microbatch one cannot estimate stable population statistics.
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def features(self, image):
        rgb = image.expand(-1, 3, -1, -1).contiguous()
        with torch.no_grad():
            self.network.dncnn.eval()
            noise = self.network.dncnn(rgb).expand(-1, 3, -1, -1).contiguous()
        rgb = self.network.prepro(rgb)
        if self.training and self.checkpoint_encoder:
            return checkpoint(self.network.backbone, rgb, noise, use_reentrant=False)
        return self.network.backbone(rgb, noise)

    def forward(self, images, valid, distances):
        expected = 3 if self.context else 1
        if images.ndim != 5 or images.shape[1:3] != (expected, 1):
            raise ValueError(f"Expected B,{expected},1,H,W input")
        h, w = images.shape[-2:]
        central_index = 1 if self.context else 0
        def padded(i):
            return F.pad(images[:, i], (0, (-w) % 32, 0, (-h) % 32), mode="replicate")
        center = self.features(padded(central_index))
        if self.context:
            left, right = self.features(padded(0)), self.features(padded(2))
            center = [gate(c, (l, r), valid[:, [0, 2]], distances[:, [0, 2]])
                      for gate, c, l, r in zip(self.fusion, center, left, right)]
        logits = self.network.decode_head(center)
        logits = F.interpolate(logits, size=(h + (-h) % 32, w + (-w) % 32),
                               mode="bilinear", align_corners=False)
        return logits[..., :h, :w]


def build_model(upstream, context, official_weights=None):
    network = official_network(upstream)
    if official_weights:
        load_official_weights(network, official_weights)
    return TruForCT(network, context=context)


def localization_loss(logits, target):
    logp = logits.float().log_softmax(dim=1)
    selected = logp.gather(1, target[:, None]).squeeze(1)
    alpha = torch.where(target.bool(), 0.75, 0.25)
    focal = (-alpha * (1 - selected.exp()).square() * selected).mean()
    probability = logp[:, 1].exp()
    truth = target.float()
    intersection = (probability * truth).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + truth.sum(dim=(-2, -1))
    per_image = 1 - (2 * intersection + 1e-6) / (denominator + 1e-6)
    # Empty masks are supervised by focal loss; empty-target Dice is nearly constant.
    dice = (per_image * (truth.sum(dim=(-2, -1)) > 0)).mean()
    return focal + dice
