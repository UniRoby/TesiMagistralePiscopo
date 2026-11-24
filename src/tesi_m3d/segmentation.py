"""MedicalNet spatial encoder, supervised decoder and bounded-memory inference."""
from __future__ import annotations

import numpy as np
import hashlib
import json
import torch
from torch import nn
from torch.nn import functional as F

from .model import MEDICALNET_WEIGHTS, Patch3DModelConfig, build_medicalnet_encoder, build_patch3d_classifier

SEGMENTER_NAME = "resnet18_medicalnet_seg"


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_signature(config):
    """Identify the controlled training comparison, independent of clipping/I/O."""
    training = dict(config["training"])
    for key in ("batch_size", "accumulation_steps", "num_workers", "volume_cache_size"):
        training.pop(key, None)
    preprocessing = {"low": 1.0, "high": 99.0, **config.get("preprocessing", {})}
    preprocessing.pop("clip", None)
    protocol = {"model": config["model"], "patches": config["patches"], "training": training,
                "preprocessing": preprocessing}
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def read_calibration(path, checkpoint):
    with open(path) as handle:
        calibration = json.load(handle)
    if (calibration.get("source_split") != "valid" or
            calibration.get("checkpoint_sha256") != file_digest(checkpoint)):
        raise ValueError("calibration must come from validation of this exact checkpoint")
    for key in ("localization_threshold", "detection_threshold"):
        if not np.isfinite(calibration[key]) or not 0 <= calibration[key] <= 1:
            raise ValueError("invalid calibrated threshold")
    return calibration


def _block(in_channels, channels, padding_mode="zeros"):
    return nn.Sequential(
        nn.Conv3d(in_channels, channels, 3, padding=1, padding_mode=padding_mode, bias=False), nn.GroupNorm(8, channels), nn.ReLU(),
        nn.Conv3d(channels, channels, 3, padding=1, padding_mode=padding_mode, bias=False), nn.GroupNorm(8, channels), nn.ReLU(),
    )


class MedicalNetSegmenter(nn.Module):
    def __init__(self, encoder, decoder_padding_mode="zeros"):
        super().__init__()
        self.encoder = encoder
        self.bridge = nn.Sequential(nn.Conv3d(896, 128, 1, bias=False), nn.GroupNorm(8, 128), nn.ReLU())
        self.decode4 = _block(128 + 64, 64, decoder_padding_mode)
        self.decode2 = _block(64 + 64, 32, decoder_padding_mode)
        self.decode1 = _block(32 + 1, 16, decoder_padding_mode)
        self.output = nn.Conv3d(16, 1, 1)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        # Volume-grouped crops must not update pretrained BatchNorm statistics.
        for module in self.encoder.modules():
            if isinstance(module, nn.BatchNorm3d):
                module.eval()
        return self

    def freeze_encoder(self, frozen):
        self.encoder.requires_grad_(not frozen)

    def forward(self, image):
        stem = self.encoder.act(self.encoder.bn1(self.encoder.conv1(image)))
        s4 = self.encoder.layer1(self.encoder.maxpool(stem))
        s8 = self.encoder.layer2(s4)
        s8b = self.encoder.layer3(s8)
        s8c = self.encoder.layer4(s8b)
        decoded = self.bridge(torch.cat((s8, s8b, s8c), dim=1))
        for skip, block in ((s4, self.decode4), (stem, self.decode2), (image, self.decode1)):
            decoded = F.interpolate(decoded, size=skip.shape[2:], mode="trilinear", align_corners=False)
            decoded = block(torch.cat((decoded, skip), dim=1))
        return self.output(decoded)


def build_segmenter(config=None, *, state_dict=None):
    config = config or {"name": SEGMENTER_NAME}
    if config.get("name") != SEGMENTER_NAME or config.get("in_channels", 1) != 1 or config.get("num_classes", 1) != 1:
        raise ValueError("segmentation requires resnet18_medicalnet_seg with one input/output channel")
    if config.get("decoder_padding_mode", "zeros") not in ("zeros", "replicate"):
        raise ValueError("decoder_padding_mode must be zeros or replicate")
    encoder = build_medicalnet_encoder(config.get("medicalnet_weights", MEDICALNET_WEIGHTS), initialize=state_dict is None)
    model = MedicalNetSegmenter(encoder, config.get("decoder_padding_mode", "zeros"))
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model


def segmentation_loss_components(logits, mask, *, foreground_normalizer=None, squared_pred=False):
    if logits.shape != mask.shape or logits.ndim != 5 or logits.shape[1] != 1:
        raise ValueError("segmentation logits and targets must have matching Bx1xDxHxW shapes")
    if not isinstance(squared_pred, bool):
        raise ValueError("squared_pred must be boolean")
    logits, mask = logits.float(), mask.float()
    bce = F.binary_cross_entropy_with_logits(logits, mask)
    probability = logits.sigmoid().flatten(1)
    target = mask.flatten(1)
    positive = target.sum(1) > 0
    prediction_sum = probability.square().sum(1) if squared_pred else probability.sum(1)
    target_sum = target.square().sum(1) if squared_pred else target.sum(1)
    dice = (2 * (probability * target).sum(1) + 1e-5) / (prediction_sum + target_sum + 1e-5)
    # Empty crops are supervised by BCE, never rewarded as perfect Dice examples.
    denominator = positive.sum().clamp_min(1) if foreground_normalizer is None else max(float(foreground_normalizer), 1e-8)
    dice_loss = ((1 - dice) * positive).sum() / denominator
    return bce, dice_loss


def segmentation_loss(logits, mask, *, foreground_normalizer=None, squared_pred=False):
    bce, dice = segmentation_loss_components(logits, mask, foreground_normalizer=foreground_normalizer,
                                            squared_pred=squared_pred)
    return 0.5 * bce + 0.5 * dice


def preprocess_scan(raw, dirname, cache_dir=None, preprocessing=None):
    from .dataset import normalize_percentile
    from .volume_io import cached_percentile_bounds
    settings = {"low": 1.0, "high": 99.0, "clip": False, **(preprocessing or {})}
    if (set(settings) != {"low", "high", "clip"} or not isinstance(settings["clip"], bool)
            or not 0 <= settings["low"] < settings["high"] <= 100):
        raise ValueError("invalid segmentation percentile preprocessing")
    bounds = cached_percentile_bounds(raw, dirname, cache_dir, low=settings["low"], high=settings["high"])
    image = normalize_percentile(raw, bounds=bounds)
    lo, hi = np.float32(bounds[0]), np.float32(bounds[1])
    background = float((np.float32(0) - lo) / (hi - lo)) if hi > lo else 0.0
    if settings["clip"]:
        np.clip(image, 0, 1, out=image)
        background = float(np.clip(background, 0, 1))
    return image, background


def predict_segmentation(model, volume, *, patch_shape=(64, 64, 64), batch_size=2,
                         device="cpu", amp=False, background=0.0):
    from monai.inferers import sliding_window_inference
    if batch_size < 1 or len(patch_shape) != 3 or min(patch_shape) < 8:
        raise ValueError("invalid sliding-window batch or patch shape")
    image = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32))[None, None]
    model.eval()
    model.to(device)

    def probability(batch):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and str(device).startswith("cuda")):
            return model(batch).float().sigmoid()

    with torch.inference_mode():
        prediction = sliding_window_inference(
            image, roi_size=tuple(patch_shape), sw_batch_size=batch_size, predictor=probability,
            overlap=0.5, mode="gaussian", sigma_scale=0.125, cval=background,
            sw_device=device, device="cpu",
        )
    result = prediction[0, 0].numpy().astype(np.float32, copy=False)
    if result.shape != volume.shape or not np.isfinite(result).all():
        raise ValueError("invalid segmentation probability map")
    return result


def load_predictor(checkpoint, device="cpu"):
    from .train import _load_torch_payload
    payload = _load_torch_payload(checkpoint)
    config = payload["config"]
    if config["model"]["name"] == SEGMENTER_NAME:
        if payload.get("task") != "segmentation" or not {"preprocessing", "patches"} <= config.keys():
            raise ValueError("segmentation checkpoint lacks task/preprocessing/geometry")
        model = build_segmenter(config["model"], state_dict=payload["model_state_dict"])
    else:
        model = build_patch3d_classifier(Patch3DModelConfig(**config["model"]), state_dict=payload["model_state_dict"])
    return model.to(device).eval(), payload


def predict_checkpoint(model, payload, raw, dirname, cache_dir=None, *, device="cpu", batch_size=2):
    """Common baseline/segmenter path; always honor checkpoint preprocessing."""
    config = payload["config"]
    shape = tuple(config.get("patches", {}).get("patch_shape", (32, 32, 32)))
    is_seg = config["model"]["name"] == SEGMENTER_NAME
    image, background = preprocess_scan(raw, dirname, cache_dir, config.get("preprocessing") if is_seg else None)
    if is_seg:
        heatmap = predict_segmentation(model, image, patch_shape=shape, batch_size=batch_size,
                                      device=device, amp=True, background=background)
    else:
        from .inference import infer_heatmap
        heatmap = infer_heatmap(model, image, shape,
                                tuple(config["patches"].get("inference_stride", (16, 16, 16))),
                                batch_size=batch_size, device=device,
                                aggregation=config.get("evaluation", {}).get("heatmap_aggregation", "average"))
    return heatmap, image
