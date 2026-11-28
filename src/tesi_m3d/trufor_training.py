"""Resumable, equal-sampling training for the two TruFor variants."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import signal
import tempfile
import time
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader

from .trufor_data import (SliceDataset, atomic_json, digest, epoch_samples,
                          validation_panel)
from .trufor_evaluation import autocast, evaluate_volumes, file_hash, summarize
from .trufor_model import build_model, localization_loss, UPSTREAM_REVISION

DEFAULTS = dict(seed=42, adapt_epochs=20, continuation_epochs=10, accumulation_steps=8,
                num_workers=4, prefetch_factor=1, persistent_workers=True, pin_memory=True,
                encoder_lr=1e-5, head_lr=1e-4, weight_decay=0.01,
                warmup_fraction=0.05, checkpoint_steps=100)


class SessionBudget:
    def __init__(self, hours):
        if hours <= 0:
            raise ValueError("session-hours must be positive")
        self.deadline = time.monotonic() + hours * 3600
        self.stop = False
        self.previous = signal.getsignal(signal.SIGINT)

    def __enter__(self):
        signal.signal(signal.SIGINT, self.request_stop)
        return self

    def request_stop(self, *_):
        self.stop = True
        print("Stopping at the next safe boundary; please wait for the checkpoint.", flush=True)

    def expired(self):
        return self.stop or time.monotonic() >= self.deadline

    def __exit__(self, *_):
        signal.signal(signal.SIGINT, self.previous)


def atomic_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=[state[0], state[1].tolist(), *state[2:]],
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def make_loader(dataset, sampler, config, device):
    workers = config["num_workers"]
    kwargs = dict(batch_size=1, sampler=sampler, num_workers=workers,
                  pin_memory=device.type == "cuda" and config["pin_memory"],
                  generator=torch.Generator().manual_seed(config["seed"]))
    if workers:
        kwargs.update(persistent_workers=config["persistent_workers"],
                      prefetch_factor=config["prefetch_factor"], timeout=120)
    return DataLoader(dataset, **kwargs)


def optimizer_for(model, config):
    encoder, heads = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (encoder if name.startswith("network.backbone.") else heads).append(parameter)
    return torch.optim.AdamW([dict(params=encoder, lr=config["encoder_lr"]),
                              dict(params=heads, lr=config["head_lr"])],
                             weight_decay=config["weight_decay"])


def validate_config(config):
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    result = {**DEFAULTS, **config}
    for key in ("adapt_epochs", "continuation_epochs", "accumulation_steps", "prefetch_factor", "checkpoint_steps"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(result["num_workers"]) is not int or result["num_workers"] < 0:
        raise ValueError("num_workers must be nonnegative")
    if not 0 < result["warmup_fraction"] < 1:
        raise ValueError("warmup_fraction must be in (0,1)")
    for key in ("encoder_lr", "head_lr", "weight_decay"):
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"Invalid {key}")
    return result


def train(manifest, config, upstream, official_weights, output, mode, stage,
          init_from, resume, device, budget):
    config = validate_config(config)
    output = Path(output)
    last = output / "last.pt"
    if stage == "adapt" and (mode != "2d" or init_from):
        raise ValueError("Adaptation must be 2d without init-from")
    if stage == "continue" and not init_from:
        raise ValueError("Continuation requires the common 2d best.pt via init-from")
    if resume and not last.exists():
        raise FileNotFoundError(f"No checkpoint to resume: {last}")
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError(f"Run directory is not empty; use --resume or a new directory: {output}")
    parent_hash = file_hash(init_from) if init_from else None
    contract = dict(config=config, manifest=manifest["fingerprint"], mode=mode, stage=stage,
                    parent_sha256=parent_hash, upstream_revision=UPSTREAM_REVISION)
    saved = load_checkpoint(last) if resume else None
    if saved and saved["contract"] != contract:
        raise ValueError("Resume configuration, dataset, parent or mode differs from the saved run")
    if saved and saved["progress"]["phase"] == "complete":
        print("Run already complete.")
        return
    seed_everything(config["seed"])
    model = build_model(upstream, mode == "context", official_weights if not (resume or init_from) else None)
    if init_from and not resume:
        parent = load_checkpoint(init_from)
        status_path = Path(init_from).parent / "run.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if (parent["contract"]["mode"] != "2d" or parent["contract"]["stage"] != "adapt"
                or status.get("phase") != "complete" or status.get("run_id") != parent["run_id"]):
            raise ValueError("Parent must be best.pt from a completed 2d adaptation")
        if parent["contract"]["manifest"] != manifest["fingerprint"]:
            raise ValueError("Parent used a different dataset manifest")
        incompat = model.load_state_dict(parent["model"], strict=False)
        expected = {k for k in model.state_dict() if k.startswith("fusion.")}
        if set(incompat.missing_keys) != expected or incompat.unexpected_keys:
            raise ValueError(f"Invalid 2d initialization: {incompat}")
        del parent
    if saved:
        model.load_state_dict(saved["model"], strict=True)
    model.to(device)
    entries = [e for e in manifest["entries"] if e["split"] == "train"]
    if not entries:
        raise ValueError("No training volumes")
    panel = validation_panel(manifest["entries"], config["seed"])
    epochs = config["adapt_epochs"] if stage == "adapt" else config["continuation_epochs"]
    updates_per_epoch = math.ceil(4 * len(entries) / config["accumulation_steps"])
    total_updates = epochs * updates_per_epoch
    optimizer = optimizer_for(model, config)
    warmup = max(1, round(total_updates * config["warmup_fraction"]))
    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_updates - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not torch.cuda.is_bf16_supported())
    progress = dict(epoch=0, cursor=0, step=0, phase="train", best=-1.0, loss_sum=0.0,
                    elapsed_seconds=0.0)
    run_id = str(uuid.uuid4())
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        progress, run_id = saved["progress"], saved["run_id"]
        restore_rng(saved["rng"])
    del saved
    # Reset after model construction: the extra fusion initialization must not
    # change the continuation's starting RNG state for the shared central encoder.
    if not resume:
        seed_everything(config["seed"])
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "protocol.json", dict(**contract, run_id=run_id,
                train_volumes=len(entries), presentations_per_epoch=4 * len(entries),
                total_updates=total_updates, validation_panel=[e["id"] for e in panel]))
    sampler = []
    loader = make_loader(SliceDataset(entries, mode == "context"), sampler, config, device)
    clock = time.monotonic()
    def save(best=False):
        nonlocal clock
        now = time.monotonic()
        progress["elapsed_seconds"] += now - clock
        clock = now
        payload = dict(model=model.state_dict(), contract=contract, run_id=run_id,
                       progress=progress.copy())
        if not best:
            payload.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                           scaler=scaler.state_dict(), rng=rng_state())
        atomic_checkpoint(output / ("best.pt" if best else "last.pt"), payload)
        atomic_json(output / "run.json", dict(run_id=run_id, **progress))
    while progress["epoch"] < epochs:
        if budget.expired():
            save()
            print("Session saved. Repeat with --resume.", flush=True)
            return
        tokens = epoch_samples(entries, progress["epoch"], config["seed"])
        if progress["phase"] == "train":
            sampler[:] = tokens[progress["cursor"]:]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            group_size = min(config["accumulation_steps"], len(tokens) - progress["cursor"])
            in_group = 0
            for batch in loader:
                images, target, valid, distances = [x.to(device, non_blocking=True) for x in batch]
                with autocast(device):
                    loss = localization_loss(model(images, valid, distances), target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss; resume from the last safe checkpoint")
                scaler.scale(loss / group_size).backward()
                progress["loss_sum"] += float(loss.detach())
                progress["cursor"] += 1
                in_group += 1
                if in_group < group_size:
                    continue
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                progress["step"] += 1
                in_group = 0
                group_size = min(config["accumulation_steps"], len(tokens) - progress["cursor"])
                if progress["step"] % 10 == 0:
                    print(f"{mode} epoch={progress['epoch'] + 1}/{epochs} "
                          f"samples={progress['cursor']}/{len(tokens)} updates={progress['step']}/{total_updates} "
                          f"loss={progress['loss_sum'] / progress['cursor']:.5f}", flush=True)
                if progress["step"] % config["checkpoint_steps"] == 0 or time.monotonic() - clock >= 600:
                    save()
                if budget.expired():
                    save()
                    print("Session saved. Repeat with --resume.", flush=True)
                    return
            progress["phase"] = "validate"
            save()
        validation_dir = output / "validation" / f"epoch-{progress['epoch'] + 1:03d}"
        rows = evaluate_volumes(model, panel, device, validation_dir,
                                [run_id, progress["epoch"], manifest["fingerprint"]], budget.expired)
        if rows is None:
            save()
            print("Validation paused; completed volumes are cached.", flush=True)
            return
        report = summarize(rows, 0.5)
        score = report["groups"]["all_fake"]["macro_f1"]
        report.update(epoch=progress["epoch"] + 1, mean_train_loss=progress["loss_sum"] / len(tokens),
                      sample_order_sha256=digest(tokens), presentations=len(tokens),
                      distinct_slices=len({(t[0], t[1]) for t in tokens}),
                      distinct_volumes=len({t[0] for t in tokens}))
        atomic_json(output / f"epoch-{progress['epoch'] + 1:03d}.json", report)
        if score > progress["best"]:
            progress["best"] = score
            save(best=True)
        print(f"Validation macro F1 @0.5: {score:.5f}; best={progress['best']:.5f}", flush=True)
        progress.update(epoch=progress["epoch"] + 1, cursor=0, phase="train", loss_sum=0.0)
        save()
    progress["phase"] = "complete"
    save()
    print(f"Completed: {output / 'best.pt'}", flush=True)
