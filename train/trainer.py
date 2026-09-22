"""
The forked trainer (FINETUNE_PLAN §7). Only trainable parameters reach the optimizer and the EMA;
the sampler, optimizer, scheduler, scaler, RNG and position in the epoch are all checkpointed, so a
resumed run replays exactly what an uninterrupted one would have done (C6).
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, gather_object

from train import checkpoint, lora
from train.config import config_hash
from train.ema import TrainableEMA
from train.patches import import_vendor
from train.rng import capture_rng, restore_rng
from train.sampler import ResumableBatchSampler
from train.stops import check_stop
from train.store import Uploader, download_checkpoint, read_latest
from train.validation import FixedNoiseValidator

STATE_KEYS = ("update", "epoch", "batch", "nan_streak", "lr_scale", "recent_losses", "val_history")


class DivergenceError(RuntimeError):
    pass


class ResumeError(RuntimeError):
    pass


def lr_factor(update, warmup, total, final_frac):
    if update < warmup:
        return (update + 1) / warmup
    progress = min(1.0, (update - warmup) / max(1, total - warmup))
    return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * progress))


def step_outcome(loss, grad_norm, scaler_active, step_skipped):
    if not math.isfinite(loss):
        return "nonfinite"
    if scaler_active:
        return "scaler_skip" if step_skipped else "ok"      # fp16 overflow: normal, not a NaN
    return "ok" if math.isfinite(grad_norm) else "nonfinite"


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


class Trainer:
    def __init__(self, cfg, *, model, train_set, val_items, store, work_dir, base_sha, manifest_sha,
                 clock=time.monotonic, on_micro=None):
        self.cfg, self.store, self.work = cfg, store, Path(work_dir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.base_sha, self.manifest_sha, self.cfg_hash = base_sha, manifest_sha, config_hash(cfg)
        self.clock, self.on_micro = clock, on_micro
        self.v = import_vendor()
        self.acc = Accelerator(mixed_precision=cfg["precision"], cpu=not torch.cuda.is_available())
        self.train_set = train_set
        b = cfg["batch"]
        self.accum = b["grad_accum"]
        self.sampler = ResumableBatchSampler([train_set.get_frame_len(i) for i in range(len(train_set))],
                                             b["frames_per_gpu"], b["max_samples"], cfg["seed"],
                                             self.acc.num_processes, self.acc.process_index)
        self.updates_per_epoch = len(self.sampler.epoch_batches(0)) // self.accum
        if self.updates_per_epoch == 0:
            raise ValueError("training set too small for one update per epoch")
        self.total_updates = self.updates_per_epoch * cfg["schedule"]["epochs"]
        self.state = {"update": 0, "epoch": 0, "batch": 0, "nan_streak": 0, "lr_scale": 1.0, "recent_losses": [],
                      "val_history": []}
        self.ema = TrainableEMA(model, cfg["ema"]["decay"], cfg["ema"]["start_after"])
        self.validator = FixedNoiseValidator(val_items, model.vocab_char_map, cfg["validation"]["seed"])
        o = cfg["optim"]
        optimizer = torch.optim.AdamW(lora.param_groups(model, cfg), betas=tuple(o["betas"]), eps=o["eps"], fused=True)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._lr_lambda)
        self.model, self.optimizer = self.acc.prepare(model, optimizer)
        self.trainable = [p for p in self.model.parameters() if p.requires_grad]
        self._last_saved, self._last_val, self._resume_band = -1, -1, None

    # ---- helpers
    def _raw(self):
        return self.acc.unwrap_model(self.model)

    def _lr_lambda(self, step):
        o = self.cfg["optim"]
        return lr_factor(step, o["warmup_updates"], self.total_updates, o["final_lr_frac"]) * self.state["lr_scale"]

    def _apply_lr(self):
        for group, base in zip(self.optimizer.param_groups, self.scheduler.base_lrs):
            group["lr"] = base * self._lr_lambda(self.scheduler.last_epoch)

    def _log(self, entry):
        if self.acc.is_main_process:
            with open(self.work / "log.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")

    def _epoch_batches(self):
        batches = self.sampler.epoch_batches(self.state["epoch"])
        return batches[: len(batches) - len(batches) % self.accum]

    # ---- the loop
    def run(self) -> str:
        self._start = self._last_save_time = self.clock()
        self.uploader = Uploader(self.store)
        self.uploader.start()
        try:
            self._maybe_resume()
            while self.state["epoch"] < self.cfg["schedule"]["epochs"]:
                epoch, batches = self.state["epoch"], self._epoch_batches()
                while self.state["epoch"] == epoch and self.state["batch"] < len(batches):
                    self._update(batches[self.state["batch"]: self.state["batch"] + self.accum])
                    reason = self._after_update()
                    if reason:
                        return reason
                if self.state["epoch"] == epoch:
                    self._end_epoch()
            if self._last_saved != self.state["update"]:
                self._save_last()
            self.uploader.wait()
            return "done"
        finally:
            self.uploader.close()

    def _update(self, group):
        total = 0.0
        for micro, idxs in enumerate(group):
            if self.on_micro:
                self.on_micro(self.state["update"] + 1, micro)
            batch = self.v.collate_fn([self.train_set[i] for i in idxs])
            mel = batch["mel"].permute(0, 2, 1).to(self.acc.device)
            lens = batch["mel_lengths"].to(self.acc.device)
            sync = nullcontext() if micro == len(group) - 1 else self.acc.no_sync(self.model)
            with sync:
                with self.acc.autocast():
                    loss, _, _ = self.model(mel, text=batch["text"], lens=lens)
                self.acc.backward(loss / len(group))
            total += float(loss.detach())
        loss_value = total / len(group)
        grad_norm = float(self.acc.clip_grad_norm_(self.trainable, self.cfg["optim"]["grad_clip"]))
        scaler_active = self.acc.scaler is not None
        if scaler_active or (math.isfinite(loss_value) and math.isfinite(grad_norm)):
            self.optimizer.step()
        outcome = step_outcome(loss_value, grad_norm, scaler_active,
                               scaler_active and self.acc.optimizer_step_was_skipped)
        self.optimizer.zero_grad(set_to_none=True)
        self.state["batch"] += len(group)
        if outcome == "nonfinite":
            self._log({"event": "nonfinite", "update": self.state["update"], "loss": loss_value})
            self.state["nan_streak"] += 1
            if self.state["nan_streak"] >= 3:
                self._rollback()
            return
        if outcome == "ok":
            self.state["update"] += 1
            self.state["nan_streak"] = 0
            self.scheduler.step()
            self.ema.update(self._raw(), self.state["update"])
            self.state["recent_losses"] = (self.state["recent_losses"] + [loss_value])[-20:]
            if self._resume_band is not None:
                low, high = self._resume_band
                if not low <= loss_value <= high:
                    raise ResumeError(f"first loss after resume {loss_value:.4f} outside [{low:.4f}, {high:.4f}]")
                self._resume_band = None
        self._log({"update": self.state["update"], "epoch": self.state["epoch"], "loss": loss_value,
                   "lr": self.scheduler.get_last_lr()[0], "grad_norm": grad_norm, "outcome": outcome})

    def _after_update(self):
        u, ck = self.state["update"], self.cfg["checkpoint"]
        if u and u % self.cfg["validation"]["every_updates"] == 0 and u != self._last_val:
            stop = self._validate()
            if stop:
                if u != self._last_saved:
                    self._save_last()
                self.uploader.wait()
                self._log({"event": "validation_stop", "update": u, "reason": stop})
                return "validation_stop"
        due = u % ck["every_updates"] == 0 or (self.clock() - self._last_save_time) / 60 >= ck["every_minutes"]
        if u and due and u != self._last_saved:
            self._save_last()
        if self.clock() - self._start >= self.cfg["runtime"]["guard_seconds"]:
            if u != self._last_saved:
                self._save_last()
            self.uploader.wait()
            self._log({"event": "time_guard", "update": u})
            return "time_guard"
        return None

    def _validate(self):
        self._last_val = self.state["update"]
        stop = None
        if self.acc.is_main_process:
            raw = self._raw()
            result = {"raw": self.validator.evaluate(raw)}
            with self.ema.swapped(raw):
                result["ema"] = self.validator.evaluate(raw)
            self._log({"event": "validation", "update": self.state["update"], **result})
            self.state["val_history"] = self.state["val_history"] + [result["ema"]]
            stop = check_stop(self.state["val_history"], self.cfg["validation"]["stop"])
        if self.acc.num_processes > 1:
            stop = broadcast_object_list([stop])[0]
        return stop

    # ---- checkpoints
    def _state_for_save(self):
        rng = capture_rng()
        ranks = gather_object([rng]) if self.acc.num_processes > 1 else [rng]
        return dict(self.state, rng=ranks, micro=0, config_hash=self.cfg_hash, manifest_sha256=self.manifest_sha,
                    base_sha256=self.base_sha, git_commit=_git_commit(), saved_at=time.time())

    def _save_last(self):
        self.uploader.wait()
        state = self._state_for_save()
        if self.acc.is_main_process:
            optim = {"optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                     "scaler": self.acc.scaler.state_dict() if self.acc.scaler else None}
            hashes = checkpoint.save_checkpoint(self.work / "last", trainable=lora.trainable_state_dict(self._raw()),
                                                ema=self.ema.shadow_state(), optim_state=optim, state=state)
            prefix = f"ckpt/update_{self.state['update']:07d}"
            self.uploader.submit(self.work / "last", prefix,
                                 {"prefix": prefix, "update": self.state["update"], "hashes": hashes})
        self.acc.wait_for_everyone()
        self._last_saved, self._last_save_time = self.state["update"], self.clock()

    def _end_epoch(self):
        k = self.state["epoch"]
        self.state["epoch"], self.state["batch"] = k + 1, 0
        state = self._state_for_save()
        if self.acc.is_main_process:
            self.uploader.wait()
            d = self.work / f"epoch_{k}"
            checkpoint.save_checkpoint(d, trainable=lora.trainable_state_dict(self._raw()),
                                       ema=self.ema.shadow_state(), optim_state=None, state=state)
            self.uploader.submit(d, f"epoch_{k}", None)
            for old in self.work.glob("epoch_*"):
                if old.is_dir() and old != d and not old.name.endswith((".tmp", ".old")):
                    shutil.rmtree(old)
        self.acc.wait_for_everyone()

    def _load(self, path):
        ck = checkpoint.load_checkpoint(path, config_hash=self.cfg_hash, manifest_sha=self.manifest_sha,
                                        base_sha=self.base_sha)
        lora.load_trainable(self._raw(), ck["trainable"])
        self.ema.load_shadow(ck["ema"])
        o = ck["optim_state"]
        self.optimizer.load_state_dict(o["optimizer"])
        self.scheduler.load_state_dict(o["scheduler"])
        if o["scaler"] is not None and self.acc.scaler is not None:
            self.acc.scaler.load_state_dict(o["scaler"])
        self.state = {k: ck["state"][k] for k in STATE_KEYS}
        restore_rng(ck["state"]["rng"][self.acc.process_index])

    def _maybe_resume(self):
        latest = read_latest(self.store)
        if latest is None:
            self._log({"event": "fresh_start"})
            return
        local = download_checkpoint(self.store, latest, self.work / "resume")
        self._load(local)
        shutil.rmtree(self.work / "last", ignore_errors=True)
        shutil.copytree(local, self.work / "last")                  # rollback target in a fresh container
        self._last_saved = self._last_val = self.state["update"]
        recent = self.state["recent_losses"]
        if recent:
            mean = sum(recent) / len(recent)
            self._resume_band = (0.5 * mean, 2.0 * mean)
        s = self.state
        print(f"resumed at update {s['update']}, epoch {s['epoch']}, batch {s['batch']}")
        self._log({"event": "resumed", "update": s["update"], "epoch": s["epoch"], "batch": s["batch"]})

    def _rollback(self):
        if self.state["lr_scale"] < 1.0:
            raise DivergenceError("non-finite updates again after one rollback; stop and investigate (plan §10b)")
        if not (self.work / "last").exists():
            raise DivergenceError("three non-finite updates in a row and no checkpoint to roll back to")
        self.uploader.wait()
        self._load(self.work / "last")
        self.state["lr_scale"], self.state["nan_streak"] = 0.5, 0
        self._apply_lr()
        self._log({"event": "rollback", "update": self.state["update"], "lr_scale": 0.5})
