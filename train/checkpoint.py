"""
Checkpoint directories (FINETUNE_PLAN §7c/§7d): trainable weights, EMA, optimizer state and a
state.json, with SHA256SUMS. Written to <name>.tmp, fsynced, then swapped in, so a crash mid-save
leaves the previous checkpoint readable. The frozen base is never written.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from train.model import file_sha256

SUMS = "SHA256SUMS"


class CheckpointError(RuntimeError):
    pass


class CheckpointMismatch(CheckpointError):
    pass


def _fsync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_checkpoint(path, *, trainable, ema, optim_state, state) -> dict[str, str]:
    path = Path(path)
    tmp, old = path.with_name(path.name + ".tmp"), path.with_name(path.name + ".old")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    save_file({k: v.contiguous() for k, v in trainable.items()}, str(tmp / "trainable.safetensors"))
    save_file({k: v.contiguous() for k, v in ema.items()}, str(tmp / "ema.safetensors"))
    if optim_state is not None:
        torch.save(optim_state, tmp / "optim.pt")
    (tmp / "state.json").write_text(json.dumps(state, sort_keys=True))
    hashes = {p.name: file_sha256(p) for p in sorted(tmp.iterdir())}
    (tmp / SUMS).write_text("".join(f"{h}  {n}\n" for n, h in hashes.items()))
    for p in tmp.iterdir():
        _fsync(p)
    _fsync(tmp)
    shutil.rmtree(old, ignore_errors=True)
    if path.exists():
        os.replace(path, old)
    os.replace(tmp, path)
    shutil.rmtree(old, ignore_errors=True)
    return hashes


def verify(path) -> dict[str, str]:
    path = Path(path)
    if not (path / SUMS).exists():
        raise CheckpointError(f"{path}: no {SUMS}")
    expected = dict(reversed(line.split("  ", 1)) for line in (path / SUMS).read_text().splitlines())
    for name, h in expected.items():
        if not (path / name).exists() or file_sha256(path / name) != h:
            raise CheckpointError(f"{path}: {name} does not match {SUMS}")
    return expected


def load_checkpoint(path, *, config_hash, manifest_sha, base_sha) -> dict:
    path = Path(path)
    verify(path)
    state = json.loads((path / "state.json").read_text())
    for field, now in (("config_hash", config_hash), ("manifest_sha256", manifest_sha), ("base_sha256", base_sha)):
        if state.get(field) != now:
            raise CheckpointMismatch(f"refusing to resume: {field} differs "
                                     f"(checkpoint {str(state.get(field))[:12]}, now {now[:12]})")
    optim = path / "optim.pt"
    return {"trainable": load_file(str(path / "trainable.safetensors")),
            "ema": load_file(str(path / "ema.safetensors")),
            "optim_state": torch.load(optim, weights_only=False) if optim.exists() else None,
            "state": state}
