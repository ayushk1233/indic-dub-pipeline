import json
import random

import numpy as np
import pytest
import torch

from train import checkpoint
from train.checkpoint import CheckpointError, CheckpointMismatch
from train.rng import capture_rng, restore_rng

STATE = {"update": 30, "config_hash": "c" * 64, "manifest_sha256": "m" * 64, "base_sha256": "b" * 64}


def _save(path, value=1.0, optim=True):
    return checkpoint.save_checkpoint(
        path, trainable={"w": torch.full((3,), value)}, ema={"w": torch.full((3,), value / 2)},
        optim_state={"optimizer": {"lr": 0.1}} if optim else None, state=dict(STATE, rng=capture_rng()))


def _load(path, **over):
    kw = dict(config_hash="c" * 64, manifest_sha="m" * 64, base_sha="b" * 64)
    kw.update(over)
    return checkpoint.load_checkpoint(path, **kw)


def test_rng_capture_restore_reproduces_all_three_generators():
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    saved = json.loads(json.dumps(capture_rng()))
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    restore_rng(saved)
    assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == first


def test_round_trip_and_hashes(tmp_path):
    hashes = _save(tmp_path / "last")
    assert set(hashes) == {"trainable.safetensors", "ema.safetensors", "optim.pt", "state.json"}
    assert checkpoint.verify(tmp_path / "last") == hashes
    ck = _load(tmp_path / "last")
    assert torch.equal(ck["trainable"]["w"], torch.ones(3)) and ck["state"]["update"] == 30
    assert ck["optim_state"] == {"optimizer": {"lr": 0.1}}


def test_epoch_checkpoint_has_no_optimizer(tmp_path):
    assert "optim.pt" not in _save(tmp_path / "epoch_0", optim=False)
    assert _load(tmp_path / "epoch_0")["optim_state"] is None


def test_tampered_file_fails_verification(tmp_path):
    _save(tmp_path / "last")
    p = tmp_path / "last" / "ema.safetensors"
    data = bytearray(p.read_bytes()); data[-1] ^= 1; p.write_bytes(bytes(data))
    with pytest.raises(CheckpointError, match="ema.safetensors"):
        checkpoint.verify(tmp_path / "last")


def test_crash_mid_save_leaves_previous_checkpoint_intact(tmp_path, monkeypatch):
    _save(tmp_path / "last", value=1.0)
    real = checkpoint.save_file
    calls = {"n": 0}

    def dying(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("killed")
        return real(*a, **k)

    monkeypatch.setattr(checkpoint, "save_file", dying)
    with pytest.raises(OSError):
        _save(tmp_path / "last", value=2.0)
    assert torch.equal(_load(tmp_path / "last")["trainable"]["w"], torch.ones(3))


@pytest.mark.c12
@pytest.mark.parametrize("field,kw", [("config_hash", "config_hash"), ("manifest_sha256", "manifest_sha"),
                                      ("base_sha256", "base_sha")])
def test_resume_refuses_on_changed_config_manifest_or_base(tmp_path, field, kw):
    _save(tmp_path / "last")
    with pytest.raises(CheckpointMismatch, match=field):
        _load(tmp_path / "last", **{kw: "x" * 64})


@pytest.mark.skipif(not (torch.cuda.is_available() or torch.backends.mps.is_available()), reason="needs a second device")
def test_optimizer_state_loads_onto_cpu(tmp_path):
    # Review #7: rank 1 must not unpickle rank 0's cuda:0 tensors onto cuda:0.
    device = "cuda" if torch.cuda.is_available() else "mps"
    checkpoint.save_checkpoint(tmp_path / "last", trainable={"w": torch.ones(1)}, ema={"w": torch.ones(1)},
                               optim_state={"t": torch.ones(2, device=device)}, state=dict(STATE, world_size=1))
    ck = checkpoint.load_checkpoint(tmp_path / "last", config_hash="c" * 64, manifest_sha="m" * 64,
                                    base_sha="b" * 64, world_size=1)
    assert ck["optim_state"]["t"].device.type == "cpu"


@pytest.mark.c12
def test_resume_refuses_a_changed_world_size(tmp_path):
    # Review #8: a 2-GPU checkpoint resumed on 1 GPU would shard batches differently.
    checkpoint.save_checkpoint(tmp_path / "last", trainable={"w": torch.ones(1)}, ema={"w": torch.ones(1)},
                               optim_state=None, state=dict(STATE, world_size=2))
    with pytest.raises(CheckpointMismatch, match="world_size"):
        checkpoint.load_checkpoint(tmp_path / "last", config_hash="c" * 64, manifest_sha="m" * 64,
                                   base_sha="b" * 64, world_size=1)
