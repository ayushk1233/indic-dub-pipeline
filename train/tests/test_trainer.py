import json
import math

import pytest
import torch

from train import lora
from train.data import SyntheticDataset
from train.model import build_cfm
from train.store import LocalStore, read_latest
from train.trainer import DivergenceError, Trainer, lr_factor, step_outcome


def _trainer(tiny_cfg, vocab, tmp_path, *, n=48, clock=None, on_micro=None, cfg_over=None):
    for k, v in (cfg_over or {}).items():
        tiny_cfg[k].update(v)
    torch.manual_seed(0)
    model = lora.attach(build_cfm(tiny_cfg, vocab), tiny_cfg)
    ds = SyntheticDataset(n, 3, vocab, min_frames=100, max_frames=100)
    kw = dict(model=model, train_set=ds, val_items=SyntheticDataset(4, 9, vocab).val_items(),
              store=LocalStore(tmp_path / "hub"), work_dir=tmp_path / "work", base_sha="b" * 64,
              manifest_sha=ds.manifest_sha256(), on_micro=on_micro)
    if clock is not None:
        kw["clock"] = clock
    return Trainer(tiny_cfg, **kw), ds


def _log(tmp_path):
    return [json.loads(l) for l in (tmp_path / "work" / "log.jsonl").read_text().splitlines()]


def test_lr_warms_up_then_decays_to_final_fraction():
    assert lr_factor(0, 10, 100, 0.1) == pytest.approx(0.1)
    assert lr_factor(9, 10, 100, 0.1) == pytest.approx(1.0)
    assert lr_factor(100, 10, 100, 0.1) == pytest.approx(0.1)
    assert 0.1 < lr_factor(55, 10, 100, 0.1) < 1.0


def test_step_outcome_scaler_skip_is_not_nonfinite():
    assert step_outcome(1.0, math.inf, scaler_active=True, step_skipped=True) == "scaler_skip"
    assert step_outcome(1.0, 2.0, scaler_active=True, step_skipped=False) == "ok"
    assert step_outcome(math.nan, math.nan, scaler_active=True, step_skipped=True) == "nonfinite"
    assert step_outcome(1.0, math.inf, scaler_active=False, step_skipped=False) == "nonfinite"


def test_full_run_counts_updates_logs_and_publishes(tiny_cfg, vocab, tmp_path):
    t, _ = _trainer(tiny_cfg, vocab, tmp_path)
    assert t.run() == "done"
    assert t.state["update"] == t.total_updates == t.updates_per_epoch * 3
    updates = [e for e in _log(tmp_path) if "outcome" in e]
    assert [e["update"] for e in updates] == list(range(1, t.total_updates + 1))
    assert all(math.isfinite(e["loss"]) for e in updates)
    assert any(e.get("event") == "validation" for e in _log(tmp_path))
    assert read_latest(LocalStore(tmp_path / "hub"))["update"] == t.total_updates
    assert (tmp_path / "work" / "epoch_2").exists() and not (tmp_path / "work" / "epoch_0").exists()


def test_poisoned_batches_roll_back_once_then_stop(tiny_cfg, vocab, tmp_path):
    holder = {}

    def poison_after(update, micro):
        if update > 8:
            holder["ds"].poison = True

    t, ds = _trainer(tiny_cfg, vocab, tmp_path, on_micro=poison_after,
                     cfg_over={"checkpoint": {"every_updates": 5}})
    holder["ds"] = ds
    with pytest.raises(DivergenceError, match="after one rollback"):
        t.run()
    events = [e.get("event") for e in _log(tmp_path)]
    assert events.count("rollback") == 1
    rollback = next(e for e in _log(tmp_path) if e.get("event") == "rollback")
    assert rollback["update"] == 5 and rollback["lr_scale"] == 0.5


def test_three_nonfinite_without_a_checkpoint_stops(tiny_cfg, vocab, tmp_path):
    t, ds = _trainer(tiny_cfg, vocab, tmp_path)
    ds.poison = True
    with pytest.raises(DivergenceError, match="no checkpoint"):
        t.run()


def test_time_guard_saves_uploads_and_returns(tiny_cfg, vocab, tmp_path):
    ticks = iter(range(0, 10**6, 10))
    t, _ = _trainer(tiny_cfg, vocab, tmp_path, clock=lambda: next(ticks),
                    cfg_over={"runtime": {"guard_seconds": 45}})
    assert t.run() == "time_guard"
    latest = read_latest(LocalStore(tmp_path / "hub"))
    assert latest["update"] == t.state["update"] < t.total_updates
    assert any(e.get("event") == "time_guard" for e in _log(tmp_path))
