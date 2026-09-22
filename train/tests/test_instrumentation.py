import json
import subprocess
import sys

import pytest
import torch

from train.run import apply_overrides

TINY = "train/tests/configs/tiny.yaml"


def test_apply_overrides_sets_nested_yaml_values():
    cfg = {"lora": {"lr": 1e-4}, "schedule": {"epochs": 10}}
    out = apply_overrides(cfg, ["lora.lr=5.0e-4", "schedule.max_updates=300", "batch.workers=2"])
    assert out["lora"]["lr"] == 5e-4 and out["schedule"]["max_updates"] == 300 and out["batch"]["workers"] == 2


def test_apply_overrides_rejects_malformed():
    with pytest.raises(SystemExit):
        apply_overrides({}, ["no_equals_sign"])


def _run(tmp_path, *extra):
    from train.tests.test_run import _base

    tmp_path.mkdir(parents=True, exist_ok=True)
    base, _ = _base(tmp_path)
    cmd = [sys.executable, "-m", "train.run", "--config", TINY, "--work-dir", str(tmp_path / "w"),
           "--store", f"local:{tmp_path / 'hub'}", "--base", base, "--data", "synthetic:164:3:100",
           "--val", "synthetic:4:9", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    return r, [json.loads(l) for l in (tmp_path / "w" / "log.jsonl").read_text().splitlines()]


@pytest.mark.slow
def test_update_log_carries_metrics_and_max_updates_stops(tmp_path):
    r, log = _run(tmp_path, "--set", "schedule.max_updates=7")
    ups = [e for e in log if e.get("outcome") == "ok"]
    assert ups[-1]["update"] == 7 and "stopped: max_updates" in r.stdout
    for k in ("step_s", "data_s", "frames", "fps", "mem_gb", "scale"):
        assert k in ups[0], k
    assert ups[0]["frames"] > 0 and ups[0]["fps"] > 0
    assert all("seconds" in e for e in log if e.get("event") == "validation")


@pytest.mark.slow
def test_workers_do_not_change_losses(tmp_path):
    _, a = _run(tmp_path / "a", "--set", "schedule.max_updates=25")
    _, b = _run(tmp_path / "b", "--set", "schedule.max_updates=25", "--set", "batch.workers=2")
    la = [e["loss"] for e in a if e.get("outcome") == "ok"]
    lb = [e["loss"] for e in b if e.get("outcome") == "ok"]
    assert la == lb and len(la) == 25


@pytest.mark.slow
def test_rollback_rebuilds_prefetch(tiny_cfg, vocab, tmp_path):
    """After a rollback, the next group fed to the model is the one at the restored batch index, not a stale prefetch."""
    from train import lora
    from train.data import SyntheticDataset
    from train.model import build_cfm
    from train.patches import import_vendor
    from train.store import LocalStore
    from train.trainer import Trainer

    cfg = apply_overrides(tiny_cfg, ["batch.workers=2", "checkpoint.every_updates=5", "schedule.max_updates=12"])
    torch.manual_seed(0)
    ds = SyntheticDataset(164, 3, vocab, 100, 100)
    t = Trainer(cfg, model=lora.attach(build_cfm(cfg, vocab), cfg), train_set=ds,
                val_items=SyntheticDataset(4, 9, vocab).val_items(), store=LocalStore(tmp_path / "hub"),
                work_dir=tmp_path / "w", base_sha="b", manifest_sha=ds.manifest_sha256())
    batches = t.sampler.epoch_batches(0)
    collate = import_vendor().collate_fn

    def fingerprint(start):
        return float(collate([ds[j] for j in batches[start]])["mel"].sum())

    seen, calls, real = [], {"n": 0}, t._update

    def watched(micro, **kw):
        calls["n"] += 1
        seen.append((t.state["epoch"], t.state["batch"], float(micro[0]["mel"].sum())))
        if 8 <= calls["n"] <= 10:                       # three non-finite updates -> roll back to update 5
            micro = [dict(m, mel=m["mel"] * float("nan")) for m in micro]
        real(micro, **kw)
    t._update = watched
    assert t.run() == "max_updates"
    log = [json.loads(l) for l in (tmp_path / "w" / "log.jsonl").read_text().splitlines()]
    assert [e["update"] for e in log if e.get("event") == "rollback"] == [5]
    assert all(fp == pytest.approx(fingerprint(b)) for e, b, fp in seen if e == 0), seen
    assert t.state["update"] == 12
