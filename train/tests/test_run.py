import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
from safetensors.torch import load_file

from train.config import load_config
from train.data import write_manifest_dir
from train.model import build_cfm, load_vocab, save_base
from train.store import LocalStore, download_checkpoint, read_latest

TINY = Path("train/tests/configs/tiny.yaml")
DATA = "synthetic:164:3:100"      # 164 items x 100 frames -> 41 batches of 4 -> 20 updates/epoch, one batch left over


def _base(tmp_path):
    cfg = load_config(TINY)
    torch.manual_seed(0)
    return str(tmp_path / "base.safetensors"), save_base(build_cfm(cfg, load_vocab()), tmp_path / "base.safetensors")


def _run(tmp_path, name, store, base, *extra, config=TINY, data=DATA):
    cmd = [sys.executable, "-m", "train.run", "--config", str(config), "--work-dir", str(tmp_path / name),
           "--store", f"local:{store}", "--base", base, "--data", data, "--val", "synthetic:4:9", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=900)


def _losses(work):
    return {e["update"]: e["loss"] for e in map(json.loads, (work / "log.jsonl").read_text().splitlines())
            if e.get("outcome") == "ok"}


def _final(store, tmp_path, tag):
    d = download_checkpoint(LocalStore(store), read_latest(LocalStore(store)), tmp_path / f"final_{tag}")
    return load_file(str(d / "trainable.safetensors"))


@pytest.fixture(scope="module")
def straight(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("straight")
    base, _ = _base(tmp)
    r = _run(tmp, "work", tmp / "hub", base)
    assert r.returncode == 0, r.stderr[-2000:]
    return tmp, base, _losses(tmp / "work"), _final(tmp / "hub", tmp, "s")


@pytest.mark.c6
@pytest.mark.slow
@pytest.mark.parametrize("die_at", ["31:0", "38:1", "41:0"])   # after a checkpoint; mid accumulation; across the epoch boundary at 40
def test_kill_and_resume_matches_the_uninterrupted_run(straight, tmp_path, die_at):
    _, base, want_losses, want_final = straight
    killed = _run(tmp_path, "leg1", tmp_path / "hub", base, "--die-at", die_at)
    assert killed.returncode == 137, killed.stderr[-2000:]
    resumed = _run(tmp_path, "leg2", tmp_path / "hub", base)          # fresh work dir: a new container
    assert resumed.returncode == 0, resumed.stderr[-2000:]
    assert "resumed at update" in resumed.stdout
    got = {**_losses(tmp_path / "leg1"), **_losses(tmp_path / "leg2")}
    assert got == want_losses
    final = _final(tmp_path / "hub", tmp_path, "k")
    assert set(final) == set(want_final) and all(torch.equal(final[k], want_final[k]) for k in final)


@pytest.mark.c8
@pytest.mark.slow
def test_time_guard_saves_uploads_and_exits_zero(tmp_path):
    base, _ = _base(tmp_path)
    cfg = yaml.safe_load(TINY.read_text()); cfg["schedule"]["epochs"] = 10_000
    (tmp_path / "long.yaml").write_text(yaml.safe_dump(cfg))
    r = _run(tmp_path, "work", tmp_path / "hub", base, "--guard-seconds", "3", config=tmp_path / "long.yaml")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "stopped: time_guard" in r.stdout
    latest = read_latest(LocalStore(tmp_path / "hub"))
    download_checkpoint(LocalStore(tmp_path / "hub"), latest, tmp_path / "check")   # verifies hashes


def _wav_manifest(root, texts):
    rows = []
    for i, t in enumerate(texts):
        p = root / "wav" / f"{i}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, (0.1 * np.sin(np.arange(24000) * (i + 1) / 40)).astype("float32"), 24000)
        rows.append({"audio_path": str(p), "text": t, "duration": 1.0})
    write_manifest_dir(rows, root / "m")
    return root / "m"


@pytest.mark.c12
@pytest.mark.slow
def test_resume_refuses_a_changed_manifest_row_or_config_value(tmp_path):
    base, _ = _base(tmp_path)
    texts = ["कखग", "गघच", "छजझ", "टठड"] * 4
    data = f"manifest:{_wav_manifest(tmp_path / 'a', texts)}"
    cfg = yaml.safe_load(TINY.read_text()); cfg["schedule"]["epochs"] = 1; cfg["checkpoint"]["every_updates"] = 1
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    assert _run(tmp_path, "w1", tmp_path / "hub", base, config=tmp_path / "c.yaml", data=data).returncode == 0

    changed = f"manifest:{_wav_manifest(tmp_path / 'b', ['कखघ'] + texts[1:])}"
    r = _run(tmp_path, "w2", tmp_path / "hub", base, config=tmp_path / "c.yaml", data=changed)
    assert r.returncode != 0 and "refusing to resume: manifest_sha256 differs" in r.stderr

    cfg["lora"]["dropout"] = 0.1
    (tmp_path / "c2.yaml").write_text(yaml.safe_dump(cfg))
    r = _run(tmp_path, "w3", tmp_path / "hub", base, config=tmp_path / "c2.yaml", data=data)
    assert r.returncode != 0 and "refusing to resume: config_hash differs" in r.stderr
