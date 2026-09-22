"""Final-review fixes for the GPU pre-flight (step-4 plan): each test reproduces one finding."""
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from train import preflight_runners as pr
from train.run import apply_overrides
from train.sampler import ResumableBatchSampler

OVERFIT = Path("finetune_data/preflight/route_a/overfit16")


def _overfit_frames():
    if OVERFIT.exists():
        from train.data import ManifestDataset

        d = ManifestDataset(OVERFIT, root=OVERFIT.parent.parent)
        return [d.get_frame_len(i) for i in range(len(d))]
    return [460.0] * 16                                  # the real set: 16 clips, 7,350 frames


@pytest.mark.parametrize("route", ["a", "b"])
def test_g6_config_gives_many_updates_per_epoch_on_overfit16(route):
    """Review C1: with the route's batch (12000 or 8000 frames x accum) the 16 clips made 0 updates per epoch."""
    cfg = apply_overrides(pr._cfg(route), pr.G6_SETS)
    b = cfg["batch"]
    batches = ResumableBatchSampler(_overfit_frames(), b["frames_per_gpu"], b["max_samples"], 0, 1, 0).epoch_batches(0)
    assert len(batches) // b["grad_accum"] >= 3
    assert cfg["checkpoint"]["epoch_saves"] is False     # ~100 epochs must not write ~100 epoch checkpoints


def test_epoch_saves_off_writes_no_epoch_checkpoints(tiny_cfg, vocab, tmp_path):
    from train.tests.test_trainer import _trainer

    cfg_over = {"checkpoint": {"epoch_saves": False}, "schedule": {"epochs": 2}}
    t, _ = _trainer(tiny_cfg, vocab, tmp_path, cfg_over=cfg_over)
    assert t.run() == "done"
    assert not list((tmp_path / "hub").glob("epoch_*")) and not list((tmp_path / "work").glob("epoch_*"))
    assert t.state["epoch"] == 2


def test_collective_probe_runs_as_a_script_on_gloo():
    """Review C2: mp.spawn cannot unpickle a function defined in `python -c`; the probe must be a real script."""
    seconds = pr.collective_probe(backend="gloo", timeout=120)
    assert seconds is not None and seconds < 10


def test_main_clears_stale_verdict_and_turns_exceptions_into_fail(tmp_path, monkeypatch):
    """Review I3: a raising runner left the previous run's gN.json in place."""
    (tmp_path / "g4.json").write_text(json.dumps({"status": "pass", "detail": "old run"}))

    def boom(**kw):
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(pr, "g4", boom)
    pr.main(["g4", "--out", str(tmp_path)])
    res = json.loads((tmp_path / "g4.json").read_text())
    assert res["status"] == "fail" and "disk on fire" in res["detail"]


def test_train_starts_clean_merges_nccl_env_and_times_out(tmp_path, monkeypatch):
    """Review I3/I4: stale work dirs resumed old runs; G1's NCCL workaround never reached training runs; no timeout."""
    (tmp_path / "nccl_env.json").write_text(json.dumps({"NCCL_P2P_DISABLE": "1"}))
    work, hub = tmp_path / "w", tmp_path / "hub"
    (work / "old").mkdir(parents=True)
    (hub / "LATEST").parent.mkdir(parents=True, exist_ok=True)
    (hub / "LATEST").write_text("{}")
    seen = {}

    def fake_run(cmd, env, capture_output, text, timeout):
        seen.update(env=env, timeout=timeout, work_exists=work.exists(), hub_exists=hub.exists())
        raise subprocess.TimeoutExpired(cmd, timeout)
    monkeypatch.setattr(pr.subprocess, "run", fake_run)
    r = pr._train("base", "d", "v", work, f"local:{hub}", out=tmp_path, timeout=5)
    assert seen["env"]["NCCL_P2P_DISABLE"] == "1" and seen["timeout"] == 5
    assert not seen["work_exists"] and not seen["hub_exists"]
    assert r.returncode == 124 and "timed out" in r.stderr


@pytest.mark.parametrize("frames", [4000, 9000, 11000, 13000, 40000])
def test_g7_measures_the_frames_it_reports(frames):
    """Review I5: frames // 1875 clips of 1875 measured up to 16% fewer frames than reported."""
    n, length = pr.g7_batch_shape(frames, max_samples=32)
    assert n <= 32 and length <= pr.FRAMES_20S
    assert n * length >= frames - n                         # within rounding of the frames it claims
    assert n == max(1, min(32, math.ceil(frames / pr.FRAMES_20S)))


def test_notebook_documents_upload_and_falls_back_without_ckpt_repo():
    """Review I6: the dataset layout was undocumented; G14 required a repo that may not exist."""
    src = "\n".join("".join(c["source"]) for c in json.loads(Path("train/kaggle/preflight_gpu.ipynb").read_text())["cells"])
    assert "hf upload ayushk1233/indicf5-finetune-preflight" in src
    assert "repo_exists" in src
