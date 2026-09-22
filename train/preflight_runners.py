"""
The heavy half of FINETUNE_PLAN §8b: one runner per GPU check, each returning a verdict from
train/preflight_gpu.py and writing OUT/<check>.json. Driven cell by cell by train/kaggle/preflight_gpu.ipynb:

  python -m train.preflight_runners g6 --route a --base <release file> --data <dataset root> --out <dir>

Training runs (G6, G8-G11, G14) are subprocesses of train.run, exactly as a real session launches them.
G3 and G13 compare against an .npz written by train/kaggle/official_mel.py in the shipping environment.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from train import preflight_gpu as v
from train.config import load_config
from train.model import build_cfm, layout_problems, load_base, load_vocab
from train.run import apply_overrides

RELEASE_SHA = "ba7f3671180fb7784e24bd1dafc96e729a38ce02e7f6d3877cdef32525a1865c"
FRAMES_20S = 1875                                   # 20 s at 93.75 frames/s, the longest clip §2c allows


def _cfg(route):
    return load_config(f"train/configs/preflight_{route}.yaml")


def _model(base, cfg, device="cpu"):
    model = build_cfm(cfg, load_vocab())
    load_base(model, base)
    return model.to(device).float()


def _log(work):
    p = Path(work) / "log.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def _train(base, data_dir, val_dir, work, store, *, route="a", gpus=2, sets=(), extra=(), out=None, timeout=None,
           fresh=True):
    """One train.run session. gpus=2 goes through accelerate launch, as the main sessions will (§6a).
    fresh: remove this run's work dir and LocalStore first, so a rerun never resumes or re-reads an old run.
    out: G1's nccl_env.json there (NCCL_P2P_DISABLE when P2P hangs) is applied. timeout: seconds, then exit 124."""
    if fresh:
        shutil.rmtree(work, ignore_errors=True)
        if store.startswith("local:"):
            shutil.rmtree(store[len("local:"):], ignore_errors=True)
    argv = ["-m", "train.run", "--config", f"train/configs/preflight_{route}.yaml", "--work-dir", str(work),
            "--store", store, "--base", str(base), "--data", f"manifest:{data_dir}", "--val", f"manifest:{val_dir}"]
    for s in sets:
        argv += ["--set", s]
    argv += list(extra)
    env = dict(os.environ)
    nccl = Path(out) / "nccl_env.json" if out else None
    if nccl is not None and nccl.exists():
        env.update(json.loads(nccl.read_text()))
    if gpus == 2:
        launcher = str(Path(sys.executable).with_name("accelerate"))
        cmd = [launcher, "launch", "--multi_gpu", "--num_processes", "2", "--mixed_precision", "fp16", *argv]
    else:
        env["CUDA_VISIBLE_DEVICES"] = "0"
        cmd = [sys.executable, *argv]
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        r = subprocess.CompletedProcess(cmd, 124, _text(e.stdout), _text(e.stderr) + f"\ntimed out after {timeout} s")
    Path(work).mkdir(parents=True, exist_ok=True)
    (Path(work) / "stdout.txt").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    return r


def _text(b):
    return b.decode(errors="replace") if isinstance(b, bytes) else (b or "")


def _frames_sets(out):
    """Use G7's measured frames_per_gpu when it is below the config's (the plan's number is an estimate)."""
    g7 = Path(out) / "g7_a.json"                 # G8-G14 run Route A
    if g7.exists():
        f = json.loads(g7.read_text()).get("frames_per_gpu")
        if f and f < _cfg("a")["batch"]["frames_per_gpu"]:
            return [f"batch.frames_per_gpu={f}"]
    return []


# ---- G1
_PROBE = """
import sys, time, torch, torch.distributed as dist, torch.multiprocessing as mp

def run(r, backend, out, port):
    dist.init_process_group(backend, rank=r, world_size=2, init_method=f"tcp://127.0.0.1:{port}")
    cuda = backend == "nccl"
    x = torch.ones(1 << 20, device=r if cuda else "cpu")
    if cuda: torch.cuda.synchronize()
    t = time.time()
    dist.all_reduce(x)
    if cuda: torch.cuda.synchronize()
    if r == 0: open(out, "w").write(str(time.time() - t))
    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(run, args=(sys.argv[1], sys.argv[2], int(sys.argv[3])), nprocs=2)
"""


def collective_probe(backend="nccl", env_extra=None, timeout=60):
    """Seconds for one 2-process all_reduce of 1M floats, or None if it failed or hung. mp.spawn children
    re-import __main__, so the probe is a real file with a main guard, never `python -c` (review C2)."""
    import socket

    with tempfile.TemporaryDirectory() as d, socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        script, result = Path(d) / "probe.py", Path(d) / "seconds.txt"
        script.write_text(_PROBE)
        sock.close()
        try:
            subprocess.run([sys.executable, str(script), backend, str(result), str(port)],
                           env={**os.environ, **(env_extra or {})}, timeout=timeout, check=True, capture_output=True)
            return float(result.read_text())
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, FileNotFoundError):
            return None


def g1(out=None, **_):
    info = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpus": torch.cuda.device_count(),
            "names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    with torch.autocast("cuda", dtype=torch.float16):
        y = torch.randn(1024, 1024, device="cuda") @ torch.randn(1024, 1024, device="cuda")
    fp16_ok = bool(torch.isfinite(y).all())
    seconds, note, used = None, "", {}
    for env_extra in ({}, {"NCCL_P2P_DISABLE": "1"}):
        seconds = collective_probe("nccl", env_extra)
        if seconds is not None:
            used, note = env_extra, " (NCCL_P2P_DISABLE=1 needed; applied to every later run)" if env_extra else ""
            break
    if out:
        Path(out, "nccl_env.json").write_text(json.dumps(used))
    ok = info["gpus"] == 2 and fp16_ok and seconds is not None and seconds < 10
    return v._r(ok, f"{info}; fp16 autocast finite={fp16_ok}; all_reduce {seconds}s{note}", **info)


# ---- G2
def g2(base, cfg=None, expected_params=v.PARAMS, out=None, **_):
    cfg = cfg or _cfg("a")
    model = build_cfm(cfg, load_vocab())
    load_base(model, base)                              # strict: 0 missing, 0 unexpected, or it raises
    if out:
        Path(out, "g2_modules.txt").write_text("\n".join(n for n, _ in model.named_modules()))
    return v.g2_verdict(layout_problems(model, cfg), sum(p.numel() for p in model.parameters()),
                        any(p.is_meta for p in model.parameters()), expected=expected_params)


# ---- G3 / G13
def _ours_vs_official(model, official):
    import soundfile as sf

    from train.generate import generate
    from train.vocoder import load_vocoder

    npz = np.load(official, allow_pickle=False)
    audio, sr = sf.read(str(npz["ref_path"]), dtype="float32", always_2d=True)
    assert sr == 24000, f"reference must be 24 kHz, got {sr}"
    ref = torch.from_numpy(audio.T).mean(dim=0, keepdim=True)
    ours = generate(model, ref, str(npz["ref_text"]), str(npz["gen_text"]), seed=int(npz["seed"]),
                    fix_duration=float(npz["fix_duration"]), vocoder=None)
    return v.g3_verdict(ours["mel"].numpy(), npz["mel"])


def g3(base, official, cfg=None, **_):
    cfg = cfg or _cfg("a")
    return _ours_vs_official(_model(base, cfg, "cuda"), official)


# ---- G4
def g4(data, cfg=None, **_):
    import soundfile as sf

    from train.data import ManifestDataset
    from train.patches import import_vendor

    cfg = cfg or _cfg("a")
    ds = ManifestDataset(Path(data) / "route_a" / "val_en", root=data)
    item = ds[0]
    audio, _ = sf.read(ds._inner.data[0]["audio_path"], dtype="float32", always_2d=True)
    wave = torch.from_numpy(audio.T).mean(dim=0, keepdim=True)
    infer_mel = import_vendor().MelSpec(**cfg["mel"])(wave).squeeze(0)
    res = v.g3_verdict(item["mel_spec"].numpy(), infer_mel.numpy())
    rms = float(torch.sqrt(torch.mean(wave ** 2)))
    res["detail"] += f"; clip rms {rms:.4f}, inference would scale a reference like it by {max(1.0, 0.1 / rms):.2f}x"
    return res


# ---- G5
def g5(base, data, out=None, **_):
    from train.data import ManifestDataset
    from train.validation import FixedNoiseValidator

    losses = {}
    for route in ("a", "b"):
        cfg = _cfg(route)
        model = _model(base, cfg, "cuda" if torch.cuda.is_available() else "cpu")
        items = ManifestDataset(Path(data) / f"route_{route}" / "val_en", root=data).val_items()
        losses[f"en_{route}"] = FixedNoiseValidator(items, model.vocab_char_map, cfg["validation"]["seed"]).evaluate(model)["en"]
    if out:
        Path(out, "baselines.json").write_text(json.dumps(losses, indent=1))
    return v.g5_verdict(losses)


# ---- G6
# 16 clips are ~7,350 frames: the route's batch (12,000 or 8,000 frames x accum) made 0 updates per epoch
# (review C1). ~2,000 frames and no accumulation give ~4 updates per epoch; with ~75 epochs in 300 updates,
# epoch checkpoints are switched off so the store does not grow by ~75 of them.
G6_SETS = ["lora.lr=5.0e-4", "optim.warmup_updates=20", "optim.final_lr_frac=1.0", "schedule.max_updates=300",
           "schedule.epochs=100000", "validation.every_updates=100000", "ema.start_after=100000",
           "batch.workers=0", "batch.frames_per_gpu=2000", "batch.grad_accum=1", "checkpoint.epoch_saves=false"]


def g6(base, data, out, route="a", **_):
    import soundfile as sf

    from train import lora
    from train.data import ManifestDataset
    from train.generate import generate
    from train.store import LocalStore, download_checkpoint, read_latest
    from train.vocoder import load_vocoder
    from safetensors.torch import load_file

    d = Path(data) / f"route_{route}"
    work, hub = Path(out) / f"g6_{route}_work", Path(out) / f"g6_{route}_hub"
    r = _train(base, d / "overfit16", d / "val_en", work, f"local:{hub}", route=route, gpus=1, sets=G6_SETS,
               out=out, timeout=3600)
    if r.returncode != 0:
        return v._r(False, f"train.run exited {r.returncode}: {r.stderr[-300:]}")
    res = v.g6_verdict(_log(work))
    cfg = apply_overrides(_cfg(route), G6_SETS)
    model = lora.attach(_model(base, cfg), cfg)
    ck = download_checkpoint(LocalStore(hub), read_latest(LocalStore(hub)), Path(out) / f"g6_{route}_final")
    lora.load_trainable(model, load_file(str(ck / "trainable.safetensors")))
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    ds = ManifestDataset(d / "overfit16", root=data)
    audio, _ = sf.read(ds._inner.data[0]["audio_path"], dtype="float32", always_2d=True)
    gen_text = ds._inner.data[1]["text"]
    wav = generate(model, torch.from_numpy(audio.T).mean(0, keepdim=True), ds._inner.data[0]["text"], gen_text,
                   seed=1234, vocoder=load_vocoder(base).to(next(model.parameters()).device))["wave"]
    sf.write(Path(out) / f"g6_{route}_overfit.wav", wav, 24000)
    Path(out, f"g6_{route}_overfit.txt").write_text(gen_text)
    res["detail"] += f"; g6_{route}_overfit.wav saved for scoring in the eval env"
    return res


# ---- G7
def g7_batch_shape(frames, max_samples):
    """Clips of at most 20 s that together hold `frames` (review I5: frames // 1875 full clips measured up to 16%
    fewer frames than reported). Memory tracks total tokens; the 20 s cap keeps attention at its longest."""
    n = max(1, min(max_samples, math.ceil(frames / FRAMES_20S)))
    return n, min(FRAMES_20S, math.ceil(frames / n))


def g7(base, route="a", **_):
    from train import lora

    cfg = _cfg(route)
    model = lora.attach(_model(base, cfg, "cuda"), cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    letters = [c for c in load_vocab() if ("a" <= c <= "z") or ("क" <= c <= "ह")]
    scaler = torch.amp.GradScaler("cuda")

    def measure(frames):
        n, length = g7_batch_shape(frames, cfg["batch"]["max_samples"])
        opt = torch.optim.AdamW(params, lr=1e-5, fused=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mel = torch.randn(n, length, 100, device="cuda")
        text = ["".join(letters[i % len(letters)] for i in range(k, k + min(300, length))) for k in range(n)]
        try:
            for _ in range(2):          # two steps: a scaler-skipped first step never allocates Adam state
                with torch.autocast("cuda", dtype=torch.float16):
                    loss, _, _ = model(mel, text=text, lens=torch.full((n,), length, device="cuda"))
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=False)
            peak = torch.cuda.max_memory_allocated()
        except torch.cuda.OutOfMemoryError:
            peak = float("inf")
        opt.zero_grad(set_to_none=True)
        del opt
        return peak

    limit = 0.85 * torch.cuda.get_device_properties(0).total_memory
    return v.g7_search(measure, 4000, 40000, limit)


# ---- G8-G12, G14: the trainer on 2 GPUs
def _ra(data):
    d = Path(data) / "route_a"
    return d / "train", d / "val_en"


def g8(base, data, out, **_):
    train, val = _ra(data)
    work = Path(out) / "g8_work"
    r = _train(base, train, val, work, f"local:{Path(out) / 'g8_hub'}",
               sets=["schedule.max_updates=220", "schedule.epochs=100000", "validation.every_updates=100000",
                     *_frames_sets(out)], out=out, timeout=3600)
    if r.returncode != 0:
        return v._r(False, f"train.run exited {r.returncode}: {r.stderr[-300:]}")
    return v.g8_verdict(_log(work))


def g9(base, data, out, **_):
    train, val = _ra(data)
    work = Path(out) / "g9_single_work"
    r = _train(base, train, val, work, f"local:{Path(out) / 'g9_hub'}", gpus=1,
               sets=["batch.grad_accum=4", "schedule.max_updates=110", "schedule.epochs=100000",
                     "validation.every_updates=100000", *_frames_sets(out)], out=out, timeout=3600)
    if r.returncode != 0:
        return v._r(False, f"single-GPU train.run exited {r.returncode}: {r.stderr[-300:]}")
    double = _log(Path(out) / "g8_work")
    if not double:
        return v._r(False, "G8's 2-GPU log is missing; run g8 first")
    return v.g9_verdict(_log(work), double)


def g10(base, data, out, **_):
    train, val = _ra(data)
    work = Path(out) / "g10_work"
    r = _train(base, train, val, work, f"local:{Path(out) / 'g10_hub'}",
               sets=["schedule.max_updates=1000", "schedule.epochs=100000", "validation.every_updates=500",
                     *_frames_sets(out)], out=out, timeout=7200)
    if r.returncode != 0:
        return v._r(False, f"train.run exited {r.returncode}: {r.stderr[-300:]}")
    return v.g10_verdict(_log(work))


def g11(base, data, out, **_):
    from train.store import LocalStore, read_latest

    train, val = _ra(data)
    store = f"local:{Path(out) / 'g11_hub'}"
    sets = ["checkpoint.every_updates=100", "schedule.max_updates=250", "schedule.epochs=100000",
            "validation.every_updates=100000", *_frames_sets(out)]
    leg1 = _train(base, train, val, Path(out) / "g11_leg1", store, sets=sets, extra=["--die-at", "150:0"],
                  out=out, timeout=3600)
    if leg1.returncode not in (137, 1):   # accelerate reports a child's os._exit(137) as its own failure
        return v._r(False, f"leg 1 was meant to die at update 150, exited {leg1.returncode}")
    leg2 = _train(base, train, val, Path(out) / "g11_leg2", store, sets=sets, out=out, timeout=3600,
                  fresh=False)                                    # the relaunch must find leg 1's checkpoint
    if leg2.returncode != 0:
        return v._r(False, f"relaunch exited {leg2.returncode}: {leg2.stderr[-300:]}")
    res = v.g11_verdict(_log(Path(out) / "g11_leg2"))
    latest = read_latest(LocalStore(Path(out) / "g11_hub"))
    if not latest or latest["update"] != 250:
        return v._r(False, res["detail"] + f"; LATEST {latest and latest['update']} != 250")
    return res


def g12(out, **_):
    log = _log(Path(out) / "g10_work")
    if not log:
        return v._r(False, "G10's log is missing; run g10 first")
    return v.g12_verdict(log)


def g13(base, out, official, **_):
    from safetensors.torch import load_file, save_file

    from train import export, lora
    from train.model import split_release, to_release
    from train.store import LocalStore, download_checkpoint, read_latest

    cfg = apply_overrides(_cfg("a"), G6_SETS)
    hub = Path(out) / "g6_a_hub"
    model = lora.attach(_model(base, cfg), cfg)
    ck = download_checkpoint(LocalStore(hub), read_latest(LocalStore(hub)), Path(out) / "g13_ck")
    lora.load_trainable(model, load_file(str(ck / "trainable.safetensors")))
    _, vocoder_sd = split_release(load_file(str(base)))
    merged = Path(out) / "g13_merged.safetensors"
    if not official:                  # first call exports (always fresh); the notebook then runs official_mel on it
        save_file(to_release(export.merged_state_dict(model), vocoder_sd), str(merged))
        return v._r(False, f"wrote {merged}; run official_mel.py --weights on it, then g13 again", status="not run")
    return _ours_vs_official(model.to("cuda" if torch.cuda.is_available() else "cpu"), official)


def g14(base, data, out, hub_repo=None, **_):
    from train.store import HfStore, LocalStore, read_latest

    train, val = _ra(data)
    store = f"hf:{hub_repo}" if hub_repo else f"local:{Path(out) / 'g14_hub'}"
    work = Path(out) / "g14_work"
    t0 = time.time()
    r = _train(base, train, val, work, store, sets=["schedule.epochs=100000", *_frames_sets(out)],
               extra=["--guard-seconds", "720"], out=out, timeout=2400)
    if r.returncode != 0:
        return v._r(False, f"train.run exited {r.returncode}: {r.stderr[-300:]}")
    st = HfStore(hub_repo, token=os.environ["HF_TOKEN"]) if hub_repo else LocalStore(Path(out) / "g14_hub")
    res = v.g14_verdict(_log(work), read_latest(st), hub=bool(hub_repo))
    res["detail"] += f"; wall {time.time() - t0:.0f}s"
    return res


CHECKS = [f"g{i}" for i in range(1, 15)]
# "the weather is lovely today so let us go for a walk", through src/text/transliterate.py (Route A form)
G3_TEXT = "द वेदर इज़ लव्ली टडे सो लेट अस गो फ़ॉर अ वॉक"


def request(data):
    """The one G3/G13 request both stacks run: the first Route A val_en clip as reference, a fixed sentence,
    fix_duration = reference seconds + 3 (FINETUNE_PLAN §0: total frames, reference included)."""
    from train.data import ManifestDataset

    row = ManifestDataset(Path(data) / "route_a" / "val_en", root=data)._inner.data[0]
    return {"ref": row["audio_path"], "ref_text": row["text"], "gen_text": G3_TEXT, "seed": 1234,
            "fix_duration": round(row["duration"] + 3.0, 3)}


def report(out):
    results = {}
    for i in range(1, 15):
        p = Path(out) / f"g{i}.json"
        results[f"G{i}"] = json.loads(p.read_text()) if p.exists() else {"status": "not run", "detail": "-"}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    v.write_report(results, commit)
    print(v.render(results, commit))


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m train.preflight_runners")
    p.add_argument("check", choices=[*CHECKS, "report", "request"])
    p.add_argument("--base")
    p.add_argument("--data")
    p.add_argument("--out", required=True)
    p.add_argument("--official")
    p.add_argument("--hub-repo")
    p.add_argument("--route", choices=["a", "b"], default="a")
    p.add_argument("--config")
    a = p.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.check == "report":
        return report(a.out)
    if a.check == "request":
        Path(a.out, "request.json").write_text(json.dumps(request(a.data), ensure_ascii=False, indent=1))
        return 0
    kw = dict(base=a.base, data=a.data, out=a.out, official=a.official, hub_repo=a.hub_repo, route=a.route)
    if a.config:
        kw["cfg"] = load_config(a.config)
    name = f"{a.check}_{a.route}" if a.check in ("g6", "g7") else a.check
    Path(a.out, f"{name}.json").unlink(missing_ok=True)          # never leave an earlier run's verdict behind
    try:
        res = globals()[a.check](**kw)
    except Exception as e:                                           # a crash is a failed check, with its cause
        res = v._r(False, f"{type(e).__name__}: {e} | {traceback.format_exc()[-600:]}")
    Path(a.out, f"{name}.json").write_text(json.dumps(res, indent=1, default=str))
    if a.check in ("g6", "g7"):                          # both routes must pass; G7 reports the smaller budget
        parts = [json.loads(q.read_text()) for q in sorted(Path(a.out).glob(f"{a.check}_[ab].json"))]
        combined = {"status": "pass" if len(parts) == 2 and all(x["status"] == "pass" for x in parts) else
                    ("fail" if any(x["status"] == "fail" for x in parts) else "partial"),
                    "detail": " | ".join(f"route {q.stem[-1]}: {x['detail']}"
                                         for q, x in zip(sorted(Path(a.out).glob(f"{a.check}_[ab].json")), parts))}
        if a.check == "g7":
            combined["frames_per_gpu"] = min((x.get("frames_per_gpu") or 0) for x in parts)
        Path(a.out, f"{a.check}.json").write_text(json.dumps(combined, indent=1))
    print(json.dumps(res, indent=1, default=str))
    return 0 if res["status"] != "fail" else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
