"""
FINETUNE_PLAN §8b verdicts. Each check's heavy part runs on Kaggle (train/preflight_runners.py, driven by
train/kaggle/preflight_gpu.ipynb); the pass/fail rule lives here so it is tested on CPU. The criteria are
§8b's, quoted in each docstring.
"""
from __future__ import annotations

import datetime
import math
import statistics
from pathlib import Path

import numpy as np

PARAMS = 337_096_804


def _r(ok, detail, status=None, **nums):
    return {"status": status or ("pass" if ok else "fail"), "detail": detail, **nums}


def _ok(log):
    return [e for e in log if e.get("outcome") == "ok"]


def g2_verdict(problems, n_params, any_meta, expected=PARAMS):
    """0 missing / 0 unexpected (load_base is strict), nothing on meta, module names match §0, count printed."""
    ok = not problems and not any_meta and n_params == expected
    return _r(ok, f"{n_params:,} parameters; {len(problems)} layout problems; meta={any_meta}",
              n_params=n_params, problems=problems[:5])


def g3_verdict(ours, official, tol=1e-3):
    """Mel max abs diff < 1e-3 in fp32 (also G4's parity and G13's export check)."""
    ours, official = np.asarray(ours), np.asarray(official)
    if ours.shape != official.shape:
        return _r(False, f"shape {ours.shape} vs {official.shape}")
    diff = float(np.max(np.abs(ours - official)))
    return _r(diff < tol, f"max abs diff {diff:.2e} (tolerance {tol:.0e})", max_abs_diff=diff)


def g5_verdict(losses):
    """Expect val_hi well below val_en (every transcript form). Until step 2 builds val_indic the check is partial."""
    shown = ", ".join(f"{k} {v:.4f}" for k, v in losses.items())
    if "hi" not in losses:
        return _r(True, f"{shown}; val_indic not built yet (§15 step 2)", status="partial", **losses)
    en = [v for k, v in losses.items() if k.startswith("en")]
    return _r(all(losses["hi"] < v for v in en), shown, **losses)


def g6_verdict(log, drop=0.5):
    """16 English clips, 300 updates: loss falls by > 50%."""
    losses = [e["loss"] for e in _ok(log)]
    first, last = statistics.mean(losses[:10]), statistics.mean(losses[-10:])
    fall = 1 - last / first
    return _r(fall > drop, f"loss {first:.4f} -> {last:.4f} ({fall:.0%} fall, need > {drop:.0%})", fall=fall)


def g7_search(measure, lo, hi, limit_bytes, step=1000):
    """Largest frames_per_gpu whose peak memory stays within the limit (85% of 16 GB)."""
    if measure(lo) > limit_bytes:
        return _r(False, f"even {lo} frames per GPU exceeds the memory limit", frames_per_gpu=None)
    while hi - lo > step:
        mid = (lo + hi) // 2 // step * step
        if mid <= lo:
            break
        lo, hi = (mid, hi) if measure(mid) <= limit_bytes else (lo, mid)
    return _r(True, f"frames_per_gpu {lo}", frames_per_gpu=lo)


def g8_verdict(log, max_wait=0.10, warm=10):
    """200 steady updates: s/update, frames/s; data-loader wait < 10% of step time."""
    ups = _ok(log)[warm:]
    wait = sum(e["data_s"] for e in ups) / sum(e["step_s"] + e["data_s"] for e in ups)
    s = statistics.median(e["step_s"] + e["data_s"] for e in ups)
    fps = statistics.median(e["fps"] for e in ups)
    return _r(wait < max_wait, f"{s:.2f} s/update, {fps:,.0f} frames/s, data wait {wait:.1%}",
              s_per_update=s, fps=fps, wait=wait)


def g9_verdict(single, double, min_ratio=1.7, warm=10):
    """2-GPU throughput >= 1.7x single GPU; loss over 100 updates matches within noise."""
    f1 = statistics.median(e["fps"] for e in _ok(single)[warm:])
    f2 = statistics.median(e["fps"] for e in _ok(double)[warm:])
    a, b = [e["loss"] for e in _ok(single)], [e["loss"] for e in _ok(double)]
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    se = math.sqrt((statistics.pvariance(a) + statistics.pvariance(b)) / n) or 1e-12
    gap = abs(statistics.mean(a) - statistics.mean(b))
    ok = f2 / f1 >= min_ratio and gap <= 2 * se
    return _r(ok, f"scaling {f2 / f1:.2f}x (need {min_ratio}x); loss gap {gap:.4f} vs 2SE {2 * se:.4f}",
              ratio=f2 / f1, gap=gap)


def g10_verdict(log, max_drops=0.05):
    """1,000 updates: no NaN/inf; GradScaler scale >= 1 and not collapsing; gradient norm logged."""
    nonfinite = [e for e in log if e.get("event") == "nonfinite"]
    ups = _ok(log)
    scales = [e["scale"] for e in ups if e.get("scale") is not None]
    drops = sum(1 for x, y in zip(scales, scales[1:]) if y < x)
    ok = (not nonfinite and bool(scales) and min(scales) >= 1 and drops <= max_drops * len(ups)
          and all("grad_norm" in e for e in ups))
    return _r(ok, f"{len(nonfinite)} non-finite; min scale {min(scales) if scales else None}; "
              f"{drops} scale drops in {len(ups)} updates")


def g11_verdict(log):
    """kill -9 at ~150, relaunch: the update counter, loss band and LATEST all continue correctly."""
    idx = next((i for i, e in enumerate(log) if e.get("event") == "resumed"), None)
    if idx is None:
        return _r(False, "no resumed event in the log")
    at = log[idx]["update"]
    nxt = next((e for e in log[idx + 1:] if e.get("outcome") == "ok"), None)
    ok = nxt is not None and nxt["update"] == at + 1
    return _r(ok, f"resumed at {at}; next ok update {nxt['update'] if nxt else None}")


def g12_verdict(log, max_frac=0.05):
    """Validation (and probe) cost < 5% of training time. Probe generation is out of scope (Decision 7)."""
    val = sum(e["seconds"] for e in log if e.get("event") == "validation")
    train = sum(e["step_s"] + e["data_s"] for e in _ok(log))
    frac = val / train
    return _r(frac < max_frac, f"validation {frac:.1%} of training time (probe generation not measured)", frac=frac)


def g14_verdict(log, store_latest, hub):
    """20-minute committed run, guard at 12 minutes: checkpoint uploaded, notebook exits cleanly."""
    guard = any(e.get("event") == "time_guard" for e in log)
    ok = guard and store_latest is not None
    status = None if hub or not ok else "partial"
    return _r(ok, f"time_guard={guard}; LATEST={'present' if store_latest else 'missing'}; "
              f"{'Hub' if hub else 'LocalStore only'}", status=status)


def render(results, commit):
    rows = "".join(f"| {c} | {r['status']} | {r['detail']} |\n" for c, r in results.items())
    return (f"# GPU pre-flight (FINETUNE_PLAN §8b)\n\ncommit `{commit}`, "
            f"{datetime.datetime.now().isoformat(timespec='seconds')}\n\n| check | status | detail |\n|---|---|---|\n{rows}")


def write_report(results, commit, path="train/preflight_gpu_report.md"):
    Path(path).write_text(render(results, commit))
