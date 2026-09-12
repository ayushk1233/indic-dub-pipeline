# ============================================================================
# DIAGNOSTIC DUMP — read-only, changes nothing, regenerates no audio.
# Paste the entire output back into the conversation.
# ============================================================================
import json, sys, wave, unicodedata
from pathlib import Path
import numpy as np

BUNDLE = Path("/content/tts_bundle")
OUT = []
def p(*a):
    line = " ".join(str(x) for x in a)
    OUT.append(line); print(line)

def section(t): p("\n" + "=" * 72); p(t); p("=" * 72)

# ---------------------------------------------------------------- environment
section("ENVIRONMENT")
p("python", sys.version.split()[0])
for mod in ("torch", "transformers", "TTS", "soundfile", "numpy"):
    try:
        m = __import__(mod)
        p(f"{mod:14s} {getattr(m, '__version__', '?')}")
    except Exception as e:
        p(f"{mod:14s} MISSING ({e.__class__.__name__})")
try:
    import torch
    p("cuda", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
except Exception as e:
    p("cuda probe failed:", e)

# ------------------------------------------------------------- what was asked
section("REQUEST AND PARAMETERS")
req = json.loads((BUNDLE / "request" / "synthesis_request.json").read_text())
p("job", req["job_id"], "| language", req["language"],
  "| request output_sample_rate", req["output_sample_rate"])

res_path = BUNDLE / "output" / "synthesis_result.json"
res = json.loads(res_path.read_text()) if res_path.exists() else {}
p("result sample_rate", res.get("sample_rate"), "| model", res.get("model_id"))
p("params actually used:", json.dumps(res.get("params", {}), ensure_ascii=False))

# What the model itself thinks, which is the check the worker never makes.
try:
    from colab.xtts_worker import XTTSWorker
    p("worker CONDITIONING_PARAMS:", XTTSWorker.__module__ and
      __import__("colab.xtts_worker", fromlist=["x"]).CONDITIONING_PARAMS)
    p("worker INFERENCE_PARAMS:   ",
      __import__("colab.xtts_worker", fromlist=["x"]).INFERENCE_PARAMS)
except Exception as e:
    p("worker constants unreadable:", e)

try:
    import TTS, torch
    from TTS.api import TTS as TTSApi
    # Only inspect if a model is already loaded in this kernel.
    mdl = None
    for v in list(globals().values()):
        if v.__class__.__name__ == "TTS" and hasattr(v, "synthesizer"):
            mdl = v; break
    if mdl is not None:
        x = mdl.synthesizer.tts_model
        cfg = x.config
        p("model output_sample_rate", getattr(cfg.audio, "output_sample_rate", "?"))
        p("model languages", getattr(cfg, "languages", "?"))
        p("language 'hi' supported:", "hi" in (getattr(cfg, "languages", []) or []))
    else:
        p("no loaded TTS object in this kernel; skipping model config")
except Exception as e:
    p("model config unreadable:", e)

# ----------------------------------------------------------------- audio math
def stats(path):
    with wave.open(str(path), "rb") as h:
        ch, width, rate, n = (h.getnchannels(), h.getsampwidth(),
                              h.getframerate(), h.getnframes())
        raw = h.readframes(n)
    dt = {1: np.int8, 2: np.int16, 4: np.int32}.get(width)
    if dt is None:
        return {"error": f"sample width {width}"}
    a = np.frombuffer(raw, dtype=dt).astype(np.float32) / float(np.iinfo(dt).max)
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if a.size == 0:
        return {"error": "empty"}
    peak = float(np.abs(a).max())
    rms = float(np.sqrt((a ** 2).mean()))
    thr = max(peak * 0.02, 1e-4)
    loud = np.abs(a) > thr
    lead = int(np.argmax(loud)) / rate if loud.any() else 0.0
    tail = (a.size - 1 - int(np.argmax(loud[::-1]))) / rate if loud.any() else 0.0
    # crude loop detector: correlate the second half against the first
    half = a.size // 2
    loop = 0.0
    if half > rate // 2:
        x, y = a[:half], a[half:2 * half]
        d = (np.linalg.norm(x) * np.linalg.norm(y))
        loop = float(abs(np.dot(x, y)) / d) if d > 0 else 0.0
    return {
        "rate": rate, "width_bytes": width, "channels": ch,
        "dur": a.size / rate, "peak": peak, "rms": rms,
        "clipped_pct": 100.0 * float((np.abs(a) > 0.995).mean()),
        "silent_pct": 100.0 * float((~loud).mean()),
        "lead_sil": lead, "trail_sil": (a.size / rate) - tail,
        "dc": float(a.mean()), "halves_corr": loop,
    }

section("REFERENCE AUDIO (what the voice was cloned from)")
ref = BUNDLE / "request" / "reference.wav"
s = stats(ref)
p(json.dumps(s, indent=2))
if "dur" in s:
    if s["dur"] < 6: p("!! reference under 6s; XTTS cloning degrades here")
    if s["rate"] < 22050: p(f"!! reference at {s['rate']} Hz; XTTS conditions at 22050+")
    if s["silent_pct"] > 40: p("!! reference is mostly silence")

section("PER SEGMENT")
by_id = {s["segment_id"]: s for s in res.get("segments", [])}
def devan(t):
    base = sum(1 for c in t if unicodedata.category(c) not in {"Mn", "Mc", "Zs", "Po"})
    return base

p(f"{'id':>3} {'slot':>6} {'gen':>7} {'ratio':>6} {'cps':>6} {'sim':>6} "
  f"{'tok':>5} {'peak':>5} {'rms':>6} {'sil%':>5} {'lead':>5} {'trail':>5} "
  f"{'corr':>5} status")
for seg in req["segments"]:
    i = seg["segment_id"]
    slot = seg["end_ts"] - seg["start_ts"]
    r = by_id.get(i, {})
    dur = r.get("duration") or 0.0
    ap = r.get("audio_path") or ""
    st = {}
    if ap:
        f = BUNDLE / ap
        if f.exists():
            st = stats(f)
    cps = (len(seg["text"]) / dur) if dur else 0.0
    fmt = lambda v, w, d: (f"{v:>{w}.{d}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}")
    p(f"{i:>3} {slot:>6.2f} {dur:>7.2f} {(dur/slot if slot else 0):>6.2f} "
      f"{cps:>6.1f} {fmt(r.get('speaker_similarity'),6,3)} "
      f"{fmt(r.get('gpt_tokens'),5,0)} {fmt(st.get('peak'),5,2)} "
      f"{fmt(st.get('rms'),6,4)} {fmt(st.get('silent_pct'),5,1)} "
      f"{fmt(st.get('lead_sil'),5,2)} {fmt(st.get('trail_sil'),5,2)} "
      f"{fmt(st.get('halves_corr'),5,2)} {r.get('status','MISSING')}")

section("TEXT BEING SPOKEN")
for seg in req["segments"]:
    t = seg["text"]
    nonhi = "".join(sorted({c for c in t if c.isalpha() and not (0x900 <= ord(c) <= 0x97F)}))
    p(f"[{seg['segment_id']}] {len(t)} chars, {devan(t)} base | non-Devanagari letters: {nonhi or 'none'}")
    p(f"     {t}")

section("QC REPORT")
try:
    for name in [n for n in list(sys.modules) if n.startswith("src.eval")]:
        del sys.modules[name]
    from src.eval.harness import build_report, render_report
    p(render_report(build_report(job_dir=BUNDLE, bundle_dir=BUNDLE)))
except Exception as e:
    import traceback; p("harness failed:", e); p(traceback.format_exc())

section("ERROR LOG")
el = BUNDLE / "logs" / "errors.log"
p(el.read_text()[-3000:] if el.exists() else "(no errors.log)")

section("ANSWER THESE IN WORDS")
p("""For each segment, one line. Which of these is it?
  wrong-voice   - intelligible Hindi, but not the speaker in reference.wav
  robotic       - right voice, flat or metallic, no natural prosody
  garbled       - not recognizable Hindi words
  mispronounced - Hindi words, wrong sounds
  looping       - repeats a word or phrase
  truncated     - cuts off before the sentence finishes
  too-fast      - intelligible but rushed
  fine          - no complaint
Also: is the reference.wav itself clean when you play it?""")

Path("/content/diagnostics.txt").write_text("\n".join(OUT), encoding="utf-8")
print("\n\n[saved to /content/diagnostics.txt]")
