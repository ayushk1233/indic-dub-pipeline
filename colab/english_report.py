"""
A technical report on the English clone, on a calibrated scale.

Every speaker-similarity number this project has produced so far has been
uninterpretable, because nothing established what the metric's range means.
0.50 is only bad if genuine same-speaker audio scores much higher and a
different speaker scores much lower. This measures both ends first:

  ceiling  - the real reference scored against itself, in halves and thirds.
             Same person, same recording, same encoder. Nothing synthetic can
             beat this, and it is almost certainly not 1.0.
  floor    - XTTS's own 58 studio speakers scored against the anchor. Genuine
             different people, same encoder. This is what "wrong voice" costs.

With those two, English and Hindi synthesis land somewhere on a real scale
instead of floating free.

The second half measures prosody rather than identity, because "robotic" is
the complaint and identity metrics cannot see it. Pitch variation in
semitones is the number that matters: a monotone read has a small one no
matter how accurate its timbre.

Read-only apart from the report file. Synthesizes nothing. Run
colab/english_check.py first so there is English audio to measure.
"""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


BUNDLE = Path("/content/tts_bundle")
ENGLISH = Path("/content/english_check")
REPORT = Path("/content/english_report.txt")

ANALYSIS_SR = 22050

# Measured from FLEURS, n=394 English and n=239 Hindi.
NATURAL_CPS = {"en": 13.13, "hi": 10.81}

# The conditioning the sweep picked, reused so the anchor here matches the
# anchor english_check.py scored against.
ANCHOR = {
    "gpt_cond_len": 60,
    "gpt_cond_chunk_len": 30,
    "max_ref_length": 60,
    "sound_norm_refs": True,
}

OUT: list[str] = []


def p(*args):
    line = " ".join(str(a) for a in args)
    OUT.append(line)
    print(line)


def section(title):
    p("\n" + "=" * 76)
    p(title)
    p("=" * 76)


# --------------------------------------------------------------- measurement

def load(path, sr=ANALYSIS_SR):
    import librosa

    audio, _ = librosa.load(str(path), sr=sr, mono=True)
    return audio


def pitch(audio, sr=ANALYSIS_SR):
    """
    Voiced F0 in semitones relative to 100 Hz.

    Semitones rather than Hz because pitch is perceived multiplicatively: a
    30 Hz swing is dramatic on a low voice and barely audible on a high one.
    Comparing a synthetic voice to a reference in Hz would mistake a pitch
    offset for a prosody difference.
    """
    import librosa

    try:
        f0, voiced, _ = librosa.pyin(
            audio, sr=sr, fmin=65.0, fmax=400.0,
            frame_length=1024, hop_length=256,
        )
    except Exception as exc:  # pyin is the one part here that can be fragile
        return {"error": f"{type(exc).__name__}: {exc}"}

    good = f0[np.isfinite(f0)]

    if good.size < 8:
        return {"voiced_pct": 0.0, "error": "too little voiced audio"}

    semitones = 12.0 * np.log2(good / 100.0)

    return {
        "voiced_pct": 100.0 * float(np.mean(voiced)) if voiced is not None else 0.0,
        "f0_median_hz": float(np.median(good)),
        "st_std": float(np.std(semitones)),
        "st_p5_p95": float(np.percentile(semitones, 95) - np.percentile(semitones, 5)),
    }


def spectral(audio, sr=ANALYSIS_SR):
    import librosa

    stft = np.abs(librosa.stft(audio, n_fft=1024, hop_length=256))

    return {
        "centroid_hz": float(np.mean(
            librosa.feature.spectral_centroid(S=stft, sr=sr)
        )),
        # Flatness separates tone from noise. Synthetic speech that sounds
        # metallic usually reads as too tonal, buzzy speech as too flat.
        "flatness": float(np.mean(librosa.feature.spectral_flatness(S=stft))),
        "rolloff_hz": float(np.mean(
            librosa.feature.spectral_rolloff(S=stft, sr=sr, roll_percent=0.95)
        )),
    }


def dynamics(audio, sr=ANALYSIS_SR):
    """
    Loudness contour and pause structure.

    A flat RMS contour with no pauses is the acoustic signature of a read
    that never breathes, which is half of what people mean by robotic.
    """
    import librosa

    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)[0]
    db = librosa.amplitude_to_db(np.maximum(rms, 1e-8))

    quiet = db < (db.max() - 35.0)
    hop_s = 256.0 / sr

    runs, current = [], 0
    for flag in quiet:
        if flag:
            current += 1
        elif current:
            runs.append(current * hop_s)
            current = 0
    if current:
        runs.append(current * hop_s)

    pauses = [r for r in runs if r >= 0.15]

    return {
        "rms_db_std": float(np.std(db[~quiet])) if (~quiet).any() else 0.0,
        "num_pauses": len(pauses),
        "pause_s": float(sum(pauses)),
        "silence_pct": 100.0 * float(np.mean(quiet)),
        "peak": float(np.abs(audio).max()) if audio.size else 0.0,
        "dur": audio.size / sr,
    }


def measure(path):
    audio = load(path)
    row = {"path": Path(path), "name": Path(path).name}
    row.update(dynamics(audio))
    row.update(spectral(audio))
    row.update(pitch(audio))
    return row


def cosine(a, b):
    a = a.reshape(1, -1).float()
    b = b.reshape(1, -1).float().to(a.device)
    return float(torch.nn.functional.cosine_similarity(a, b).item())


def fmt(value, width, places):
    if isinstance(value, (int, float)) and np.isfinite(value):
        return f"{value:>{width}.{places}f}"
    return f"{'-':>{width}}"


# -------------------------------------------------------------------- report

def main():
    if not BUNDLE.exists():
        p(f"!! no bundle at {BUNDLE}")
        return

    reference = BUNDLE / "request" / "reference.wav"

    section("WHAT IS BEING MEASURED")
    for module in ("librosa", "soundfile", "torch", "TTS"):
        try:
            p(f"{module:12s} {getattr(__import__(module), '__version__', '?')}")
        except Exception as exc:
            p(f"{module:12s} MISSING ({type(exc).__name__})")

    english_files = sorted(ENGLISH.rglob("en_*.wav")) if ENGLISH.exists() else []
    hindi_files = sorted((BUNDLE / "output").glob("seg_*.wav"))

    p(f"\nreference     {reference}")
    p(f"english       {len(english_files)} files under {ENGLISH}")
    p(f"hindi         {len(hindi_files)} files under {BUNDLE / 'output'}")

    if not english_files:
        p("\n!! No English audio. Run colab/english_check.py first, in this "
          "same runtime, then re-run this cell.")

    # ------------------------------------------------------ the speaker scale
    from colab.xtts_worker import XTTSWorker

    worker = XTTSWorker(BUNDLE)
    worker.load_bundle()
    worker.load_model()

    with torch.no_grad():
        _, anchor = worker.xtts.get_conditioning_latents(
            audio_path=[str(reference)], **ANCHOR
        )

    section("CALIBRATION — what this similarity metric can actually reach")

    audio, sr = sf.read(str(reference), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    parts = Path("/content/_ref_parts")
    parts.mkdir(parents=True, exist_ok=True)

    ceiling = []
    for label, count in (("half", 2), ("third", 3)):
        edges = np.linspace(0, audio.size, count + 1).astype(int)
        for index in range(count):
            piece = audio[edges[index]:edges[index + 1]]
            piece_path = parts / f"{label}_{index}.wav"
            sf.write(str(piece_path), piece, sr, subtype="PCM_16")
            try:
                with torch.no_grad():
                    _, embedding = worker.xtts.get_conditioning_latents(
                        audio_path=[str(piece_path)]
                    )
                score = cosine(anchor, embedding)
                ceiling.append(score)
                p(f"  real speaker, {label} {index}  "
                  f"({piece.size / sr:5.2f}s)   {score:.3f}")
            except Exception as exc:
                p(f"  real speaker, {label} {index}  failed: {exc}")

    floor = []
    try:
        speakers = worker.xtts.speaker_manager.speakers
        for name, entry in speakers.items():
            embedding = entry["speaker_embedding"]
            floor.append((name, cosine(anchor, embedding)))
    except Exception as exc:
        p(f"  impostor floor unavailable: {type(exc).__name__}: {exc}")

    if floor:
        values = sorted(s for _, s in floor)
        best = max(floor, key=lambda t: t[1])
        p(f"\n  {len(floor)} different real speakers (XTTS studio voices)")
        p(f"  min {values[0]:.3f}   median {np.median(values):.3f}   "
          f"max {values[-1]:.3f}  ({best[0]})")

    ceiling_mean = float(np.mean(ceiling)) if ceiling else float("nan")
    floor_median = float(np.median([s for _, s in floor])) if floor else float("nan")

    p("")
    p(f"  CEILING (same person, real audio)     {ceiling_mean:.3f}")
    p(f"  FLOOR   (different people, real audio) {floor_median:.3f}")
    p("")
    p("  A synthetic score is only meaningful as a position between these.")
    p("  Anything at or below the floor is a different voice. Anything near")
    p("  the ceiling is as close as this metric can see.")

    def placed(score):
        if not np.isfinite(score) or not np.isfinite(ceiling_mean):
            return "-"
        span = ceiling_mean - floor_median
        if span <= 0:
            return "-"
        return f"{100.0 * (score - floor_median) / span:5.0f}%"

    # ------------------------------------------------------------- the audio
    section("ACOUSTIC MEASUREMENTS")

    groups = {"reference": [reference]}
    for path in english_files:
        groups.setdefault(f"en/{path.parent.name}", []).append(path)
    if hindi_files:
        groups["hi/bundle"] = hindi_files

    rows = []
    p(f"{'file':<26}{'dur':>7}{'peak':>6}{'sil%':>6}{'paus':>5}"
      f"{'rmsSD':>7}{'F0':>7}{'stSD':>7}{'stRNG':>7}"
      f"{'cent':>7}{'flat':>7}{'sim':>7}{'scale':>7}")

    for group, paths in groups.items():
        p(f"\n[{group}]")
        for path in paths:
            try:
                row = measure(path)
            except Exception as exc:
                p(f"  {path.name:<24} measurement failed: {exc}")
                continue

            row["group"] = group

            if path == reference:
                row["sim"] = ceiling_mean
            else:
                try:
                    with torch.no_grad():
                        _, embedding = worker.xtts.get_conditioning_latents(
                            audio_path=[str(path)]
                        )
                    row["sim"] = cosine(anchor, embedding)
                except Exception:
                    row["sim"] = float("nan")

            rows.append(row)

            p(f"  {row['name']:<24}"
              f"{fmt(row.get('dur'), 7, 2)}"
              f"{fmt(row.get('peak'), 6, 2)}"
              f"{fmt(row.get('silence_pct'), 6, 1)}"
              f"{fmt(row.get('num_pauses'), 5, 0)}"
              f"{fmt(row.get('rms_db_std'), 7, 1)}"
              f"{fmt(row.get('f0_median_hz'), 7, 1)}"
              f"{fmt(row.get('st_std'), 7, 2)}"
              f"{fmt(row.get('st_p5_p95'), 7, 2)}"
              f"{fmt(row.get('centroid_hz'), 7, 0)}"
              f"{fmt(row.get('flatness'), 7, 4)}"
              f"{fmt(row.get('sim'), 7, 3)}"
              f"{placed(row.get('sim', float('nan'))):>7}")

    # ------------------------------------------------------------- group view
    section("BY GROUP — the reference row is the target for every other row")

    p(f"{'group':<16}{'n':>3}{'sim':>8}{'on scale':>10}"
      f"{'stSD':>8}{'stRNG':>8}{'F0':>8}{'rmsSD':>8}{'cent':>8}{'sil%':>7}")

    def mean_of(items, key):
        values = [r[key] for r in items
                  if isinstance(r.get(key), (int, float)) and np.isfinite(r[key])]
        return float(np.mean(values)) if values else float("nan")

    summary = {}
    for group in groups:
        items = [r for r in rows if r["group"] == group]
        if not items:
            continue
        summary[group] = {k: mean_of(items, k) for k in
                          ("sim", "st_std", "st_p5_p95", "f0_median_hz",
                           "rms_db_std", "centroid_hz", "silence_pct")}
        s = summary[group]
        p(f"{group:<16}{len(items):>3}"
          f"{fmt(s['sim'], 8, 3)}{placed(s['sim']):>10}"
          f"{fmt(s['st_std'], 8, 2)}{fmt(s['st_p5_p95'], 8, 2)}"
          f"{fmt(s['f0_median_hz'], 8, 1)}{fmt(s['rms_db_std'], 8, 1)}"
          f"{fmt(s['centroid_hz'], 8, 0)}{fmt(s['silence_pct'], 7, 1)}")

    # ----------------------------------------------------------- what it says
    section("READING")

    ref = summary.get("reference")

    if ref:
        for group, s in summary.items():
            if group == "reference":
                continue
            p(f"\n{group}")

            if np.isfinite(s["sim"]):
                p(f"  identity   {s['sim']:.3f}, {placed(s['sim']).strip()} of the "
                  f"way from a different speaker to this one")

            if np.isfinite(s["st_std"]) and np.isfinite(ref["st_std"]) and ref["st_std"]:
                ratio = s["st_std"] / ref["st_std"]
                verdict = ("monotone against the reference" if ratio < 0.7
                           else "comparable pitch movement" if ratio < 1.3
                           else "more pitch movement than the reference")
                p(f"  prosody    {s['st_std']:.2f} semitones of pitch variation "
                  f"vs {ref['st_std']:.2f} ({ratio:.2f}x) — {verdict}")

            if np.isfinite(s["f0_median_hz"]) and np.isfinite(ref["f0_median_hz"]):
                delta = 12.0 * np.log2(s["f0_median_hz"] / ref["f0_median_hz"])
                if abs(delta) > 1.5:
                    p(f"  pitch      median F0 is {delta:+.1f} semitones off the "
                      f"reference, which alone reads as a different person")

            if np.isfinite(s["centroid_hz"]) and np.isfinite(ref["centroid_hz"]):
                ratio = s["centroid_hz"] / ref["centroid_hz"]
                if ratio > 1.25:
                    p(f"  timbre     {ratio:.2f}x brighter than the reference "
                      f"— the usual measurable correlate of metallic")
                elif ratio < 0.8:
                    p(f"  timbre     {ratio:.2f}x duller than the reference "
                      f"— muffled, losing high detail")

    section("PACE")
    for group, s in summary.items():
        if group == "reference":
            continue
        lang = "hi" if group.startswith("hi") else "en"
        items = [r for r in rows if r["group"] == group]
        speech = [r["dur"] * (1 - r["silence_pct"] / 100.0) for r in items]
        p(f"{group:<16} mean {np.mean([r['dur'] for r in items]):5.2f}s, "
          f"{np.mean(speech):5.2f}s of it speech, "
          f"natural {lang} is {NATURAL_CPS[lang]} cps")

    section("WHAT TO SAY IN WORDS")
    p("""For the English clips, answer these three separately. They fail
independently and the fix for each is different:

  1. IDENTITY   Does it sound like the same person as reference.wav?
                yes / close but not them / clearly someone else
  2. HUMANNESS  Would you believe a person said it, ignoring who?
                yes / uncanny / obviously synthetic
  3. ARTIFACTS  Buzzing, metallic ring, clicks, wrong stress, swallowed
                words, wrong pauses. Name what you hear.

Then one line comparing English to the Hindi you already heard: is the
English better, the same, or worse on each of the three?""")

    REPORT.write_text("\n".join(OUT), encoding="utf-8")
    print(f"\n\n[saved to {REPORT}]")


if __name__ == "__main__":
    main()
