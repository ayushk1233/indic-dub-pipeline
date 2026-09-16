"""
Four arms, one speaker, one session: what is actually broken.

Every result so far has confounded three things — a noisy reference, a hard
cross-lingual task, and whatever XTTS-v2 does badly on its own. This separates
them, because the same person now exists on tape in both languages:

  en -> en   XTTS's strongest language, cloning from clean English. If this
             fails nothing else matters.
  hi -> hi   Monolingual Hindi, the setup that sounded good in the desktop
             app. The model's Hindi ceiling.
  en -> hi   Production. English voice, Hindi words.
  hi -> en   The reverse, which isolates whether the loss travels one way.

The measurement that makes the rest legible is none of those. It is the real
Hindi recording scored against the anchor built from the real English one: the
same human, two languages, no synthesis anywhere. Speaker embeddings shift
across languages even for a real person, so that number is the honest ceiling
for `en -> hi`, and nobody has ever measured it here. Scoring cross-lingual
synthesis against a same-language ceiling, which is what every earlier run
did, charges the model for a gap the metric creates by itself.

Nothing here touches a bundle. The fixtures ship in the repo, so this runs in
a fresh runtime with no upload.
"""

import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from colab.english_report import cosine, measure


REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "fixtures"
OUT = Path("/content/four_arm")

# Held at the values Coqui shipped in XTTS-v2's own config.json. The earlier
# sweep of seven conditioning settings spanned 0.046 on a scale that is now
# known to span roughly 0.80, so conditioning is not a variable worth moving
# here. Language is the variable under test.
CONDITIONING = {
    "gpt_cond_len": 30,
    "gpt_cond_chunk_len": 4,
    "max_ref_length": 30,
    "sound_norm_refs": False,
}

# One anchor per language, computed from the full take rather than the
# 25-second reference, so the identity target is as rich as the recording
# allows and is never the same audio a clone was conditioned on.
ANCHOR = {
    "gpt_cond_len": 60,
    "gpt_cond_chunk_len": 30,
    "max_ref_length": 60,
    "sound_norm_refs": False,
}

DECODES = {
    "sampled": {"do_sample": True, "temperature": 0.75,
                "repetition_penalty": 5.0, "enable_text_splitting": False},
    "greedy": {"do_sample": False, "repetition_penalty": 5.0,
               "enable_text_splitting": False},
}

SENTENCES_PER_ARM = 3

# Measured from FLEURS, n=394 English and n=239 Hindi.
NATURAL_CPS = {"en": 13.13, "hi": 10.81}

# Devanagari danda as well as ASCII terminators.
_SENTENCE_RE = re.compile(r"(?<=[.!?।])\s+")

LINES: list[str] = []


def p(*args):
    line = " ".join(str(a) for a in args)
    LINES.append(line)
    print(line)


def section(title):
    p("\n" + "=" * 76)
    p(title)
    p("=" * 76)


def sentences(text, limit=SENTENCES_PER_ARM, minimum=25):
    parts = [s.strip() for s in _SENTENCE_RE.split(text or "") if len(s.strip()) >= minimum]
    return parts[:limit]


def fmt(value, width=7, places=3):
    if isinstance(value, (int, float)) and np.isfinite(value):
        return f"{value:>{width}.{places}f}"
    return f"{'-':>{width}}"


def main():
    missing = [n for n in ("english_speech.wav", "hindi_speech.wav",
                           "english_reference.wav", "hindi_reference.wav")
               if not (FIXTURES / n).exists()]

    if missing:
        p(f"!! missing fixtures: {missing}")
        p(f"   Run `python -m scripts.build_fixtures` locally, commit, push, "
          f"then `git pull` here.")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    scripted = json.loads((FIXTURES / "scripted_text.json").read_text(encoding="utf-8"))
    text = {"en": sentences(scripted["en"]), "hi": sentences(scripted["hi"])}

    section("SETUP")
    for module in ("torch", "TTS", "librosa", "soundfile"):
        try:
            p(f"{module:12s} {getattr(__import__(module), '__version__', '?')}")
        except Exception as exc:
            p(f"{module:12s} MISSING ({type(exc).__name__})")
    p(f"cuda         {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
    p(f"\n{len(text['en'])} English sentences, {len(text['hi'])} Hindi sentences, "
      f"{len(DECODES)} decoders, 4 arms")

    from colab.xtts_worker import XTTSWorker

    # The worker owns the licence agreement, the preflight checks and the
    # loader. Only the model is wanted here, so it is handed a bundle path it
    # will never read; load_bundle() is deliberately not called.
    worker = XTTSWorker(Path("/content/unused_bundle"))
    worker.load_model()
    xtts = worker.xtts

    languages = getattr(xtts.config, "languages", []) or []
    p(f"model languages: {len(languages)}; 'hi' supported: {'hi' in languages}")

    # ------------------------------------------------------------- anchors
    def latents(path, **params):
        with torch.no_grad():
            return xtts.get_conditioning_latents(audio_path=[str(path)], **params)

    _, anchor_en = latents(FIXTURES / "english_speech.wav", **ANCHOR)
    _, anchor_hi = latents(FIXTURES / "hindi_speech.wav", **ANCHOR)

    cond = {
        "english": latents(FIXTURES / "english_reference.wav", **CONDITIONING),
        "hindi": latents(FIXTURES / "hindi_reference.wav", **CONDITIONING),
    }

    section("CALIBRATION — the three numbers everything else is read against")

    def pieces_of(path, count, label, anchor, scores):
        audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        edges = np.linspace(0, audio.size, count + 1).astype(int)
        for index in range(count):
            piece = OUT / f"{label}_{index}.wav"
            sf.write(str(piece), audio[edges[index]:edges[index + 1]], rate, subtype="PCM_16")
            try:
                _, embedding = latents(piece)
                scores.append(cosine(anchor, embedding))
            except Exception as exc:
                p(f"  {label} {index} failed: {exc}")

    same = []
    pieces_of(FIXTURES / "english_speech.wav", 3, "en_third", anchor_en, same)
    p(f"  SAME LANGUAGE CEILING   you in English vs your English anchor")
    p(f"     {'  '.join(f'{s:.3f}' for s in same)}   mean {np.mean(same):.3f}")

    cross = [cosine(anchor_en, anchor_hi)]
    pieces_of(FIXTURES / "hindi_speech.wav", 3, "hi_third", anchor_en, cross)
    p(f"\n  CROSS LANGUAGE CEILING  you in Hindi vs your English anchor")
    p(f"     {'  '.join(f'{s:.3f}' for s in cross)}   mean {np.mean(cross):.3f}")
    p(f"     No synthesis involved. This is the most any en->hi clone can score.")

    floor_scores = []
    try:
        for name, entry in xtts.speaker_manager.speakers.items():
            floor_scores.append(cosine(anchor_en, entry["speaker_embedding"]))
    except Exception as exc:
        p(f"  floor unavailable: {type(exc).__name__}: {exc}")

    floor = float(np.median(floor_scores)) if floor_scores else 0.0
    ceiling_same = float(np.mean(same)) if same else float("nan")
    ceiling_cross = float(np.mean(cross)) if cross else float("nan")

    p(f"\n  FLOOR                   {len(floor_scores)} different real speakers")
    p(f"     median {floor:.3f}   max {max(floor_scores):.3f}" if floor_scores else "     -")

    def position(score, ceiling):
        span = ceiling - floor
        if not np.isfinite(score) or not np.isfinite(span) or span <= 0:
            return float("nan")
        return 100.0 * (score - floor) / span

    p("")
    p(f"  scale for en->en, hi->hi, hi->en : {floor:.3f} .. {ceiling_same:.3f}")
    p(f"  scale for en->hi                 : {floor:.3f} .. {ceiling_cross:.3f}")

    # ------------------------------------------------- what the takes sound like
    section("THE REAL RECORDINGS")
    p(f"{'take':<24}{'dur':>8}{'sil%':>7}{'paus':>6}{'F0':>8}{'stSD':>7}"
      f"{'stRNG':>7}{'cent':>7}{'cps':>7}")

    real = {}
    for label, name, lang in (("english (you)", "english_speech.wav", "en"),
                              ("hindi (you)", "hindi_speech.wav", "hi")):
        row = measure(FIXTURES / name)
        row["cps"] = len(scripted[lang]) / row["dur"] if row.get("dur") else 0.0
        real[lang] = row
        p(f"{label:<24}{fmt(row.get('dur'),8,2)}{fmt(row.get('silence_pct'),7,1)}"
          f"{fmt(row.get('num_pauses'),6,0)}{fmt(row.get('f0_median_hz'),8,1)}"
          f"{fmt(row.get('st_std'),7,2)}{fmt(row.get('st_p5_p95'),7,2)}"
          f"{fmt(row.get('centroid_hz'),7,0)}{fmt(row['cps'],7,1)}")

    p(f"\n  Your measured rate vs FLEURS: "
      f"en {real['en']['cps']:.1f} against {NATURAL_CPS['en']}, "
      f"hi {real['hi']['cps']:.1f} against {NATURAL_CPS['hi']}")
    p(f"  Your own English->Hindi expansion: "
      f"{real['hi']['dur'] / real['en']['dur']:.2f}x")

    # ---------------------------------------------------------------- the arms
    arms = [
        ("en_to_en", "english", "en", anchor_en, ceiling_same),
        ("hi_to_hi", "hindi", "hi", anchor_hi, ceiling_same),
        ("en_to_hi", "english", "hi", anchor_en, ceiling_cross),
        ("hi_to_en", "hindi", "en", anchor_hi, ceiling_same),
    ]

    section("SYNTHESIS")
    rows = []

    for arm, ref_name, lang, anchor, ceiling in arms:
        latent, embedding = cond[ref_name]
        p(f"\n--- {arm}   reference {ref_name}, speaking {lang}")

        for decode_name, decode in DECODES.items():
            directory = OUT / arm / decode_name
            directory.mkdir(parents=True, exist_ok=True)

            for index, sentence in enumerate(text[lang]):
                started = time.perf_counter()
                try:
                    with torch.no_grad():
                        result = xtts.inference(
                            text=sentence, language=lang,
                            gpt_cond_latent=latent, speaker_embedding=embedding,
                            **decode,
                        )
                except Exception as exc:
                    p(f"    [{decode_name} {index}] FAILED {type(exc).__name__}: {exc}")
                    continue

                wav = np.asarray(result["wav"], dtype=np.float32)
                path = directory / f"{index:02d}.wav"
                sf.write(str(path), wav, 24000, subtype="PCM_16")

                try:
                    _, produced = latents(path)
                    score = cosine(anchor, produced)
                except Exception:
                    score = float("nan")

                duration = wav.size / 24000
                rows.append({
                    "arm": arm, "decode": decode_name, "index": index,
                    "lang": lang, "text": sentence, "path": path,
                    "dur": duration, "sim": score, "ceiling": ceiling,
                    "cps": len(sentence) / duration if duration else 0.0,
                    "gen_s": time.perf_counter() - started,
                })
                p(f"    [{decode_name} {index}] {duration:6.2f}s  "
                  f"{rows[-1]['cps']:5.1f} cps  sim {score:.3f}  "
                  f"{position(score, ceiling):4.0f}% of scale")

    # ------------------------------------------------------------------ result
    section("RESULT")
    p(f"{'arm':<12}{'decode':<10}{'sim':>8}{'ceiling':>9}{'position':>10}"
      f"{'cps':>8}{'natural':>9}")

    summary = {}
    for arm, _, lang, _, ceiling in arms:
        for decode_name in DECODES:
            items = [r for r in rows if r["arm"] == arm and r["decode"] == decode_name]
            if not items:
                continue
            scores = [r["sim"] for r in items if np.isfinite(r["sim"])]
            mean = float(np.mean(scores)) if scores else float("nan")
            cps = float(np.mean([r["cps"] for r in items]))
            summary[(arm, decode_name)] = mean
            p(f"{arm:<12}{decode_name:<10}{fmt(mean)}{fmt(ceiling)}"
              f"{fmt(position(mean, ceiling), 9, 0)}%{cps:>8.1f}"
              f"{cps / NATURAL_CPS[lang]:>8.2f}x")

    best = {arm: max((summary.get((arm, d), float('nan')) for d in DECODES),
                     key=lambda v: (np.isfinite(v), v))
            for arm, _, _, _, _ in arms}

    section("VERDICT")

    pos_en_en = position(best["en_to_en"], ceiling_same)
    pos_hi_hi = position(best["hi_to_hi"], ceiling_same)
    pos_en_hi = position(best["en_to_hi"], ceiling_cross)

    p(f"  en->en {pos_en_en:5.0f}% of scale")
    p(f"  hi->hi {pos_hi_hi:5.0f}%")
    p(f"  en->hi {pos_en_hi:5.0f}%  (against the cross-language ceiling)")
    p("")

    if pos_en_en < 45:
        p("  XTTS-v2 cannot clone this voice even in its strongest language,")
        p("  from a 31 dB SNR reference. The reference is no longer the")
        p("  explanation. Fine-tuning or a different model.")
    elif pos_en_hi >= 70:
        p("  Cross-lingual synthesis is close to what the metric can see.")
        p("  Identity is not the remaining problem; judge it by ear for")
        p("  naturalness and compare against IndicF5.")
    elif pos_hi_hi - pos_en_hi >= 20:
        p("  Monolingual Hindi clones well, cross-lingual does not. That is")
        p("  the cross-lingual gap, isolated. The fix is voice conversion")
        p("  onto a Hindi donor, not fine-tuning.")
    else:
        p("  Both Hindi arms sit well below the English arm. XTTS-v2's Hindi")
        p("  is the ceiling, independent of the reference. Test IndicF5 next.")

    p("\n  Accent is a separate axis this metric cannot see. If the English")
    p("  still comes back American, only fine-tuning changes that.")

    report = Path("/content/four_arm_report.txt")
    report.write_text("\n".join(LINES), encoding="utf-8")
    print(f"\n[saved to {report}]")

    return rows


def listen(rows=None):
    """
    Play the real takes, then every arm, grouped so the comparison is direct.
    """
    from IPython.display import Audio, HTML, display

    display(HTML("<h2>The real recordings</h2>"))
    for label, name in (("You, English", "english_speech.wav"),
                        ("You, Hindi", "hindi_speech.wav")):
        display(HTML(f"<b>{label}</b>"))
        display(Audio(filename=str(FIXTURES / name)))

    # Re-running only the listening cell after a fresh kernel is normal, so
    # rebuild the index from disk when the caller has no rows in hand.
    if not rows:
        rows = []
        for arm_dir in sorted(d for d in OUT.iterdir() if d.is_dir()):
            for decode_dir in sorted(d for d in arm_dir.iterdir() if d.is_dir()):
                for path in sorted(decode_dir.glob("*.wav")):
                    rows.append({"arm": arm_dir.name, "decode": decode_dir.name,
                                 "index": int(path.stem), "path": path,
                                 "text": "", "dur": 0.0, "sim": float("nan")})

    for arm in ("en_to_en", "hi_to_hi", "en_to_hi", "hi_to_en"):
        items = [r for r in rows if r["arm"] == arm]
        if not items:
            continue
        display(HTML(f"<h2 style='margin-top:24px'>{arm}</h2>"))
        for index in sorted({r["index"] for r in items}):
            same_index = [r for r in items if r["index"] == index]
            display(HTML(f"<div style='margin-top:12px;font-size:15px'>"
                         f"{same_index[0].get('text') or f'sentence {index}'}</div>"))
            for row in same_index:
                label = row["decode"]
                if np.isfinite(row.get("sim", float("nan"))):
                    label += f" — {row['dur']:.2f}s, sim {row['sim']:.3f}"
                display(HTML(f"<div style='color:#666;margin-top:6px'>{label}</div>"))
                display(Audio(filename=str(row["path"])))


if __name__ == "__main__":
    main()
