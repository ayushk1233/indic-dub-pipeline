"""
Can IndicF5 clone this speaker in English, and does an English reference work
once its duration is corrected?

Two open questions, one run, because they share a reference clip and an answer.

The first is the one that has been chased through four mechanisms. With the
duration corrected and a single chunk, IndicF5 still opens a Hindi generation
with about three seconds of speech the speaker never said, and transcribed back
that speech reads as garbled English — the reference transcript, not the
sentence asked for. infer_batch_process hands the model `ref_text + gen_text`
as one sequence, conditions on the reference audio, and then strips exactly
`ref_audio_len` frames off the front with no alignment check behind the slice.
If the model cannot align the reference transcript to the reference audio, the
remainder is spoken at the start of the kept region. A vocabulary gap would
have explained that and was ruled out: the English transcript tokenizes at 100%
against IndicF5's own vocabulary, in which Latin is the largest script.

That leaves two candidates, and the English-to-English arm separates them.

  - If `en_en` is clean, the model aligns an English reference to English audio
    perfectly well, and what breaks is the switch from Latin to Devanagari
    inside one sequence. The next test transliterates the reference transcript
    and holds everything else fixed.
  - If `en_en` carries the same prefix, the model cannot align English audio to
    English text at all, and no arrangement of the Hindi side will fix it. An
    English reference would then need a transcript in Devanagari, or the
    reference has to be Hindi.

The second question is the one this project actually ships on, and it has been
unmeasured since the duration fix: content is not identity. Correcting the
duration changes what the model generates, so the 92% of scale that the broken
`en_ref_10s` arm scored cannot be carried over to the fixed one.

English-to-English is also the only arm in this project with no duration
confound at all. Latin reference text and Latin generated text make the byte
ratio 1:1, so IndicF5's formula is correct by construction — which makes
`got/natural` on that arm an independent check on the whole byte-ratio
explanation. If it comes back near 1.0 with no correction applied, the account
holds.

    import colab.indicf5_english as eng
    rows = eng.main()
    eng.listen(rows)
"""

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from colab import speaker_scale
from colab.english_report import cosine
from colab.indicf5_check import (
    FIXTURES,
    MAX_EXTRA,
    MAX_MISSING,
    NATURAL_CPS_HI,
    as_float_wave,
    load_indicf5,
    sentences,
    transcribe_outputs,
)
from colab.indicf5_diagnose import install_patches, _mode

OUT = Path("/content/indicf5_english")
REPORT = Path("/content/indicf5_english.txt")

SAMPLE_RATE = 24000
SEED = 0
NATURAL_CPS_EN = 13.13          # measured on FLEURS, n=394
MAX_LEAD = 0.05                 # a prefix worth more than 5% of the sentence

# name, reference clip, transcript key, generated language, speed, anchor, curve
#
# Every arm is single-chunk. Chunking was investigated and settled: it neither
# causes the babbling nor prevents it, and with the duration corrected nothing
# exceeds F5-TTS's 25s window, so there is no reason to split and no cross-fade
# seam to argue about.
#
# en_en takes no speed correction on purpose. Latin reference text against
# Latin generated text is the one case the byte formula gets right unaided, and
# leaving it alone turns that arm into a test of the explanation.
ARMS = [
    ("en_en", "english_reference_short.wav", "english_short", "en", None,  "en", "same"),
    ("en_hi", "english_reference_short.wav", "english_short", "hi", "auto", "en", "cross"),
    ("hi_hi", "hindi_reference_short.wav",   "hindi_short",   "hi", None,   "hi", "same"),
]

NATURAL = {"en": NATURAL_CPS_EN, "hi": NATURAL_CPS_HI}

_lines = []


def p(text=""):
    print(text)
    _lines.append(text)


def section(title):
    p("")
    p("=" * 76)
    p(title)
    p("=" * 76)


def mean(rows, key):
    values = [r.get(key, np.nan) for r in rows]
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def main(sentence_count=7):
    needed = ["english_speech.wav", "hindi_speech.wav", "scripted_text.json",
              "reference_text.json", "english_reference_short.wav",
              "hindi_reference_short.wav"]
    gone = [n for n in needed if not (FIXTURES / n).exists()]
    if gone:
        p(f"!! missing fixtures: {gone}")
        return []

    OUT.mkdir(parents=True, exist_ok=True)
    scripted = json.loads((FIXTURES / "scripted_text.json").read_text(encoding="utf-8"))
    transcripts = json.loads((FIXTURES / "reference_text.json").read_text(encoding="utf-8"))

    lines = {"en": sentences(scripted["en"])[:sentence_count],
             "hi": sentences(scripted["hi"])[:sentence_count]}

    section("SETUP")
    p(f"{len(lines['en'])} English and {len(lines['hi'])} Hindi sentences, "
      f"{len(ARMS)} arms, single chunk throughout")

    from colab.xtts_worker import XTTSWorker

    worker = XTTSWorker(Path("/content/unused_bundle"))
    worker.load_model()
    xtts = worker.xtts

    section("CALIBRATION")
    scale = speaker_scale.build(xtts, FIXTURES, OUT / "_pieces")
    p(f"  floor  {scale.floor:.3f}")
    p("")
    p(f"  {'clip length':>13}{'same-language':>16}{'cross-language':>16}")
    for (d, same), (_, cross) in zip(scale.curves["same"], scale.curves["cross"]):
        p(f"  {d:>11.1f}s{same:>16.3f}{cross:>16.3f}")

    install_patches()
    model = load_indicf5()

    rows = []
    for arm, filename, key, language, speed, anchor_name, curve in ARMS:
        transcript = transcripts[key]["text"]
        anchor = scale.anchors[anchor_name]
        p(f"\n--- {arm}   {filename} -> {language}"
          + ("   duration corrected" if speed else "   duration untouched"))

        for index, sentence in enumerate(lines[language]):
            _mode["speed"] = speed
            _mode["one_chunk"] = True

            torch.manual_seed(SEED)
            started = time.perf_counter()
            try:
                audio = model(sentence,
                              ref_audio_path=str(FIXTURES / filename),
                              ref_text=transcript)
            except Exception as exc:
                p(f"    [{index}] FAILED {type(exc).__name__}: {exc}")
                continue

            wave = as_float_wave(audio)
            path = OUT / arm / f"{index:02d}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(path), wave, SAMPLE_RATE, subtype="PCM_16")

            duration = len(wave) / SAMPLE_RATE
            with torch.no_grad():
                _, embedding = xtts.get_conditioning_latents(
                    audio_path=[str(path)], gpt_cond_len=30,
                    gpt_cond_chunk_len=4, max_ref_length=30,
                    sound_norm_refs=False)
            similarity = cosine(anchor, embedding)

            rows.append({
                "arm": arm, "index": index, "text": sentence, "path": path,
                "language": language, "curve": curve,
                "actual_s": duration,
                "natural_s": len(sentence) / NATURAL[language],
                "sim": similarity,
                "pos": scale.position(similarity, duration, curve),
                "wall_s": time.perf_counter() - started,
            })
            p(f"    [{index}] {duration:5.2f}s  "
              f"{duration / rows[-1]['natural_s']:.2f}x natural  "
              f"sim {similarity:.3f}  {rows[-1]['pos']:.0f}% of scale")

    if not rows:
        p("\n!! nothing generated")
        return rows

    del model
    torch.cuda.empty_cache()
    transcribe_outputs(rows)

    section("RESULT")
    p(f"{'arm':<8}{'n':>3}{'sim':>8}{'se':>7}{'position':>10}{'got/nat':>9}"
      f"{'extra':>8}{'lead':>7}{'lead_s':>8}{'cer':>7}")
    p("")
    table = {}
    for arm, _, _, language, _, _, _ in ARMS:
        items = [r for r in rows if r["arm"] == arm]
        if not items:
            continue
        sims = [r["sim"] for r in items]
        se = float(np.std(sims, ddof=1) / np.sqrt(len(sims))) if len(sims) > 1 else 0.0
        # A prefix in characters becomes a prefix in seconds at the rate the
        # rest of the clip is running, which is what a listener actually hears.
        lead_s = float(np.mean([r.get("lead_chars", 0) / NATURAL[language]
                                for r in items]))
        table[arm] = {"sim": float(np.mean(sims)), "se": se,
                      "pos": mean(items, "pos"), "lead": mean(items, "lead"),
                      "lead_s": lead_s, "extra": mean(items, "extra")}
        p(f"{arm:<8}{len(items):>3}{table[arm]['sim']:>8.3f}{se:>7.3f}"
          f"{table[arm]['pos']:>9.0f}%"
          f"{float(np.mean([r['actual_s'] / r['natural_s'] for r in items])):>9.2f}"
          f"{table[arm]['extra']:>8.3f}{table[arm]['lead']:>7.2f}"
          f"{lead_s:>7.1f}s{mean(items, 'cer'):>7.3f}")

    section("THE PREFIX")
    p("  lead is speech before the sentence begins, as a fraction of the")
    p("  sentence. It is measured separately from extra because it is")
    p("  contiguous and positional: three seconds at the front of a twelve")
    p("  second clip scores 0.12 on a total insertion rate and passes.")
    p("")
    for arm in table:
        worst = max((r for r in rows if r["arm"] == arm),
                    key=lambda r: r.get("lead_chars", 0), default=None)
        if worst is None:
            continue
        chars = worst.get("lead_chars", 0)
        p(f"  {arm:<8} mean {table[arm]['lead_s']:.1f}s, worst clip "
          f"[{worst['index']}] {chars} chars")
        if chars and worst.get("heard"):
            p(f"           spoke: {worst['heard'][:chars + 10]!r}")

    section("VERDICT")
    en_en = table.get("en_en", {})
    en_hi = table.get("en_hi", {})
    hi_hi = table.get("hi_hi", {})

    p("  Question 1 — where does the prefix come from?")
    if en_en and en_en["lead"] <= MAX_LEAD:
        p("    en_en is clean. The model aligns an English reference to English")
        p("    audio without trouble, so what breaks is the switch from Latin to")
        p("    Devanagari inside one sequence — not the English reference. Next")
        p("    test: the same English clip with its transcript transliterated")
        p("    into Devanagari, which moves the script and nothing else.")
    elif en_en:
        p("    en_en carries the prefix too. The model cannot align this English")
        p("    audio to its English transcript at all, so no arrangement of the")
        p("    Hindi side fixes it. An English reference needs a transcript the")
        p("    model can align, or the reference has to be Hindi.")

    p("")
    p("  Question 2 — does IndicF5 clone this speaker in English?")
    if en_en:
        p(f"    {en_en['sim']:.3f}, {en_en['pos']:.0f}% of scale. XTTS-v2 scored")
        p("    0.502 and 47% on the same speaker, and sounded heavily accented.")
        p("    Identity is not accent: listen before reading this as a win.")

    p("")
    p("  Question 3 — what does the corrected English arm actually score?")
    if en_hi and hi_hi:
        gap = en_hi["sim"] - hi_hi["sim"]
        combined = float(np.hypot(en_hi["se"], hi_hi["se"]))
        p(f"    en_hi {en_hi['sim']:.3f} ({en_hi['pos']:.0f}%) against "
          f"hi_hi {hi_hi['sim']:.3f} ({hi_hi['pos']:.0f}%)")
        p(f"    difference {gap:+.3f}, combined standard error {combined:.3f}"
          + (f", {abs(gap) / combined:.1f} standard errors" if combined else ""))
        p("    The 92% the broken en_ref_10s arm scored does not carry over and")
        p("    is not quoted here. This is the first measurement of identity on")
        p("    the corrected configuration.")

    p("")
    p("  Cross-check on the byte-ratio account: en_en takes no correction, and")
    p("  Latin against Latin is the one case the formula gets right unaided.")
    if en_en:
        got = float(np.mean([r["actual_s"] / r["natural_s"]
                             for r in rows if r["arm"] == "en_en"]))
        p(f"    en_en ran at {got:.2f}x natural English with speed untouched."
          + ("  The account holds." if 0.85 <= got <= 1.2
             else "  That is off, and the account needs revisiting."))

    REPORT.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nwrote {REPORT}")
    return rows


def listen(rows, arms=None, indexes=None):
    from IPython.display import Audio, display

    order = [a for a, *_ in ARMS]
    picked = [r for r in rows
              if (arms is None or r["arm"] in arms)
              and (indexes is None or r["index"] in indexes)]

    for index in sorted({r["index"] for r in picked}):
        group = sorted([r for r in picked if r["index"] == index],
                       key=lambda r: order.index(r["arm"]))
        for row in group:
            notes = []
            if row.get("lead", 0) > MAX_LEAD:
                notes.append(f"PREFIX {row.get('lead_chars', 0)} chars")
            if row.get("extra", 0) > MAX_EXTRA:
                notes.append("GIBBERISH")
            if row.get("missing", 0) > MAX_MISSING:
                notes.append("CUT SHORT")
            print(f"\n{row['arm']}  [{index}]  {row['actual_s']:.2f}s  "
                  f"{row['actual_s'] / row['natural_s']:.2f}x natural  "
                  f"sim {row['sim']:.3f}  {row['pos']:.0f}%"
                  + ("   " + "  ".join(notes) if notes else ""))
            print(f"  asked: {row['text']}")
            if row.get("heard") and row.get("cer", 0) > 0.05:
                print(f"  heard: {row['heard']}")
            display(Audio(str(row["path"])))
