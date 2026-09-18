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

That leaves two candidates: the model cannot align Latin reference text to
audio at all, or it cannot carry a script change across one sequence. `en_deva`
separates them. It is the same English clip, the same Hindi output and the same
corrected duration as `en_hi`, with only the reference transcript's script
moved — and the Devanagari transcript is not a transliteration of the English
text but Whisper's own Hindi-mode reading of the reference audio, so it
describes what is actually on the tape rather than what the English spelling
suggests. A transliteration would have introduced its own alignment error and
made a null result unreadable.

  - `en_deva` clean, `en_hi` not  -> the script inside the sequence is the
        problem, and an English reference works with a Devanagari transcript.
  - both carry the prefix          -> the model cannot align this English audio
        to any transcript, and the reference has to be Hindi.

IndicF5 declares eleven Indian languages and English is not among them, so
`en_en` is not load-bearing for that comparison: poor English would make its
prefix reading unreliable, which is exactly why the discriminator is `en_deva`
and not `en_en`. What `en_en` answers is a separate question worth its own arm.

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
from colab import workspace
from colab.english_report import cosine
from colab.indicf5_check import (
    ASR_DECODE,
    ASR_MODEL,
    _PUNCT,
    FIXTURES,
    MAX_CER,
    MAX_EXTRA,
    MAX_MISSING,
    NATURAL_CPS_HI,
    as_float_wave,
    load_indicf5,
    sentences,
    transcribe_outputs,
)
from colab.indicf5_diagnose import install_patches, _mode

OUT = workspace.out("indicf5_english")
REPORT = workspace.report("indicf5_english.txt")
PROBE_REPORT = workspace.report("indicf5_probe.txt")

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
    ("en_en",   "english_reference_short.wav", "english_short", "en", None,   "en", "same"),
    ("en_hi",   "english_reference_short.wav", "english_short", "hi", "auto", "en", "cross"),
    ("en_deva", "english_reference_short.wav", "@deva",         "hi", "auto", "en", "cross"),
    ("hi_hi",   "hindi_reference_short.wav",   "hindi_short",   "hi", None,   "hi", "same"),
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


DEVANAGARI = range(0x0900, 0x0980)


def devanagari_fraction(text):
    """
    How much of a string is actually in the script it is supposed to be in.

    Whisper's language argument is a hint, not a constraint. Asked to read ten
    seconds of English "in Hindi" it returned English in Roman script, and the
    arm built on that transcript was labelled `en_deva` and reported as a test
    of script while differing from its control only in punctuation. Nothing in
    the run could tell, because 132 Latin characters and 132 Devanagari
    characters look the same until you count the bytes.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(ord(c) in DEVANAGARI for c in letters) / len(letters)


def plain(text):
    """The same words with the punctuation and capitalisation taken off."""
    return " ".join(_PUNCT.sub(" ", (text or "").lower()).split())


def transliterate(text):
    """
    Latin to Devanagari with IndicXlit, from the same lab as IndicF5.

    Not sanscript/ITRANS, which reads its input as a transliteration scheme
    rather than as English and renders "tell you" as तेल्ल् योउ. A null result
    from that could not be told apart from a bad transliteration.
    """
    try:
        from ai4bharat.transliteration import XlitEngine
    except ImportError:
        return None

    engine = XlitEngine("hi", beam_width=4, src_script_type="en")
    out = engine.translit_sentence(plain(text), lang_code="hi")
    if isinstance(out, dict):
        out = out.get("hi", "")
    return (out or "").strip() or None


def hear(path, language):
    """
    One clip, one transcript. Whisper is loaded and dropped around it.

    Pinned decoding, like transcribe_outputs. This transcript is not a score —
    it is handed back to IndicF5 as the reference text, so a sampled decode
    would change the model's conditioning between two runs that are supposed to
    differ in one variable.
    """
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    asr = pipeline("automatic-speech-recognition", model=ASR_MODEL, device=device,
                   torch_dtype=torch.float16 if device == 0 else torch.float32)
    try:
        out = asr(str(path), generate_kwargs={"language": language,
                                              "task": "transcribe",
                                              **ASR_DECODE})
        return (out or {}).get("text", "").strip()
    finally:
        del asr
        torch.cuda.empty_cache()


def main(sentence_count=7):
    _lines.clear()
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

    # Whisper first and alone. The Devanagari reference transcript has to exist
    # before IndicF5 is asked to condition on it, and loading the two models
    # sequentially rather than together keeps a T4 comfortable.
    section("REFERENCE TRANSCRIPTS")
    deva = hear(FIXTURES / "english_reference_short.wav", "hi")
    transcripts["@deva"] = {"text": deva}
    latin = transcripts["english_short"]["text"]
    p(f"  latin      {len(latin):>4} chars, {len(latin.encode('utf-8')):>4} bytes")
    p(f"             {latin}")
    p(f"  devanagari {len(deva):>4} chars, {len(deva.encode('utf-8')):>4} bytes")
    p(f"             {deva}")
    p("")
    p("  The same ten seconds of English, read by Whisper in Hindi. Not a")
    p("  transliteration of the English spelling — a description of what is on")
    p("  the tape, in the script the model was trained on.")

    if not deva:
        p("  !! empty, so en_deva cannot run and the discriminator is lost")

    from colab.xtts_worker import XTTSWorker

    worker = XTTSWorker(workspace.out("unused_bundle"))
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
        looped = sum(1 for r in items if r.get("asr_looped"))
        table[arm] = {"sim": float(np.mean(sims)), "se": se,
                      "pos": mean(items, "pos"), "lead": mean(items, "lead"),
                      "lead_s": lead_s, "extra": mean(items, "extra"),
                      "looped": looped, "n": len(items)}
        p(f"{arm:<8}{len(items):>3}{table[arm]['sim']:>8.3f}{se:>7.3f}"
          f"{table[arm]['pos']:>9.0f}%"
          f"{float(np.mean([r['actual_s'] / r['natural_s'] for r in items])):>9.2f}"
          f"{table[arm]['extra']:>8.3f}{table[arm]['lead']:>7.2f}"
          f"{lead_s:>7.1f}s{mean(items, 'cer'):>7.3f}"
          + (f"   {looped}/{len(items)} unreadable" if looped else ""))

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

    deva = table.get("en_deva", {})
    p("  Question 1 — where does the prefix come from?")
    p("    en_hi and en_deva differ in one thing: the script of the reference")
    p("    transcript. Same clip, same sentences, same corrected duration.")
    if deva and en_hi:
        p(f"    en_hi   prefix {en_hi['lead_s']:.1f}s")
        p(f"    en_deva prefix {deva['lead_s']:.1f}s")
        if deva["lead"] <= MAX_LEAD < en_hi["lead"]:
            p("    The script inside the sequence is the problem. An English")
            p("    reference works, given a transcript in the script the model")
            p("    was trained on — and the pipeline's ASR already produces one.")
        elif deva["lead"] > MAX_LEAD and en_hi["lead"] > MAX_LEAD:
            p("    Both carry it, so this is not about script. The model cannot")
            p("    align this English audio to any transcript, and the reference")
            p("    has to be Hindi. Recording each speaker once in Hindi returns")
            p("    to the table as the shipping answer.")
        else:
            p("    Neither reading holds cleanly. Listen before concluding.")

    p("")
    p("  Question 2 — does IndicF5 clone this speaker in English?")
    if en_en and en_en.get("looped"):
        p(f"    {en_en['looped']} of {en_en['n']} clips produced audio Whisper")
        p("    could not parse — it looped instead, emitting more characters")
        p("    than the clip could physically contain. Those clips are not")
        p("    scored for content, and the identity figure below describes")
        p("    whatever sound was produced, not English speech. A high cosine")
        p("    on unintelligible audio is the failure this project already")
        p("    documented once: the encoder reads timbre and nothing else.")
    if en_en:
        p(f"    {en_en['sim']:.3f}, {en_en['pos']:.0f}% of scale. XTTS-v2 scored")
        p("    0.502 and 47% on the same speaker, and sounded heavily accented.")
        p("    Identity is not accent: listen before reading this as a win.")
        p("    English is not one of IndicF5's eleven declared languages, so a")
        p("    poor result here is a limit of the model rather than a fault to")
        p("    chase, and a good one is a bonus for code-mixed Hinglish input.")

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
            if row.get("asr_looped"):
                notes.append("UNREADABLE (ASR looped)")
            if row.get("lead", 0) > MAX_LEAD:
                notes.append(f"PREFIX {row.get('lead_chars', 0)} chars")
            if row.get("extra", 0) > MAX_EXTRA:
                notes.append("GIBBERISH")
            if row.get("missing", 0) > MAX_MISSING:
                notes.append("CUT SHORT")
            # Substituted gibberish is the one failure the positional checks
            # cannot see. "Seven were impossible, and we had to rewrite them."
            # came back as "Sraindari ansu alwe atcho rureshi chong." — right
            # length, right rhythm, no inserted or missing span anywhere, and
            # not one correct word. Only the error rate catches that.
            if row.get("cer", 0) > MAX_CER:
                notes.append("MANGLED")
            print(f"\n{row['arm']}  [{index}]  {row['actual_s']:.2f}s  "
                  f"{row['actual_s'] / row['natural_s']:.2f}x natural  "
                  f"sim {row['sim']:.3f}  {row['pos']:.0f}%"
                  + ("   " + "  ".join(notes) if notes else ""))
            print(f"  asked: {row['text']}")
            heard = row.get("heard") or ""
            if row.get("asr_looped"):
                print(f"  heard: {heard[:160]}...  [{len(heard)} chars from a "
                      f"{row['actual_s']:.1f}s clip]")
            elif heard and row.get("cer", 0) > 0.05:
                print(f"  heard: {heard}"
                      + (f"   [cer {row['cer']:.2f}]"
                         if row.get("cer", 0) > MAX_CER else ""))
            display(Audio(str(row["path"])))


def probe(sentence_count=7):
    """
    What about the reference transcript causes the prefix?

    Three variants of the same transcript for the same ten seconds of English
    audio, each one step from the last, so a difference can be attributed:

        en_latin   the transcript as the ASR stage produces it, with
                   punctuation and capitalisation. The baseline, and the
                   configuration that leaves a prefix on one clip in seven.
        en_plain   the same words, lower case, punctuation removed.
        en_deva    those words transliterated into Devanagari with IndicXlit.

    latin against plain isolates punctuation. plain against deva isolates
    script, with punctuation already gone from both, which is the comparison
    the first attempt at this probe believed it was making and was not:
    Whisper's language argument is a hint, and asked to read English "in Hindi"
    it returned English in Roman script. The arm was labelled deva, differed
    from its control only in punctuation, and the verdict named script. Nothing
    in the run could tell — 132 Latin characters and 132 Devanagari characters
    look identical until the bytes are counted, which is now done and printed.

    No XTTS and no calibration. The prefix is a content failure and identity
    has already been measured.

        rows = eng.probe()
        eng.listen(rows, indexes=(1,))
    """
    _lines.clear()
    OUT.mkdir(parents=True, exist_ok=True)
    scripted = json.loads((FIXTURES / "scripted_text.json").read_text(encoding="utf-8"))
    transcripts = json.loads((FIXTURES / "reference_text.json").read_text(encoding="utf-8"))
    hindi = sentences(scripted["hi"])[:sentence_count]

    reference = FIXTURES / "english_reference_short.wav"
    latin = transcripts["english_short"]["text"]

    section("REFERENCE TRANSCRIPTS")
    variants = [("en_latin", latin), ("en_plain", plain(latin))]

    deva = transliterate(latin)
    if deva is None:
        p("  !! IndicXlit is not installed, so the script arm cannot run:")
        p("     pip install -q ai4bharat-transliteration")
        p("     Without it this probe tests punctuation only, which is worth")
        p("     knowing but is not the question it was written for.")
    elif devanagari_fraction(deva) < 0.8:
        p(f"  !! the transliteration came back {devanagari_fraction(deva):.0%} "
          "Devanagari, which is not a script arm. Dropped rather than run")
        p(f"     and mislabelled: {deva[:80]!r}")
        deva = None
    else:
        variants.append(("en_deva", deva))

    for name, text in variants:
        p(f"\n  {name:<9} {len(text):>4} chars, {len(text.encode('utf-8')):>4} bytes, "
          f"{devanagari_fraction(text):.0%} Devanagari")
        p(f"            {text}")
    p("")
    p("  Bytes against characters is the check that matters. One byte per")
    p("  character is Latin whatever the arm is called.")

    install_patches()
    model = load_indicf5()

    rows = []
    for arm, transcript in variants:
        p(f"\n--- {arm}")
        for index, sentence in enumerate(hindi):
            _mode["speed"] = "auto"
            _mode["one_chunk"] = True
            torch.manual_seed(SEED)
            try:
                audio = model(sentence, ref_audio_path=str(reference),
                              ref_text=transcript)
            except Exception as exc:
                p(f"    [{index}] FAILED {type(exc).__name__}: {exc}")
                continue

            wave = as_float_wave(audio)
            path = OUT / arm / f"{index:02d}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(path), wave, SAMPLE_RATE, subtype="PCM_16")

            rows.append({
                "arm": arm, "index": index, "text": sentence, "path": path,
                "language": "hi", "actual_s": len(wave) / SAMPLE_RATE,
                "natural_s": len(sentence) / NATURAL_CPS_HI,
                "sim": float("nan"), "pos": float("nan"),
            })
            p(f"    [{index}] {rows[-1]['actual_s']:5.2f}s")

    if not rows:
        return rows

    del model
    torch.cuda.empty_cache()
    transcribe_outputs(rows)

    section("PREFIX, BY REFERENCE TRANSCRIPT")
    p(f"{'arm':<10}{'clips':>7}{'with prefix':>13}{'worst':>9}{'mean lead':>11}"
      f"{'extra':>8}{'cer':>7}")
    p("")
    flagged = {}
    for arm, _ in variants:
        items = [r for r in rows if r["arm"] == arm]
        if not items:
            continue
        bad = [r for r in items if r.get("lead", 0) > MAX_LEAD]
        worst = max(items, key=lambda r: r.get("lead_chars", 0))
        flagged[arm] = len(bad)
        p(f"{arm:<10}{len(items):>7}{len(bad):>13}"
          f"{worst.get('lead_chars', 0):>6} ch{mean(items, 'lead'):>11.3f}"
          f"{mean(items, 'extra'):>8.3f}{mean(items, 'cer'):>7.3f}")
        if worst.get("lead_chars"):
            p(f"          [{worst['index']}] opens with "
              f"{(worst.get('heard') or '')[:worst['lead_chars']]!r}")

    section("VERDICT")
    base = flagged.get("en_latin")
    no_punct = flagged.get("en_plain")
    script = flagged.get("en_deva")

    p(f"  punctuation   en_latin {base} -> en_plain {no_punct}")
    if script is None:
        p("  script        not tested")
    else:
        p(f"  script        en_plain {no_punct} -> en_deva {script}")
    p("")

    if base and no_punct == 0:
        p("  Punctuation in the reference transcript is what the model cannot")
        p("  reconcile with the audio. Stripping it is free, needs no")
        p("  transliteration, and the ASR stage can emit it either way.")
        if script == 0:
            p("  The script arm adds nothing on top, so Devanagari is not")
            p("  required and IndicXlit stays out of the pipeline.")
        elif script:
            p("  Transliterating on top makes it worse, so do not.")
    elif base and no_punct and script == 0:
        p("  Script is the cause and punctuation is not. The reference")
        p("  transcript has to be transliterated into the target script, which")
        p("  puts IndicXlit in the pipeline.")
    elif base and no_punct == base:
        p("  Neither changes it. The transcript is not the cause and the")
        p("  reference audio is what is left. Shorten it and re-measure.")
    else:
        p("  No clean reading. One flagged clip in the baseline is thin")
        p("  evidence either way — raise sentence_count before concluding.")

    p("")
    p(f"  Baseline had {base} flagged clip(s) of {sentence_count}. A change")
    p("  from one to zero is suggestive, not established. What makes it worth")
    p("  acting on anyway is that stripping punctuation costs nothing.")

    PROBE_REPORT.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nwrote {PROBE_REPORT}")
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
            if row.get("asr_looped"):
                notes.append("UNREADABLE (ASR looped)")
            if row.get("lead", 0) > MAX_LEAD:
                notes.append(f"PREFIX {row.get('lead_chars', 0)} chars")
            if row.get("extra", 0) > MAX_EXTRA:
                notes.append("GIBBERISH")
            if row.get("missing", 0) > MAX_MISSING:
                notes.append("CUT SHORT")
            # Substituted gibberish is the one failure the positional checks
            # cannot see. "Seven were impossible, and we had to rewrite them."
            # came back as "Sraindari ansu alwe atcho rureshi chong." — right
            # length, right rhythm, no inserted or missing span anywhere, and
            # not one correct word. Only the error rate catches that.
            if row.get("cer", 0) > MAX_CER:
                notes.append("MANGLED")
            print(f"\n{row['arm']}  [{index}]  {row['actual_s']:.2f}s  "
                  f"{row['actual_s'] / row['natural_s']:.2f}x natural  "
                  f"sim {row['sim']:.3f}  {row['pos']:.0f}%"
                  + ("   " + "  ".join(notes) if notes else ""))
            print(f"  asked: {row['text']}")
            heard = row.get("heard") or ""
            if row.get("asr_looped"):
                print(f"  heard: {heard[:160]}...  [{len(heard)} chars from a "
                      f"{row['actual_s']:.1f}s clip]")
            elif heard and row.get("cer", 0) > 0.05:
                print(f"  heard: {heard}"
                      + (f"   [cer {row['cer']:.2f}]"
                         if row.get("cer", 0) > MAX_CER else ""))
            display(Audio(str(row["path"])))


def probe(sentence_count=7):
    """
    Just the prefix question, without paying for the rest of the run.

    main() loads XTTS to build the calibrated scale, which is most of its wall
    clock and none of its value here: the prefix is a content failure and
    identity has already been measured. This loads Whisper once for the
    reference transcript, IndicF5 once for the synthesis, and Whisper again to
    read the result back.

    Two arms, differing in one thing — the script of the reference transcript.
    Both are regenerated rather than compared against a previous run's numbers,
    so the comparison is controlled even though the seed makes it reproducible.

        rows = eng.probe()
        eng.listen(rows, arms=("en_hi", "en_deva"))
    """
    _lines.clear()
    OUT.mkdir(parents=True, exist_ok=True)
    scripted = json.loads((FIXTURES / "scripted_text.json").read_text(encoding="utf-8"))
    transcripts = json.loads((FIXTURES / "reference_text.json").read_text(encoding="utf-8"))
    hindi = sentences(scripted["hi"])[:sentence_count]

    reference = FIXTURES / "english_reference_short.wav"
    latin = transcripts["english_short"]["text"]

    section("REFERENCE TRANSCRIPTS")
    deva = hear(reference, "hi")
    p(f"  latin      {len(latin):>4} chars, {len(latin.encode('utf-8')):>4} bytes")
    p(f"             {latin}")
    p(f"  devanagari {len(deva):>4} chars, {len(deva.encode('utf-8')):>4} bytes")
    p(f"             {deva}")
    p("")
    p("  The same ten seconds of English, read by Whisper in Hindi. Not a")
    p("  transliteration of the spelling — a description of what is on the tape,")
    p("  in the script the model was trained on.")

    if not deva:
        p("\n!! empty transcript, nothing to compare")
        return []

    install_patches()
    model = load_indicf5()

    rows = []
    for arm, transcript in (("en_hi", latin), ("en_deva", deva)):
        p(f"\n--- {arm}")
        for index, sentence in enumerate(hindi):
            _mode["speed"] = "auto"
            _mode["one_chunk"] = True
            torch.manual_seed(SEED)
            try:
                audio = model(sentence, ref_audio_path=str(reference),
                              ref_text=transcript)
            except Exception as exc:
                p(f"    [{index}] FAILED {type(exc).__name__}: {exc}")
                continue

            wave = as_float_wave(audio)
            path = OUT / arm / f"{index:02d}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(path), wave, SAMPLE_RATE, subtype="PCM_16")

            rows.append({
                "arm": arm, "index": index, "text": sentence, "path": path,
                "language": "hi", "actual_s": len(wave) / SAMPLE_RATE,
                "natural_s": len(sentence) / NATURAL_CPS_HI,
                "sim": float("nan"), "pos": float("nan"),
            })
            p(f"    [{index}] {rows[-1]['actual_s']:5.2f}s")

    if not rows:
        return rows

    del model
    torch.cuda.empty_cache()
    transcribe_outputs(rows)

    section("PREFIX, BY REFERENCE SCRIPT")
    p(f"{'arm':<10}{'clips':>7}{'with prefix':>13}{'worst':>8}{'mean lead':>11}"
      f"{'extra':>8}{'cer':>7}")
    p("")
    lead = {}
    for arm in ("en_hi", "en_deva"):
        items = [r for r in rows if r["arm"] == arm]
        if not items:
            continue
        flagged = [r for r in items if r.get("lead", 0) > MAX_LEAD]
        worst = max(items, key=lambda r: r.get("lead_chars", 0))
        lead[arm] = len(flagged)
        p(f"{arm:<10}{len(items):>7}{len(flagged):>13}"
          f"{worst.get('lead_chars', 0):>6} ch{mean(items, 'lead'):>11.3f}"
          f"{mean(items, 'extra'):>8.3f}{mean(items, 'cer'):>7.3f}")
        if worst.get("lead_chars"):
            spoken = (worst.get("heard") or "")[:worst["lead_chars"]]
            p(f"          [{worst['index']}] opens with {spoken!r}")

    section("VERDICT")
    if len(lead) == 2:
        if lead["en_deva"] == 0 < lead["en_hi"]:
            p("  The script of the reference transcript is the cause. The model")
            p("  cannot align Latin text to audio, speaks the unconsumed")
            p("  remainder at the start of the generated region, and stops doing")
            p("  so the moment the transcript is in a script it can read.")
            p("")
            p("  This is a pipeline decision, not a tweak: the reference")
            p("  transcript should come from an ASR pass in the target script,")
            p("  which the ASR stage already performs, rather than from the")
            p("  source-language transcript.")
        elif lead["en_deva"] >= lead["en_hi"]:
            p("  Script is not the cause — the prefix survives it, or worsens.")
            p("  What is left is the reference audio itself: the model cannot")
            p("  align English speech to any transcript. Shortening the")
            p("  reference is the next lever, since a shorter clip leaves less")
            p("  unconsumed text to spill. A Hindi reference remains the only")
            p("  configuration measured clean.")
        else:
            p(f"  Mixed: {lead['en_hi']} against {lead['en_deva']} of "
              f"{sentence_count}. Too few clips to call. Listen.")

    PROBE_REPORT.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nwrote {PROBE_REPORT}")
    return rows
