"""
Why IndicF5 babbles when the reference clip is English.

The hi_ref arm is clean and the en_ref arms are not, so the difference is
something the reference language changes. Reading f5_tts/infer/utils_infer.py
rather than guessing, an English reference changes three things at once:

  1. Duration allocation. infer_batch_process computes

         duration = ref_audio_len + ref_audio_len / ref_text_bytes
                                    * gen_text_bytes / speed

     entirely in UTF-8 bytes. Latin is 1 byte per character and Devanagari is
     3, so an English reference values a Hindi character at three times its
     real cost. Our English reference runs 0.0749 s/byte and our Hindi one
     0.0348 s/byte: the same Hindi sentence is handed roughly 2.15x the time
     it needs. F5-TTS is an in-filling model — it is told the total duration
     up front and must produce exactly that many frames. Invented speech is
     what filling the surplus would look like.

  2. Chunking. infer_process sets

         max_chars = ref_text_bytes / ref_seconds * (25 - ref_seconds)

     also in bytes, so the English reference permits about 190 bytes of
     generated text — roughly 64 Devanagari characters — while the Hindi one
     permits about 450. Longer sentences therefore take a completely different
     code path on the English arm: split at punctuation, generated
     independently, and cross-faded back together at 0.15 s.

  3. Reference truncation, on the 25 s arm only. preprocess_ref_audio_text
     clips audio over 15 s and never truncates ref_text to match, so the model
     is told that 13.7 s of audio contains 25 s of transcript.

Three candidate causes, and the last one cannot explain the 10 s arm. This
module separates the first two by varying one at a time and measuring the
result, instead of fixing both and declaring victory.

The measurement is taken from inside the library. chunk_text and
infer_batch_process are patched in f5_tts.infer.utils_infer, and model.sample
is wrapped to capture the duration actually requested. Both patched names are
resolved as module globals at call time, so this works no matter how IndicF5's
remote code imported infer_process. If the patch never fires the report says
so rather than reporting arithmetic this file invented.

    import colab.indicf5_diagnose as diag
    rows = diag.main()
    diag.listen(rows)
"""

import inspect
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

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

OUT = Path("/content/indicf5_diagnose")
REPORT = Path("/content/indicf5_diagnose.txt")

SAMPLE_RATE = 24000
HOP_LENGTH = 256
FRAME_RATE = SAMPLE_RATE / HOP_LENGTH      # 93.75 mel frames per second
SEED = 0

# name, reference clip, transcript key, speed override, force a single chunk
#
# Only one thing moves between en10_base and each of the next two. en10_both
# is not a third hypothesis, it is the check that the two effects do not
# interact. hi10_base is the arm already known to be clean, carried along so
# the run contains its own control. en25_base carries the truncation bug.
ARMS = [
    ("en10_base",  "english_reference_short.wav", "english_short", None,   False),
    ("en10_speed", "english_reference_short.wav", "english_short", "auto", False),
    ("en10_one",   "english_reference_short.wav", "english_short", None,   True),
    ("en10_both",  "english_reference_short.wav", "english_short", "auto", True),
    ("hi10_base",  "hindi_reference_short.wav",   "hindi_short",   None,   False),
    ("en25_base",  "english_reference.wav",       "english",       None,   False),
]

_calls = []
_pending = {}
_mode = {"speed": None, "one_chunk": False}

_lines = []


def p(text=""):
    print(text)
    _lines.append(text)


def section(title):
    p("")
    p("=" * 76)
    p(title)
    p("=" * 76)


def install_patches():
    """
    Instrument utils_infer in place. Idempotent, so a second import or a
    re-run inside the same kernel does not stack wrappers.
    """
    from f5_tts.infer import utils_infer

    if getattr(utils_infer, "_diagnose_installed", False):
        return

    original_chunk = utils_infer.chunk_text
    original_batch = utils_infer.infer_batch_process
    batch_signature = inspect.signature(original_batch)

    def chunk_text(text, max_chars=135):
        _pending["max_chars"] = max_chars
        if _mode["one_chunk"]:
            return [text]
        return original_chunk(text, max_chars=max_chars)

    def infer_batch_process(*args, **kwargs):
        bound = batch_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments

        audio, sr = arguments["ref_audio"]
        ref_seconds = audio.shape[-1] / sr

        # Replicating the one-space append infer_batch_process is about to do,
        # so the byte count recorded here is the byte count it will divide by.
        ref_text = arguments["ref_text"]
        if ref_text and len(ref_text[-1].encode("utf-8")) == 1:
            ref_text = ref_text + " "
        ref_bytes = len(ref_text.encode("utf-8"))

        batches = list(arguments["gen_text_batches"])
        gen_bytes = sum(len(b.encode("utf-8")) for b in batches)
        gen_chars = sum(len(b) for b in batches)

        allocated = ref_seconds / ref_bytes * gen_bytes if ref_bytes else float("nan")
        wanted = gen_chars / NATURAL_CPS_HI if gen_chars else float("nan")

        if _mode["speed"] == "auto":
            # Divide out exactly the over-allocation measured on this call.
            # Nothing here assumes a script or a bytes-per-character constant:
            # it is the time the formula asked for over the time the text needs
            # at the Hindi rate measured from FLEURS.
            arguments["speed"] = allocated / wanted if wanted else 1.0
        elif _mode["speed"] is not None:
            arguments["speed"] = float(_mode["speed"])

        model = arguments["model_obj"]
        original_sample = model.sample
        sample_signature = inspect.signature(original_sample)
        frames = []

        def sample(*sample_args, **sample_kwargs):
            inner = sample_signature.bind(*sample_args, **sample_kwargs)
            inner.apply_defaults()
            frames.append(int(inner.arguments["duration"]))
            return original_sample(*sample_args, **sample_kwargs)

        model.sample = sample
        try:
            result = original_batch(**arguments)
        finally:
            try:
                del model.sample
            except AttributeError:
                pass

        ref_frames = int(ref_seconds * SAMPLE_RATE) // HOP_LENGTH
        requested = (sum(frames) - len(frames) * ref_frames) / FRAME_RATE

        _calls.append({
            "ref_s": ref_seconds,
            "ref_bytes": ref_bytes,
            "s_per_byte": ref_seconds / ref_bytes if ref_bytes else float("nan"),
            "max_chars": _pending.get("max_chars"),
            "chunks": len(batches),
            "gen_bytes": gen_bytes,
            "gen_chars": gen_chars,
            "speed": arguments.get("speed"),
            "formula_s": allocated,
            "requested_s": requested,
            "natural_s": wanted,
        })
        return result

    utils_infer.chunk_text = chunk_text
    utils_infer.infer_batch_process = infer_batch_process
    utils_infer._diagnose_installed = True


def correlation(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys)
             if np.isfinite(x) and np.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    a = np.array([x for x, _ in pairs])
    b = np.array([y for _, y in pairs])
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main(sentence_count=7):
    needed = ["scripted_text.json", "reference_text.json",
              "english_reference_short.wav", "hindi_reference_short.wav",
              "english_reference.wav"]
    missing = [n for n in needed if not (FIXTURES / n).exists()]
    if missing:
        p(f"!! missing fixtures: {missing}")
        return []

    OUT.mkdir(parents=True, exist_ok=True)
    scripted = json.loads((FIXTURES / "scripted_text.json").read_text(encoding="utf-8"))
    transcripts = json.loads((FIXTURES / "reference_text.json").read_text(encoding="utf-8"))
    hindi = sentences(scripted["hi"])[:sentence_count]

    install_patches()

    section("SETUP")
    p(f"{len(hindi)} Hindi sentences x {len(ARMS)} arms")
    p("")
    p("  Held constant across every arm: the sentences, the seed, the model,")
    p("  and the scorer. Between en10_base and en10_speed only the duration")
    p("  handed to the sampler moves. Between en10_base and en10_one only the")
    p("  chunking moves. Anything that changes in both was not caused by either.")

    model = load_indicf5()

    rows = []
    for arm, filename, key, speed, one_chunk in ARMS:
        transcript = transcripts[key]["text"]
        p(f"\n--- {arm}   ref {filename}, {len(transcript)} chars of transcript")

        for index, sentence in enumerate(hindi):
            _mode["speed"] = speed
            _mode["one_chunk"] = one_chunk
            _calls.clear()
            _pending.clear()

            torch.manual_seed(SEED)
            started = time.perf_counter()
            try:
                audio = model(sentence,
                              ref_audio_path=str(FIXTURES / filename),
                              ref_text=transcript)
            except Exception as exc:
                p(f"    [{index}] FAILED {type(exc).__name__}: {exc}")
                continue
            elapsed = time.perf_counter() - started

            wave = as_float_wave(audio)
            path = OUT / arm / f"{index:02d}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, wave, SAMPLE_RATE)

            call = dict(_calls[-1]) if _calls else {}
            row = {
                "arm": arm, "index": index, "text": sentence, "path": path,
                "actual_s": len(wave) / SAMPLE_RATE,
                "natural_s": len(sentence) / NATURAL_CPS_HI,
                "wall_s": elapsed, "instrumented": bool(_calls),
            }
            row.update(call)
            rows.append(row)
            p(f"    [{index}] {row['actual_s']:5.2f}s "
              f"asked {call.get('requested_s', float('nan')):5.2f}s "
              f"natural {row['natural_s']:5.2f}s "
              f"chunks {call.get('chunks', '?')}")

    if not rows:
        p("\n!! nothing generated")
        return rows

    if not any(r["instrumented"] for r in rows):
        section("INSTRUMENTATION FAILED")
        p("  The patched chunk_text and infer_batch_process were never called,")
        p("  so IndicF5 does not reach the generated audio through")
        p("  f5_tts.infer.utils_infer on this install. Every duration and chunk")
        p("  column below would be this file's arithmetic rather than a")
        p("  measurement. Stop here and find the real call path before reading")
        p("  anything else as evidence.")
        return rows

    del model
    torch.cuda.empty_cache()
    transcribe_outputs(rows)

    # ------------------------------------------------------- what each arm did
    section("WHAT THE LIBRARY DID — measured from inside infer_batch_process")
    p(f"{'arm':<12}{'ref_s':>7}{'ref_B':>7}{'s/byte':>9}{'max_chars':>11}"
      f"{'chunks':>8}{'speed':>8}")
    p("")
    for arm, _, _, _, _ in ARMS:
        items = [r for r in rows if r["arm"] == arm and r.get("ref_s")]
        if not items:
            continue
        first = items[0]
        chunks = np.mean([r["chunks"] for r in items])
        p(f"{arm:<12}{first['ref_s']:>7.1f}{first['ref_bytes']:>7}"
          f"{first['s_per_byte']:>9.4f}{str(first['max_chars']):>11}"
          f"{chunks:>8.1f}{first.get('speed') or 1.0:>8.2f}")
    p("")
    p("  ref_s is the clip after the library's own silence trimming and 15s")
    p("  clipping, not the file on disk. s/byte is what the duration formula")
    p("  believes a byte of text costs.")

    # --------------------------------------------------------------- per clip
    section("PER CLIP")
    p(f"{'arm':<12}{'i':>3}{'chars':>7}{'chunk':>6}{'asked':>8}{'natural':>9}"
      f"{'asked/nat':>11}{'got/nat':>9}{'extra':>8}{'missing':>9}")
    p("")
    for row in rows:
        asked = row.get("requested_s", float("nan"))
        natural = row["natural_s"]
        flag = ""
        if row.get("extra", 0) > MAX_EXTRA:
            flag = "  <-- invented"
        elif row.get("missing", 0) > MAX_MISSING:
            flag = "  <-- cut short"
        p(f"{row['arm']:<12}{row['index']:>3}{len(row['text']):>7}"
          f"{row.get('chunks', 0):>6}{asked:>8.2f}{natural:>9.2f}"
          f"{asked / natural:>11.2f}{row['actual_s'] / natural:>9.2f}"
          f"{row.get('extra', float('nan')):>8.2f}"
          f"{row.get('missing', float('nan')):>9.2f}{flag}")

    # ---------------------------------------------------------------- summary
    section("PER ARM")
    p(f"{'arm':<12}{'n':>3}{'asked/nat':>11}{'got/nat':>9}{'extra':>8}"
      f"{'missing':>9}{'cer':>7}{'bad':>6}")
    p("")
    summary = {}
    for arm, _, _, _, _ in ARMS:
        items = [r for r in rows if r["arm"] == arm]
        if not items:
            continue
        ratio = float(np.nanmean([r.get("requested_s", np.nan) / r["natural_s"]
                                  for r in items]))
        got = float(np.mean([r["actual_s"] / r["natural_s"] for r in items]))
        extra = float(np.nanmean([r.get("extra", np.nan) for r in items]))
        miss = float(np.nanmean([r.get("missing", np.nan) for r in items]))
        cer = float(np.nanmean([r.get("cer", np.nan) for r in items]))
        bad = sum(1 for r in items if r.get("extra", 0) > MAX_EXTRA
                  or r.get("missing", 0) > MAX_MISSING)
        summary[arm] = extra
        p(f"{arm:<12}{len(items):>3}{ratio:>11.2f}{got:>9.2f}{extra:>8.3f}"
          f"{miss:>9.3f}{cer:>7.3f}{bad:>6}")
    p("")
    p(f"  extra is inserted characters over intended length; anything above")
    p(f"  {MAX_EXTRA:.2f} is speech that was never asked for.")

    section("CORRELATION")
    over = [r.get("requested_s", np.nan) / r["natural_s"] for r in rows]
    chunks = [float(r.get("chunks", np.nan)) for r in rows]
    extras = [r.get("extra", np.nan) for r in rows]
    p(f"  invented speech vs over-allocation   r = {correlation(over, extras):+.2f}")
    p(f"  invented speech vs chunk count       r = {correlation(chunks, extras):+.2f}")
    p(f"  over-allocation vs chunk count       r = {correlation(over, chunks):+.2f}")
    p("")
    p("  Across all clips, so a high r on one and not the other separates the")
    p("  two mechanisms even before the arms are compared.")

    # ----------------------------------------------------------------- verdict
    section("READING — written before the numbers, so it cannot be fitted to them")
    base = summary.get("en10_base", float("nan"))
    for arm in ("en10_speed", "en10_one", "en10_both", "hi10_base", "en25_base"):
        value = summary.get(arm, float("nan"))
        p(f"  {arm:<12} extra {value:.3f}   vs en10_base {base:.3f}")
    p("")
    p("  en10_speed clean, en10_one dirty   -> duration over-allocation is the")
    p("        cause. The byte formula cannot price a 3-byte script from a")
    p("        1-byte reference, and the surplus time gets filled with speech.")
    p("  en10_one clean, en10_speed dirty   -> chunking is the cause. The")
    p("        English reference forces short sentences to be split and")
    p("        cross-faded, and the seam is what is being heard.")
    p("  both clean on their own            -> two independent faults, both")
    p("        real, and en10_both is the only configuration that ships.")
    p("  neither clean, en10_both clean     -> they interact; neither fix alone")
    p("        is worth landing.")
    p("  none clean, hi10_base clean        -> it is the English audio itself,")
    p("        not the arithmetic around it. The next test is an English clip")
    p("        with its transcript transliterated into Devanagari, which holds")
    p("        the audio fixed and moves the script.")
    p("")
    p("  en25_base is not part of that comparison. It carries a third fault of")
    p("  its own — 25s of transcript against 13.7s of clipped audio — and is")
    p("  here only to confirm it is worse than en10_base rather than the same.")

    REPORT.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nwrote {REPORT}")
    return rows


def listen(rows, arms=None, indexes=None):
    """
    Play one sentence at a time across arms, so the same words are heard
    under each condition back to back.

        listen(rows, arms=("en10_base", "en10_speed"))
    """
    from IPython.display import Audio, display

    order = [a for a, _, _, _, _ in ARMS]
    picked = [r for r in rows
              if (arms is None or r["arm"] in arms)
              and (indexes is None or r["index"] in indexes)]

    for index in sorted({r["index"] for r in picked}):
        group = sorted([r for r in picked if r["index"] == index],
                       key=lambda r: order.index(r["arm"]))
        if not group:
            continue
        print(f"\n{'=' * 70}\n[{index}] {group[0]['text']}\n{'=' * 70}")
        for row in group:
            extra = row.get("extra", float("nan"))
            note = "  GIBBERISH" if extra > MAX_EXTRA else ""
            if row.get("missing", 0) > MAX_MISSING:
                note += "  CUT SHORT"
            print(f"\n{row['arm']}   {row['actual_s']:.2f}s  "
                  f"{row['actual_s'] / row['natural_s']:.2f}x natural  "
                  f"extra {extra:.2f}{note}")
            if row.get("heard") and row.get("cer", 0) > 0.05:
                print(f"  heard: {row['heard']}")
            display(Audio(str(row["path"])))
