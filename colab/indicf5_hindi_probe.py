"""
hi -> hi on IndicF5: the leg the project has been calling good without measuring it.

FINDINGS §1 reports 93% of scale for `en -> hi` and §7 reports 0.719 for
`hi -> hi` — but §7 is XTTS-v2, on a different model, a different encoder and a
different scale. No IndicF5 number for `hi -> hi` exists anywhere in this repo.
Two of the product's four legs (`hi -> hi`, `hi -> en`) have been asserted from
the `en -> hi` result rather than measured, and the assertion is load-bearing:
if Hindi is not actually clean, the fine-tuning plan is aimed at the wrong half
of the problem.

This is the cheap half of closing that gap. Same shape as the transliteration
probe and deliberately so — one arm, three seeds, his own tape as the floor —
because the two are meant to be read side by side, sentence for sentence:
fixtures/sentences/fixture7_hi.json is fixture7 selected by source index, not
by a re-applied filter, so row 5 here and row 5 there are the same sentence in
two languages.

What it measures: script, content against his own floor, pace, and the ear.
What it does not measure: identity. The calibrated scale in
colab/speaker_scale.py builds its floor from `xtts.speaker_manager.speakers`,
so reporting identity today means loading XTTS — and XTTS is out of this
project. A speaker encoder that does not drag it in needs its own floor and
ceiling built from scratch (FINDINGS §2: a cosine without both is not a
measurement), which is its own piece of work and not something to bolt onto a
content run. Reporting a raw cosine here would be exactly the mistake §2 exists
to prevent.

Three things carried over from the transliteration probe, each because it has
already cost this project a wrong answer:

  - **Three seeds.** FINDINGS §14 records the seven-configuration conditioning
    sweep as a claim that turned out wrong, because the sweep's whole span was
    smaller than the scale's own noise. An arm with one seed cannot be read.

  - **His own recordings as a `floor` arm, through the identical path.** A
    synthesis CER next to zero is not a measurement. Whisper misreads his real
    Hindi too, and the slicing run shows how: `फीट` for `फ़िट`, `साथ` for
    `सात`, `लेक्शर` for `लेक्चर`. Expect this floor to sit *higher* than the
    English one — Whisper is weaker on Hindi — so the English floor of 0.035 is
    not the bar here and must not be used as one.

  - **Sentences 0 and 1 fall inside the 10 s reference clip.** The model is
    handed that audio and its transcript. An arm that works only there has
    shown nothing, so those rows are flagged and read apart.

    import colab.indicf5_hindi_probe as hindi
    rows = hindi.main()
    hindi.listen(rows)
"""

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from colab import workspace
from colab.indicf5_check import (
    FIXTURES,
    MAX_CER,
    MAX_EXTRA,
    MAX_MISSING,
    NATURAL_CPS_HI,
    as_float_wave,
    load_indicf5,
    transcribe_outputs,
)
from colab.indicf5_diagnose import FRAME_RATE, install_patches, _calls, _mode
from colab.indicf5_english import MAX_LEAD, plain
from src.eval.translation_metrics import script_ratio

OUT = workspace.out("indicf5_hindi_probe")
REPORT = workspace.report("indicf5_hindi_probe.txt")
ROWS = OUT / "rows.json"

SAMPLE_RATE = 24000
REFERENCE = "hindi_reference_short.wav"
REFERENCE_KEY = "hindi_short"

SENTENCES = FIXTURES / "sentences" / "fixture7_hi.json"
SLOTS = FIXTURES / "hi_speaker" / "slots.json"
HI_SPEAKER = FIXTURES / "hi_speaker"

ARM = "hi"
SEEDS = (0, 1, 2)

# Hindi synthesis may still come back transcribed in Latin — Whisper's
# `language` is a hint, not a constraint (FINDINGS §13). A transcript in the
# wrong script scores as all errors, which reads as the model failing when the
# ruler is what moved. Gate on script before reading any error rate.
MIN_DEVANAGARI_FRACTION = 0.9

# The generated span must match the slot to within a mel frame. Wider than that
# and fix_duration did not take effect, whatever the report says.
ONE_HOP_S = 1.0 / FRAME_RATE
SPAN_TOLERANCE_S = 2 * ONE_HOP_S

SAVED = ("arm", "seed", "label", "index", "text", "path", "language",
         "actual_s", "slot_s", "natural_s", "requested_s", "in_reference",
         "instrumented", "wall_s")

_lines = []


def p(text=""):
    print(text)
    _lines.append(text)


def section(title):
    p("\n" + "=" * 76)
    p(title)
    p("=" * 76)


def mean(rows, key):
    values = [r[key] for r in rows
              if isinstance(r.get(key), (int, float)) and np.isfinite(r[key])]
    return float(np.mean(values)) if values else float("nan")


def fmt(value, width=7, places=3):
    if isinstance(value, (int, float)) and np.isfinite(value):
        return f"{value:>{width}.{places}f}"
    return f"{'-':>{width}}"


# ------------------------------------------------------------------- rows


def save_rows(rows):
    """
    What was generated, so a lost kernel does not cost the GPU time again.

    Paths are written as strings; everything unserializable is dropped rather
    than allowed to fail the write after the synthesis has already happened.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    payload = []
    for row in rows:
        entry = {k: row.get(k) for k in SAVED if k in row}
        entry["path"] = str(row["path"])
        payload.append(entry)
    ROWS.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def _rebuild_rows():
    """Every clip on disk, whether or not rows.json remembers it."""
    if not OUT.is_dir():
        return []

    frozen = {e["id"]: e for e in
              json.loads(SENTENCES.read_text(encoding="utf-8"))["sentences"]}
    slots = {s["id"]: s for s in
             json.loads(SLOTS.read_text(encoding="utf-8"))["slots"]}

    rows = []
    for clip in sorted(OUT.glob("*/*.wav")):
        label = clip.parent.name
        try:
            index = int(clip.stem)
        except ValueError:
            continue
        if index not in frozen:
            continue
        arm, _, seed = label.partition("_s")
        rows.append({
            "arm": arm, "seed": int(seed) if seed.isdigit() else 0,
            "label": label.replace("_s", "/s"), "index": index,
            "text": frozen[index]["text"], "path": clip, "language": "hi",
            "actual_s": sf.info(str(clip)).duration,
            "slot_s": slots[index]["duration_s"],
            "natural_s": frozen[index]["chars"] / NATURAL_CPS_HI,
            "in_reference": slots[index].get("in_reference", False),
        })
    return rows


def floor_rows():
    """
    His own Hindi recordings, scored through the identical path.

    These rows are a ruler, not an arm: excluded from the seed spread and the
    verdict by arm name. Read every synthesis CER below against this row and
    never against zero or against the English floor — Whisper is weaker on
    Hindi than on English and this number carries that.
    """
    if not HI_SPEAKER.is_dir():
        return []

    frozen = json.loads(SENTENCES.read_text(encoding="utf-8"))["sentences"]
    slots = {s["id"]: s for s in
             json.loads(SLOTS.read_text(encoding="utf-8"))["slots"]}

    rows = []
    for entry in frozen:
        path = HI_SPEAKER / f"{entry['id']:02d}.wav"
        if not path.exists():
            continue
        rows.append({
            "arm": "floor", "seed": 0, "label": "floor (him)",
            "index": entry["id"], "text": entry["text"],
            "path": path, "language": "hi",
            "actual_s": sf.info(str(path)).duration,
            "slot_s": slots[entry["id"]]["duration_s"],
            "natural_s": entry["chars"] / NATURAL_CPS_HI,
            "in_reference": slots[entry["id"]].get("in_reference", False),
        })
    return rows


def _provenance():
    """What code is actually running, against what is on disk — colab/reimport.py."""
    import subprocess

    from colab import indicf5_check

    repo = Path(__file__).resolve().parent.parent
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        head = "unknown"

    lines = [f"  on disk     git {head}",
             f"  loaded      ASR_DECODE {indicf5_check.ASR_DECODE}"]
    unsupported = indicf5_check.unsupported_decode_params()
    if unsupported:
        lines.append(f"  !! {unsupported} is not accepted by the installed "
                     "Whisper. If that name is not in ASR_DECODE on disk,")
        lines.append("     this kernel is stale — from colab.reimport import fresh")
    return lines


# ------------------------------------------------------------------- main


def main(seeds=SEEDS):
    _lines.clear()

    for path in (SENTENCES, SLOTS):
        if not path.exists():
            p(f"!! missing {path}")
            p("   Run scripts.slice_hindi_sentences locally, commit, then pull.")
            return []

    OUT.mkdir(parents=True, exist_ok=True)

    frozen = json.loads(SENTENCES.read_text(encoding="utf-8"))["sentences"]
    slot_data = json.loads(SLOTS.read_text(encoding="utf-8"))
    slots = {s["id"]: s for s in slot_data["slots"]}
    transcripts = json.loads(
        (FIXTURES / "reference_text.json").read_text(encoding="utf-8"))

    reference_path = FIXTURES / REFERENCE
    reference_seconds = sf.info(str(reference_path)).duration
    reference_text = plain(transcripts[REFERENCE_KEY]["text"])

    section("SETUP")
    for line in _provenance():
        p(line)
    p("")
    p(f"  reference   {REFERENCE}  {reference_seconds:.2f}s")
    p(f"              {len(reference_text)} chars, "
      f"{len(reference_text.encode('utf-8'))} bytes, punctuation stripped")
    p(f"              {reference_text}")
    p("")
    p(f"  sentences   {len(frozen)} from {SENTENCES.name}")
    p(f"  slots       his own, sliced from hindi_speech.wav; "
      f"{slot_data['measured_cps']:.1f} cps against {NATURAL_CPS_HI} on FLEURS")
    p(f"  seeds       {tuple(seeds)}")

    inside = [i for i, s in slots.items() if s.get("in_reference")]
    if inside:
        p("")
        p(f"  !! sentences {inside} fall inside the reference clip. The model is")
        p("     handed that audio and its transcript, so read those rows apart")
        p("     from the rest — an arm that works only there has shown nothing.")

    section("TEXT GATE — before any synthesis")
    p(f"{'id':>3}{'chars':>7}{'bytes':>7}{'deva':>7}{'slot':>8}{'asked cps':>11}")
    p("")
    for entry in frozen:
        text = entry["text"]
        slot = slots[entry["id"]]["duration_s"]
        p(f"{entry['id']:>3}{len(text):>7}{len(text.encode('utf-8')):>7}"
          f"{script_ratio(text, 'hi'):>6.0%}{slot:>7.2f}s"
          f"{len(text) / slot:>11.1f}")
    p("")
    p("  This gate is nearly free here — the text is the Hindi that was read")
    p("  aloud, so it is in script by construction. It is run anyway because")
    p("  the same table for the English arms is where a bad transliteration")
    p("  would be caught, and a report that changes shape between runs is")
    p("  harder to read than one that repeats an easy pass.")

    install_patches()
    model = load_indicf5()

    rows = []
    for seed in seeds:
        label = f"{ARM}/s{seed}"
        p(f"\n--- {label}")

        for entry in frozen:
            index = entry["id"]
            text = entry["text"]
            slot = slots[index]["duration_s"]

            # A total, reference included. Injected through the patch rather
            # than passed to the model, because IndicF5's remote __call__
            # decides what it forwards and a dropped keyword reverts to the
            # byte formula without raising.
            _mode["speed"] = None
            _mode["one_chunk"] = True
            _mode["fix_duration"] = reference_seconds + slot
            _calls.clear()

            torch.manual_seed(seed)
            started = time.perf_counter()
            try:
                audio = model(text, ref_audio_path=str(reference_path),
                              ref_text=reference_text)
            except Exception as exc:
                p(f"    [{index}] FAILED {type(exc).__name__}: {exc}")
                continue

            wave = as_float_wave(audio)
            path = OUT / label.replace("/", "_") / f"{index:02d}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(path), wave, SAMPLE_RATE, subtype="PCM_16")

            call = dict(_calls[-1]) if _calls else {}
            duration = len(wave) / SAMPLE_RATE
            rows.append({
                "arm": ARM, "seed": seed, "label": label, "index": index,
                "text": text, "path": path, "language": "hi",
                "actual_s": duration,
                "slot_s": slot,
                "natural_s": entry["chars"] / NATURAL_CPS_HI,
                "requested_s": call.get("requested_s", float("nan")),
                "in_reference": slots[index].get("in_reference", False),
                "instrumented": bool(_calls),
                "wall_s": time.perf_counter() - started,
            })
            p(f"    [{index}] {duration:5.2f}s  slot {slot:5.2f}s  "
              f"asked {rows[-1]['requested_s']:5.2f}s  "
              f"{duration / slot:.2f}x slot")

    if not rows:
        p("\n!! nothing generated")
        return rows

    if not any(r["instrumented"] for r in rows):
        section("INSTRUMENTATION FAILED")
        p("  The patched infer_batch_process was never called, so fix_duration")
        p("  was never applied and every duration below came from the byte")
        p("  formula. Nothing here is a measurement.")
        return rows

    section("DID fix_duration TAKE EFFECT?")
    off = [r for r in rows
           if not np.isfinite(r["requested_s"])
           or abs(r["requested_s"] - r["slot_s"]) > SPAN_TOLERANCE_S]
    if off:
        p(f"  !! {len(off)} of {len(rows)} clips were not asked for their slot.")
        for row in off[:5]:
            p(f"     {row['label']} [{row['index']}] asked "
              f"{row['requested_s']:.2f}s for a {row['slot_s']:.2f}s slot")
        p("     Every duration and pace number below describes something else.")
    else:
        p(f"  All {len(rows)} clips asked for their own slot, within "
          f"{SPAN_TOLERANCE_S * 1000:.0f} ms.")

    del model
    torch.cuda.empty_cache()
    save_rows(rows)
    rows = floor_rows() + rows
    transcribe_outputs(rows)
    return report_rows(rows)


def rescore():
    """
    Everything after synthesis, over clips already on disk.

    The Kaggle failure this exists for: a container is replaced, or the report
    is wrong for a reason that has nothing to do with the audio. Re-synthesis
    is the expensive half and the audio was never the problem.
    """
    rows = []
    if ROWS.exists():
        for entry in json.loads(ROWS.read_text(encoding="utf-8")):
            entry["path"] = Path(entry["path"])
            rows.append(entry)

    covered = {(r["label"], r["index"]) for r in rows}
    orphans = [r for r in _rebuild_rows() if (r["label"], r["index"]) not in covered]
    if orphans:
        p(f"  picked up {len(orphans)} clip(s) not in rows.json")
    rows += orphans

    if not rows:
        p(f"!! no clips under {OUT}")
        return []

    _lines.clear()
    rows = floor_rows() + rows
    transcribe_outputs(rows)
    return report_rows(rows)


# ----------------------------------------------------------------- report


def report_rows(rows):
    arms = [r for r in rows if r["arm"] == ARM]
    floor = [r for r in rows if r["arm"] == "floor"]

    section("SCRIPT GATE — is the transcript even in Devanagari?")
    p("  Whisper's `language` is a hint, not a constraint. A Hindi clip read")
    p("  back in Latin scores as every character wrong, and that is the ruler")
    p("  moving rather than the model.")
    p("")
    for row in rows:
        row["script"] = script_ratio(row.get("heard") or "", "hi")
    failed = [r for r in rows if r["script"] < MIN_DEVANAGARI_FRACTION]
    if failed:
        for row in failed:
            p(f"  !! {row['label']:<14}[{row['index']}] "
              f"{row['script']:.0%} Devanagari: {(row.get('heard') or '')[:60]}")
        p("")
        p("  Rows above are excluded from every content number below.")
    else:
        p(f"  All {len(rows)} transcripts came back in Devanagari.")
    scored = [r for r in rows if r["script"] >= MIN_DEVANAGARI_FRACTION]

    section("CONTENT — every arm against his own recording")
    p("  cer/extra/missing/lead are fractions of the intended length. `bad` is")
    p("  a clip over any of the thresholds. The floor row is Whisper on his")
    p("  own Hindi through this identical path: that is the number to read")
    p("  against, not zero, and not the 0.035 English floor.")
    p("")
    p(f"{'label':<16}{'n':>3}{'cer':>8}{'extra':>8}{'missing':>8}{'lead':>8}"
      f"{'bad':>5}")
    p("")

    def line(label, group):
        if not group:
            return
        bad = [r for r in group
               if (r.get("cer", 0) > MAX_CER or r.get("extra", 0) > MAX_EXTRA
                   or r.get("missing", 0) > MAX_MISSING
                   or r.get("lead", 0) > MAX_LEAD or r.get("asr_looped"))]
        p(f"{label:<16}{len(group):>3}{fmt(mean(group, 'cer'), 8)}"
          f"{fmt(mean(group, 'extra'), 8)}{fmt(mean(group, 'missing'), 8)}"
          f"{fmt(mean(group, 'lead'), 8)}{len(bad):>5}")

    line("floor (him)", [r for r in scored if r["arm"] == "floor"])
    for seed in sorted({r["seed"] for r in scored if r["arm"] == ARM}):
        line(f"{ARM}/s{seed}", [r for r in scored
                                if r["arm"] == ARM and r["seed"] == seed])
    line(f"{ARM} (all)", [r for r in scored if r["arm"] == ARM])

    section("LEAKAGE — the two sentences inside the reference clip")
    p("  The model was handed this audio and its transcript. If these score")
    p("  well and the rest do not, nothing has been shown.")
    p("")
    synth = [r for r in scored if r["arm"] == ARM]
    p(f"  inside  n={len([r for r in synth if r['in_reference']]):<3} "
      f"cer {fmt(mean([r for r in synth if r['in_reference']], 'cer'))}")
    p(f"  outside n={len([r for r in synth if not r['in_reference']]):<3} "
      f"cer {fmt(mean([r for r in synth if not r['in_reference']], 'cer'))}")

    section("SEED SPREAD — is any difference bigger than the noise?")
    per_seed = [mean([r for r in scored if r["arm"] == ARM and r["seed"] == s], "cer")
                for s in sorted({r["seed"] for r in scored if r["arm"] == ARM})]
    per_seed = [v for v in per_seed if np.isfinite(v)]
    if len(per_seed) > 1:
        spread = max(per_seed) - min(per_seed)
        p(f"  cer by seed   {'  '.join(f'{v:.3f}' for v in per_seed)}")
        p(f"  spread        {spread:.3f}")
        p("")
        p("  Nothing smaller than this spread is a result. FINDINGS §14 records")
        p("  a seven-configuration sweep whose entire span was inside it.")
    else:
        p("  one seed — the spread is unknown and no comparison can be read")

    section("PACE")
    p("  actual/slot is what the assembly stage has to absorb. actual/natural")
    p("  is how fast the model speaks against FLEURS Hindi at "
      f"{NATURAL_CPS_HI} cps.")
    p("")
    p(f"{'label':<16}{'act/slot':>10}{'act/natural':>13}")
    p("")
    for label, group in (("floor (him)", floor), (f"{ARM} (all)", arms)):
        if not group:
            continue
        ratio = [r["actual_s"] / r["slot_s"] for r in group if r.get("slot_s")]
        natural = [r["actual_s"] / r["natural_s"] for r in group if r.get("natural_s")]
        p(f"{label:<16}{np.mean(ratio):>10.2f}{np.mean(natural):>13.2f}")

    section("PER CLIP")
    p(f"{'label':<14}{'id':>3}{'cer':>7}{'extra':>7}{'miss':>7}{'lead':>6}"
      f"  heard")
    p("")
    for row in sorted(rows, key=lambda r: (r["arm"] != "floor", r["label"], r["index"])):
        flag = " *" if row.get("in_reference") else "  "
        p(f"{row['label']:<14}{row['index']:>3}{fmt(row.get('cer'))}"
          f"{fmt(row.get('extra'))}{fmt(row.get('missing'))}"
          f"{fmt(row.get('lead'), 6)}{flag}{(row.get('heard') or row.get('asr_error') or '')[:70]}")
    p("")
    p("  * inside the reference clip")

    section("VERDICT")
    floor_cer = mean([r for r in scored if r["arm"] == "floor"], "cer")
    arm_cer = mean([r for r in scored if r["arm"] == ARM], "cer")
    bad = [r for r in scored if r["arm"] == ARM
           and (r.get("cer", 0) > MAX_CER or r.get("extra", 0) > MAX_EXTRA
                or r.get("missing", 0) > MAX_MISSING
                or r.get("lead", 0) > MAX_LEAD or r.get("asr_looped"))]
    p(f"  floor (him)   cer {fmt(floor_cer)}")
    p(f"  {ARM} synthesis  cer {fmt(arm_cer)}   {len(bad)} bad clip(s) "
      f"of {len([r for r in scored if r['arm'] == ARM])}")
    if np.isfinite(floor_cer) and np.isfinite(arm_cer):
        p(f"  gap           {arm_cer - floor_cer:+.3f}")
    p("")
    p("  Content only. Identity is not measured here and no number in this")
    p("  report speaks to whether the clone sounds like him — see the module")
    p("  docstring. Listen before concluding anything: two of this project's")
    p("  findings were caught by ear and by no metric (FINDINGS §3b).")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    print(f"\nreport written to {REPORT}")
    return rows


def listen(rows, indexes=None):
    """Play the floor clip and each seed for a sentence, back to back."""
    from IPython.display import Audio, display

    wanted = set(indexes) if indexes is not None else None
    frozen = {e["id"]: e for e in
              json.loads(SENTENCES.read_text(encoding="utf-8"))["sentences"]}

    for index in sorted({r["index"] for r in rows}):
        if wanted is not None and index not in wanted:
            continue
        entry = frozen[index]
        print("\n" + "=" * 76)
        print(f"[{index}]  {entry['text']}")
        print(f"      en: {entry.get('english', '')}")
        print("=" * 76)
        for row in sorted([r for r in rows if r["index"] == index],
                          key=lambda r: (r["arm"] != "floor", r["label"])):
            print(f"\n{row['label']}   cer {fmt(row.get('cer'))}   "
                  f"{row.get('heard') or row.get('asr_error') or ''}")
            display(Audio(str(row["path"])))
