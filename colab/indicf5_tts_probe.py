"""
TTS generation across a duration ladder, from two references.

Everything IndicF5 has been asked to do in this project so far has been seven
sentences long, between 1.9 and 12.7 seconds, read off tape the speaker had
already recorded. That is the right shape for calibrating a ruler and the wrong
shape for finding out what the model does. This probe asks the other question:
**given arbitrary text and a duration, what comes out?**

Two arms, both generating the same Hindi. Only the reference moves:

    hi_ref    hindi_reference_short.wav   + its Hindi transcript
    en_ref    english_reference_short.wav + its English transcript

`en_ref` is the SHIPPING configuration — FINDINGS §1, 93% of scale for
`en -> hi`. That number is identity on three sentences; its content was never
anchored to anything. `hi_ref` is what §15 measured. Putting them in one run on
one script is the first time the two have been comparable.

**There is no content floor for this script.** Nothing in
fixtures/sentences/tts_ladder_hi.json was ever read aloud, so a CER here cannot
be read against a human the way §15's could. The run therefore carries a
CONTROL BLOCK — the seven fixture7_hi sentences, hi_ref, one seed — whose floor
is recorded at 0.084 and whose arm is recorded at 0.055. If the control
reproduces, the ladder numbers are comparable to §15. If it does not, something
about this session moved and nothing else in the report should be read.

**The ladder is also a chunk-length test, and the two cannot be separated.**
`fix_duration` requires `one_chunk=True` — a chunked generation prepends the
reference to every chunk and the total-frame accounting stops meaning anything
(§5a) — so a 21 s row is one 259-character chunk plus a 10 s reference, about
31 s of total mel. Degradation at the top of the ladder is a chunk-limit
finding, not a duration finding. The report says which rows are up there.

Asked characters-per-second is held constant at the speaker's own 12.19 across
the whole ladder, so what varies row to row is duration and not pace. Without
that the two are impossible to tell apart.

    import colab.indicf5_tts_probe as tts
    rows = tts.main()
    tts.listen(rows)
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
    normalize,
    transcribe_outputs,
)
from colab.indicf5_diagnose import FRAME_RATE, install_patches, _calls, _mode
from colab.indicf5_english import MAX_LEAD, plain
from colab.indicf5_hindi_probe import MIN_DEVANAGARI_FRACTION, MIXED_DEVANAGARI_FRACTION
from src.eval.translation_metrics import script_ratio
from src.text.loanwords import latin_words

OUT = workspace.out("indicf5_tts_probe")
REPORT = workspace.report("indicf5_tts_probe.txt")
ROWS = OUT / "rows.json"

SAMPLE_RATE = 24000
LADDER = FIXTURES / "sentences" / "tts_ladder_hi.json"
CONTROL = FIXTURES / "sentences" / "fixture7_hi.json"
CONTROL_SLOTS = FIXTURES / "hi_speaker" / "slots.json"

# arm -> (reference file, reference transcript key). The key `deva` is not a
# key in reference_text.json; it means fixtures/xlit/reference_deva.json, the
# English reference transcript written in Devanagari and reviewed by the
# speaker.
ARMS = {
    "hi_ref": ("hindi_reference_short.wav", "hindi_short"),
    "en_ref": ("english_reference_short.wav", "english_short"),
    # §16b: every leading prefix in the ladder run was en_ref, four clips in
    # twenty-four, all on one seed, one of them the literal English word
    # `question` in front of a Hindi sentence. The mechanism named there is a
    # reference whose audio and transcript are English while the generated text
    # is Devanagari — nothing forces the unanchored ref_audio_len slice into
    # alignment across scripts. §4d showed that transliterating this exact
    # transcript fixed the English route outright. This arm is that same change
    # applied to the shipping en -> hi path, and it is the one arm that could
    # remove a defect from the product's main route.
    #
    # Read against en_ref ON THE SAME SEEDS. The prefix appears on some seeds
    # and not others, so an unpaired comparison cannot see it go away.
    "en_ref_deva": ("english_reference_short.wav", "deva"),
}
DEFAULT_ARMS = ("hi_ref", "en_ref")
PREFIX_ARMS = ("en_ref", "en_ref_deva")
SEEDS = (0, 1)

REFERENCE_DEVA = FIXTURES / "xlit" / "reference_deva.json"

# What §15 measured on the control block, with the same model, reference and
# settings. The control is here to say whether this session is that session.
CONTROL_ARM_CER = 0.055
CONTROL_FLOOR_CER = 0.084
CONTROL_SEED_SPREAD = 0.024

# Where one chunk stops being a duration measurement. The reference is 10 s and
# F5-TTS's own chunker would never hand the sampler a span this long; past here
# a bad row is evidence about chunk length, not about duration.
LONG_CHUNK_S = 15.0

# Reading buckets for the ladder. Dubbing segments live in the first three.
BUCKETS = ((0.0, 3.0, "under 3s"), (3.0, 8.0, "3-8s"),
           (8.0, 15.0, "8-15s"), (15.0, 1e9, "over 15s"))

ONE_HOP_S = 1.0 / FRAME_RATE
SPAN_TOLERANCE_S = 2 * ONE_HOP_S

SAVED = ("arm", "seed", "label", "block", "index", "kind", "text", "path",
         "language", "actual_s", "target_s", "natural_s", "requested_s",
         # A row whose synthesis raised is the most interesting row in the run
         # and has no audio. Without this it survives save_rows() as a clip
         # that merely went missing, and rescore() drops it for having no file.
         "synthesis_error", "instrumented", "wall_s")

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


def bucket_of(seconds):
    for low, high, name in BUCKETS:
        if low <= seconds < high:
            return name
    return BUCKETS[-1][2]


def load_ladder():
    data = json.loads(LADDER.read_text(encoding="utf-8"))
    return data, data["sentences"]


def control_sentences():
    """fixture7_hi with his own slots — the block that has a floor."""
    frozen = json.loads(CONTROL.read_text(encoding="utf-8"))["sentences"]
    slots = {s["id"]: s for s in
             json.loads(CONTROL_SLOTS.read_text(encoding="utf-8"))["slots"]}
    return [{"id": e["id"], "kind": "control", "text": e["text"],
             "chars": e["chars"], "target_s": slots[e["id"]]["duration_s"]}
            for e in frozen]


def save_rows(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    payload = []
    for row in rows:
        entry = {k: row.get(k) for k in SAVED if k in row}
        entry["path"] = str(row["path"])
        payload.append(entry)
    ROWS.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def _provenance():
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
        lines.append(f"  !! {unsupported} not accepted by the installed Whisper —")
        lines.append("     this kernel may be stale. See colab/reimport.py.")
    return lines


# ------------------------------------------------------------------- main


def main(arms=DEFAULT_ARMS, seeds=SEEDS, control=True):
    _lines.clear()

    if not LADDER.exists():
        p(f"!! missing {LADDER} — git pull")
        return []

    OUT.mkdir(parents=True, exist_ok=True)
    data, ladder = load_ladder()
    transcripts = json.loads(
        (FIXTURES / "reference_text.json").read_text(encoding="utf-8"))

    section("SETUP")
    for line in _provenance():
        p(line)
    p("")
    references = {}
    for arm in arms:
        filename, key = ARMS[arm]
        path = FIXTURES / filename
        if key == "deva":
            deva = json.loads(REFERENCE_DEVA.read_text(encoding="utf-8"))
            if not deva.get("reviewed"):
                p(f"  !! {REFERENCE_DEVA.name} is not reviewed — {arm} not run")
                continue
            text, language = plain(deva["devanagari"]), "hi (transliterated en)"
        else:
            text, language = plain(transcripts[key]["text"]), transcripts[key]["language"]
        references[arm] = (path, sf.info(str(path)).duration, text)
        p(f"  {arm:<12}  {filename}  {references[arm][1]:.2f}s")
        p(f"                {len(text)} chars, {len(text.encode('utf-8'))} bytes, "
          f"{language}")
        p(f"                {text[:72]}")
    arms = [a for a in arms if a in references]
    p("")
    p(f"  seeds     {tuple(seeds)}")
    p(f"  ladder    {len(ladder)} sentences, "
      f"{ladder[0]['target_s']:.1f}s to {ladder[-1]['target_s']:.1f}s, "
      f"asked at {data['rate_cps']} cps throughout")
    p("")
    p("  !! THIS SCRIPT HAS NO FLOOR. Nothing in it was ever read aloud, so a")
    p("     CER below cannot be read against a human. The control block is the")
    p("     only calibrated thing in this run.")

    if not data.get("reviewed"):
        section("THE SCRIPT HAS NOT BEEN REVIEWED — read it before spending GPU")
        p("  Drafted by the assistant, not by the speaker. Well-formed Hindi is")
        p("  not the same as his Hindi, and an unnatural sentence makes this")
        p("  test less representative without making it fail, which is the")
        p("  worst way for text to be wrong. Fix anything that reads oddly in")
        p(f"  {LADDER} and set reviewed=true.")
        p("")
        for entry in ladder:
            p(f"  [{entry['id']:>2}] {entry['target_s']:>5.1f}s  {entry['kind']}")
            p(f"       {entry['text']}")

    section("THE LADDER — before any synthesis")
    p(f"{'id':>3}{'chars':>7}{'bytes':>7}{'target':>9}{'asked cps':>11}"
      f"{'deva':>7}  bucket")
    p("")
    for entry in ladder:
        flag = "  <-- past one chunk" if entry["target_s"] > LONG_CHUNK_S else ""
        p(f"{entry['id']:>3}{entry['chars']:>7}{entry['bytes']:>7}"
          f"{entry['target_s']:>8.2f}s{entry['chars'] / entry['target_s']:>11.1f}"
          f"{script_ratio(entry['text'], 'hi'):>6.0%}  "
          f"{bucket_of(entry['target_s'])}{flag}")
    p("")
    p("  asked cps is constant by construction, so a difference between two")
    p("  rows below is duration and not pace. Rows past one chunk are also a")
    p(f"  chunk-length test: {LONG_CHUNK_S:.0f}s of generation plus a 10s")
    p("  reference is more total mel than F5-TTS's own chunker would ever")
    p("  hand the sampler in one call.")

    blocks = [("ladder", ladder, list(arms), list(seeds))]
    if control:
        blocks.append(("control", control_sentences(), ["hi_ref"], [0]))

    install_patches()
    model = load_indicf5()

    rows = []
    for block, sentences, block_arms, block_seeds in blocks:
        for arm in block_arms:
            reference_path, reference_seconds, reference_text = references[arm]
            for seed in block_seeds:
                label = f"{block}/{arm}/s{seed}"
                p(f"\n--- {label}")

                for entry in sentences:
                    target = entry["target_s"]
                    _mode["speed"] = None
                    _mode["one_chunk"] = True
                    _mode["fix_duration"] = reference_seconds + target
                    _calls.clear()

                    torch.manual_seed(seed)
                    started = time.perf_counter()
                    try:
                        audio = model(entry["text"],
                                      ref_audio_path=str(reference_path),
                                      ref_text=reference_text)
                    except Exception as exc:
                        p(f"    [{entry['id']:>2}] FAILED "
                          f"{type(exc).__name__}: {exc}")
                        rows.append({
                            "arm": arm, "seed": seed, "label": label,
                            "block": block, "index": entry["id"],
                            "kind": entry["kind"], "text": entry["text"],
                            "path": OUT / "missing.wav", "language": "hi",
                            "actual_s": float("nan"), "target_s": target,
                            "requested_s": float("nan"), "instrumented": False,
                            "synthesis_error": f"{type(exc).__name__}: {exc}",
                            "wall_s": time.perf_counter() - started,
                        })
                        continue

                    wave = as_float_wave(audio)
                    path = (OUT / label.replace("/", "_")
                            / f"{entry['id']:02d}.wav")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(path), wave, SAMPLE_RATE, subtype="PCM_16")

                    call = dict(_calls[-1]) if _calls else {}
                    duration = len(wave) / SAMPLE_RATE
                    rows.append({
                        "arm": arm, "seed": seed, "label": label,
                        "block": block, "index": entry["id"],
                        "kind": entry["kind"], "text": entry["text"],
                        "path": path, "language": "hi",
                        "actual_s": duration, "target_s": target,
                        "natural_s": entry["chars"] / NATURAL_CPS_HI,
                        "requested_s": call.get("requested_s", float("nan")),
                        "instrumented": bool(_calls),
                        "wall_s": time.perf_counter() - started,
                    })
                    p(f"    [{entry['id']:>2}] {duration:5.2f}s  "
                      f"target {target:5.2f}s  "
                      f"asked {rows[-1]['requested_s']:5.2f}s  "
                      f"{duration / target:.2f}x  "
                      f"{time.perf_counter() - started:5.1f}s wall")

    if not rows:
        p("\n!! nothing generated")
        return rows

    if not any(r.get("instrumented") for r in rows):
        section("INSTRUMENTATION FAILED")
        p("  fix_duration was never applied. Every duration came from the byte")
        p("  formula and nothing here is a measurement.")
        return rows

    section("DID fix_duration TAKE EFFECT?")
    off = [r for r in rows
           if not np.isfinite(r.get("requested_s", float("nan")))
           or abs(r["requested_s"] - r["target_s"]) > SPAN_TOLERANCE_S]
    if off:
        p(f"  !! {len(off)} of {len(rows)} clips were not asked for their target.")
        for row in off[:8]:
            p(f"     {row['label']} [{row['index']}] asked "
              f"{row['requested_s']:.2f}s for {row['target_s']:.2f}s")
    else:
        p(f"  All {len(rows)} clips asked for their own target, within "
          f"{SPAN_TOLERANCE_S * 1000:.0f} ms.")

    del model
    torch.cuda.empty_cache()
    save_rows(rows)
    transcribe_outputs(rows)
    return report_rows(rows)


def rescore():
    """Re-transcribe and re-report clips already on disk. No synthesis."""
    _lines.clear()
    if not ROWS.exists():
        p(f"!! no {ROWS}")
        return []
    rows = []
    for entry in json.loads(ROWS.read_text(encoding="utf-8")):
        entry["path"] = Path(entry["path"])
        if entry["path"].exists():
            rows.append(entry)
    if not rows:
        p(f"!! no clips under {OUT}")
        return []
    transcribe_outputs(rows)
    return report_rows(rows)


# ----------------------------------------------------------------- report


def report_rows(rows):
    for row in rows:
        heard = row.get("heard") or ""
        folded = normalize(heard, "hi")
        row["script_heard"] = script_ratio(heard, "hi")
        row["script"] = script_ratio(folded, "hi")
        row["latin_left"] = latin_words(folded)
        row["bad"] = bool(
            row.get("cer", 0) > MAX_CER or row.get("extra", 0) > MAX_EXTRA
            or row.get("missing", 0) > MAX_MISSING
            or row.get("lead", 0) > MAX_LEAD or row.get("asr_looped")
            or row.get("synthesis_error"))

    section("SCRIPT GATE")
    failed = [r for r in rows if r["script"] < MIN_DEVANAGARI_FRACTION
              and not r.get("synthesis_error")]
    mixed = [r for r in rows if r["script_heard"] < MIXED_DEVANAGARI_FRACTION
             and r not in failed]
    p(f"  {len(mixed)} script-mixed (loanwords folded), {len(failed)} not Hindi")
    for row in failed:
        p(f"  !! {row['label']:<22}[{row['index']:>2}] {row['script']:.0%}: "
          f"{(row.get('heard') or '')[:50]}")
    scored = [r for r in rows if r["script"] >= MIN_DEVANAGARI_FRACTION]

    control = [r for r in scored if r["block"] == "control"]
    ladder = [r for r in scored if r["block"] == "ladder"]

    section("CONTROL — is this session the session §15 measured?")
    if not control:
        p("  No control block in this run. Nothing below is comparable to §15.")
    else:
        got = mean(control, "cer")
        p(f"  fixture7_hi, hi_ref, seed 0   cer {fmt(got)}")
        p(f"  §15 recorded                  cer {CONTROL_ARM_CER:.3f}   "
          f"(floor 0.084, seed spread {CONTROL_SEED_SPREAD:.3f})")
        drift = got - CONTROL_ARM_CER
        p(f"  drift                         {drift:+.3f}")
        p("")
        if abs(drift) > CONTROL_SEED_SPREAD:
            p("  !! The control moved further than one seed's worth. Something")
            p("     about this session is not the session §15 measured — the")
            p("     install, the ASR, or the reference. Read nothing below as")
            p("     comparable to §15 until that is explained.")
        else:
            p("  Inside one seed's spread. The ladder numbers are comparable to")
            p(f"  §15's arm ({CONTROL_ARM_CER:.3f}) and its floor "
              f"({CONTROL_FLOOR_CER:.3f}).")

    section("BY DURATION — the question this run was written for")
    p("  delivered/asked is how much of the requested span came back as audio.")
    p("  cer has no floor on this script; read it against the control above.")
    p("")
    p(f"{'bucket':<12}{'arm':<9}{'n':>3}{'cer':>8}{'extra':>8}{'missing':>8}"
      f"{'lead':>7}{'got/askd':>10}{'bad':>5}")
    p("")
    for _, _, name in BUCKETS:
        for arm in sorted({r["arm"] for r in ladder}):
            group = [r for r in ladder
                     if r["arm"] == arm and bucket_of(r["target_s"]) == name]
            if not group:
                continue
            ratio = np.mean([r["actual_s"] / r["target_s"] for r in group
                             if np.isfinite(r.get("actual_s", float("nan")))])
            p(f"{name:<12}{arm:<9}{len(group):>3}{fmt(mean(group, 'cer'), 8)}"
              f"{fmt(mean(group, 'extra'), 8)}{fmt(mean(group, 'missing'), 8)}"
              f"{fmt(mean(group, 'lead'), 7)}{ratio:>10.2f}"
              f"{sum(r['bad'] for r in group):>5}")

    section("BY SENTENCE — where the ladder breaks, if it does")
    p(f"{'id':>3}{'target':>8}{'kind':<24}{'arm':<9}{'cer':>8}{'got/askd':>10}"
      f"  flag")
    p("")
    for index in sorted({r["index"] for r in ladder}):
        for arm in sorted({r["arm"] for r in ladder}):
            group = [r for r in ladder if r["index"] == index and r["arm"] == arm]
            if not group:
                continue
            first = group[0]
            ratio = np.mean([r["actual_s"] / r["target_s"] for r in group
                             if np.isfinite(r.get("actual_s", float("nan")))])
            flags = []
            if first["target_s"] > LONG_CHUNK_S:
                flags.append("past one chunk")
            if any(r.get("asr_looped") for r in group):
                flags.append("ASR LOOPED")
            if any(r.get("synthesis_error") for r in group):
                flags.append("SYNTHESIS FAILED")
            if sum(r["bad"] for r in group):
                flags.append(f"{sum(r['bad'] for r in group)} bad")
            p(f"{index:>3}{first['target_s']:>7.1f}s{first['kind']:<24}"
              f"{arm:<9}{fmt(mean(group, 'cer'), 8)}{ratio:>10.2f}  "
              f"{'; '.join(flags)}")

    section("ARM — hi_ref against en_ref, on the same text")
    p("  en_ref is the shipping en -> hi configuration (§1, 93% of scale).")
    p("  Its content has never been measured next to hi_ref on one script.")
    p("")
    p(f"{'arm':<9}{'n':>4}{'cer':>8}{'extra':>8}{'missing':>8}{'lead':>7}{'bad':>5}")
    p("")
    for arm in sorted({r["arm"] for r in ladder}):
        group = [r for r in ladder if r["arm"] == arm]
        p(f"{arm:<9}{len(group):>4}{fmt(mean(group, 'cer'), 8)}"
          f"{fmt(mean(group, 'extra'), 8)}{fmt(mean(group, 'missing'), 8)}"
          f"{fmt(mean(group, 'lead'), 7)}{sum(r['bad'] for r in group):>5}")
    per_seed = [mean([r for r in ladder if r["seed"] == s], "cer")
                for s in sorted({r["seed"] for r in ladder})]
    per_seed = [v for v in per_seed if np.isfinite(v)]
    if len(per_seed) > 1:
        p("")
        p(f"  seed spread across the whole ladder  "
          f"{max(per_seed) - min(per_seed):.3f}")
        p("  An arm difference smaller than that is not a difference.")

    section("LEADING PREFIX — invented speech before the sentence begins")
    p("  §16b: every prefix in the first ladder run was en_ref, four clips in")
    p("  twenty-four, all on one seed, one of them the English word `question`")
    p("  in front of a Hindi sentence. It is in the shipping configuration and")
    p("  it is seed-dependent, so it must be read per seed and not pooled.")
    p("")
    p(f"{'arm':<14}{'seed':>5}{'n':>4}{'with a prefix':>15}{'mean lead':>11}"
      f"{'worst':>8}")
    p("")
    for arm in sorted({r["arm"] for r in ladder}):
        for seed in sorted({r["seed"] for r in ladder if r["arm"] == arm}):
            group = [r for r in ladder if r["arm"] == arm and r["seed"] == seed]
            hit = [r for r in group if r.get("lead", 0) > 0]
            worst = max((r.get("lead", 0) for r in group), default=0.0)
            p(f"{arm:<14}{seed:>5}{len(group):>4}{len(hit):>15}"
              f"{fmt(mean(group, 'lead'), 11)}{worst:>8.3f}")
    prefixed = [r for r in ladder if r.get("lead", 0) > MAX_LEAD]
    if prefixed:
        p("")
        p("  over the threshold, with what the model said first:")
        for row in prefixed:
            p(f"     {row['label']:<22}[{row['index']:>2}] lead {row['lead']:.3f}  "
              f"{(row.get('heard') or '')[:56]}")

    section("PER CLIP")
    p(f"{'label':<22}{'id':>3}{'targ':>6}{'got':>6}{'cer':>7}{'extra':>7}"
      f"{'miss':>7}{'lead':>6}{'deva':>6}  heard")
    p("")
    for row in sorted(rows, key=lambda r: (r["block"] != "control",
                                           r["label"], r["index"])):
        deva = row.get("script_heard")
        got = row.get("actual_s")
        p(f"{row['label']:<22}{row['index']:>3}{row['target_s']:>5.1f}s"
          f"{(f'{got:.1f}s' if isinstance(got, float) and np.isfinite(got) else '-'):>6}"
          f"{fmt(row.get('cer'))}{fmt(row.get('extra'))}{fmt(row.get('missing'))}"
          f"{fmt(row.get('lead'), 6)}"
          f"{(f'{deva:.0%}' if isinstance(deva, float) else '-'):>6}  "
          f"{(row.get('heard') or row.get('synthesis_error') or row.get('asr_error') or '')[:60]}")

    section("VERDICT")
    p(f"  control        cer {fmt(mean(control, 'cer'))}   "
      f"(§15 arm {CONTROL_ARM_CER:.3f}, floor {CONTROL_FLOOR_CER:.3f})")
    for arm in sorted({r["arm"] for r in ladder}):
        group = [r for r in ladder if r["arm"] == arm]
        p(f"  ladder {arm:<8} cer {fmt(mean(group, 'cer'))}   "
          f"{sum(r['bad'] for r in group)} bad of {len(group)}")
    p("")
    p("  No floor on the ladder script. The control is the anchor, and the")
    p("  numbers above are readable only relative to it. Listen before")
    p("  concluding: nothing here measures identity, naturalness or prosody,")
    p("  and this project has twice found by ear what no metric caught (§3b).")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    print(f"\nreport written to {REPORT}")
    return rows


def listen(rows, indexes=None, block="ladder"):
    """Every arm and seed for a sentence, back to back, longest last."""
    from IPython.display import Audio, display

    wanted = set(indexes) if indexes is not None else None
    chosen = [r for r in rows if r["block"] == block and r["path"].exists()]

    for index in sorted({r["index"] for r in chosen},
                        key=lambda i: [r for r in chosen if r["index"] == i][0]["target_s"]):
        if wanted is not None and index not in wanted:
            continue
        group = sorted([r for r in chosen if r["index"] == index],
                       key=lambda r: r["label"])
        print("\n" + "=" * 76)
        print(f"[{index}]  {group[0]['target_s']:.1f}s target — {group[0]['kind']}")
        print(f"      {group[0]['text']}")
        print("=" * 76)
        for row in group:
            print(f"\n{row['label']}   {row.get('actual_s', float('nan')):.1f}s   "
                  f"cer {fmt(row.get('cer'))}")
            print(f"   {row.get('heard') or row.get('synthesis_error') or ''}")
            display(Audio(str(row["path"])))
