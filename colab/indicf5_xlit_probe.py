"""
Phase 0: can IndicF5 speak intelligible Indian English out of Devanagari?

FINDINGS §4 settled that IndicF5 cannot generate English from Latin text — not
accented English, not English at all. "Seven were impossible, and we had to
rewrite them." came back as `Sraindari ansu alwe atcho rureshi chong.` The
vocabulary was ruled out as the explanation: Latin is the largest script in its
2545-token vocabulary and the English transcript tokenizes at 100%.

The hypothesis here is that the model is fluent in Devanagari and that Hindi
speech is full of English loanwords spoken with Indian phonology, so English
*spelled* in Devanagari should come out as intelligible, Indian-accented
English through the same cloning path that already scores 93% for `en -> hi`.

This is a kill shot, not the experiment. Two arms, 28 clips, one session, no
new modules and no new dependencies. If `deva_hand` — a human writing English
in Devanagari as he would say it, the upper bound for every automatic arm — is
not intelligible, then no transliterator is, and TRANSLITERATION_PLAN's phases
1 to 5 and FINETUNE_PLAN's Route A are all answered at once, for an hour of GPU
instead of a week of building.

Three things are deliberate:

  - **Three seeds on the hypothesis arm, here rather than at the end.**
    FINDINGS §14 records "the seven-configuration conditioning sweep found
    something" as a claim that turned out wrong, because the sweep's entire
    span was smaller than the scale's own noise. Every arm comparison in the
    later phases is unreadable until the seed-to-seed spread is known, so it is
    measured first, on the arm that matters, for fourteen extra clips.

  - **The `latin` control is regenerated rather than cited.** It is known to
    fail. Running it proves the content check still fires in *this* session; if
    it comes back clean, the harness is wrong and nothing else in the report
    means anything.

  - **Sentences 0 and 1 sit inside the reference clip.** fixtures/en_speaker
    records which, because the model is handed that audio and its transcript,
    and an arm that works only on those has not been shown to generalise.

    import colab.indicf5_xlit_probe as probe
    rows = probe.main()
    probe.listen(rows)
"""

import json
import re
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
    as_float_wave,
    load_indicf5,
    score_text,
    transcribe_outputs,
)
from colab.indicf5_diagnose import FRAME_RATE, install_patches, _calls, _mode
from colab.indicf5_english import MAX_LEAD, NATURAL_CPS_EN, devanagari_fraction, plain
from src.eval.translation_metrics import script_ratio

OUT = workspace.out("indicf5_xlit_probe")
REPORT = workspace.report("indicf5_xlit_probe.txt")

SAMPLE_RATE = 24000
REFERENCE = "english_reference_short.wav"
REFERENCE_KEY = "english_short"

SENTENCES = FIXTURES / "sentences" / "fixture7.json"
SLOTS = FIXTURES / "en_speaker" / "slots.json"
XLIT = FIXTURES / "xlit"

# One seed for the control, three for the hypothesis. The control's job is to
# fail; measuring how consistently it fails is not worth 14 clips.
CONTROL_SEEDS = (0,)
HYPOTHESIS_SEEDS = (0, 1, 2)

# Arms that are the hypothesis rather than the ruler. `deva_ref` is `deva_hand`
# with the reference transcript transliterated and nothing else changed, so the
# two are read as a pair and both are excluded from the control and the floor.
HYPOTHESIS_ARMS = ("deva_hand", "deva_ref")

# The accent dials, each on top of `deva_ref` — which §4d established as the
# better baseline, at the Whisper content floor. One variable from the best
# known configuration, not from the first one that worked.
#
# `deva_ref` is the paired comparison for all of them, so it must run whenever
# they do; an accent dial read against `deva_hand` would be reading two changes.
DIAL_ARMS = ("deva_dental", "deva_soft")
DIAL_SEEDS = (0, 1, 2)

# Every arm that is an arm rather than a ruler, in reading order.
REPORTED_ARMS = HYPOTHESIS_ARMS + DIAL_ARMS

# What main() runs when it is not told otherwise. `deva_hand` is deliberately
# absent: §4d settled it against `deva_ref`, which is now the baseline the
# dials are read against, and re-proving a closed comparison costs 21 clips.
# `latin` stays at 7 — it is the only thing that shows the content check fires
# in *this* session, and a clean control means the harness is broken.
#
#     probe.main()                                  70 clips
#     probe.main(only=("latin", "deva_ref"))        the §4d configuration
#     probe.main(only=ALL_ARMS)                     everything, 91 clips
DEFAULT_ARMS = ("latin", "deva_ref") + DIAL_ARMS
ALL_ARMS = ("latin",) + REPORTED_ARMS

# Indian-accented English may come back transcribed in Devanagari — Whisper's
# `language` is a hint, not a constraint (FINDINGS §13). A transcript in the
# wrong script scores as all errors, which would read as the model failing when
# the ruler is what moved. Gate on script before reading any error rate.
MIN_LATIN_FRACTION = 0.9

# The generated span must match the slot to within a mel frame. Wider than that
# and fix_duration did not take effect, whatever the report says.
ONE_HOP_S = 1.0 / FRAME_RATE
SPAN_TOLERANCE_S = 2 * ONE_HOP_S

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


def load_arm(name):
    """
    One arm's generated text, per sentence id.

    An unreviewed hand transliteration is refused rather than run. The whole
    value of this arm is that it is a human upper bound; scoring a draft and
    reporting it as `deva_hand` would answer a different question than the one
    the report claims to answer.
    """
    path = XLIT / f"{name}.json"
    if not path.exists():
        return None, f"{path} does not exist"

    data = json.loads(path.read_text(encoding="utf-8"))

    if not data.get("reviewed"):
        unreviewed = [s["id"] for s in data["sentences"] if not s.get("reviewed")]
        return None, (
            f"{path.name} is not reviewed (sentences {unreviewed}). A native "
            "speaker has to correct the draft and set reviewed=true before it "
            "can be called a gold standard."
        )

    return {s["id"]: s["devanagari"] for s in data["sentences"]}, None


REFERENCE_DEVA = XLIT / "reference_deva.json"

# Which xlit fixture supplies an arm's generated text. `deva_ref` reuses
# deva_hand's sentences unchanged — the only thing it varies is ref_text — so
# the two arms must never diverge here, or the comparison acquires a second
# variable without saying so.
ARM_TEXT = {"deva_hand": "deva_hand", "deva_ref": "deva_hand",
            "deva_dental": "deva_hand", "deva_soft": "deva_hand"}


def arm_generated_text(arm, base):
    """
    What an arm actually asks the model to say, given its source fixture.

    `deva_ref` varies the reference and not the sentences, so it is `base`
    unchanged. The dial arms are `base` put through one transform from
    src/text/en_to_deva.py — the text is derived rather than frozen, so an arm
    is a reviewed fixture plus a named function and the two cannot drift apart
    in the way two hand-maintained JSON files would.
    """
    from src.text.en_to_deva import DIALS

    if arm not in DIALS:
        return base
    return {index: DIALS[arm](text) for index, text in base.items()}


def load_reference_deva():
    """
    The reference transcript in Devanagari, for the `deva_ref` arm.

    Reviewed the same way deva_hand is, and refused the same way when it is
    not. It is a smaller review — 25 words, of which 23 are already reviewed
    inside deva_hand — but an unreviewed reference would put a drafting error
    into the conditioning of every clip in the arm rather than into one
    sentence, which is worse, not better.

    Shape differs from an arm file: one text, not seven, so `load_arm` cannot
    read it.
    """
    if not REFERENCE_DEVA.exists():
        return None, f"{REFERENCE_DEVA} does not exist"

    data = json.loads(REFERENCE_DEVA.read_text(encoding="utf-8"))
    if not data.get("reviewed"):
        return None, (
            f"{REFERENCE_DEVA.name} is not reviewed. The new words are "
            f"{data.get('new_words')} — everything else is lifted from "
            "deva_hand.json. Set reviewed=true once a speaker has read it."
        )
    return data, None


def check_text(text):
    """Refuse text the model would silently mispronounce or swallow."""
    problems = []
    fraction = devanagari_fraction(text)
    if fraction < 0.95:
        problems.append(f"only {fraction:.0%} Devanagari — a word came back in Latin")
    return problems


ROWS = OUT / "rows.json"

# Fields worth carrying across a process boundary. `path` is rebuilt from the
# layout rather than stored, so a zip unpacked somewhere else still resolves.
SAVED = ("arm", "seed", "label", "index", "text", "given", "ref_text",
         "language", "actual_s", "slot_s", "natural_s", "requested_s",
         "in_reference")


def save_rows(rows):
    """Write the synthesis result beside the audio, before transcription."""
    OUT.mkdir(parents=True, exist_ok=True)
    ROWS.write_text(json.dumps(
        [{key: row[key] for key in SAVED if key in row} for row in rows],
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _rebuild_rows():
    """
    Reconstruct rows from the clips on disk, for a run that saved none.

    The first phase 0 run predates `save_rows`, and its audio is the only copy
    of an hour of GPU. Everything the report needs is recoverable: the label
    from the directory name, the sentence id from the filename, the intended
    English and the slot from the fixtures, and the measured duration from the
    file itself. `requested_s` is not recoverable, which is why the
    fix_duration section belongs to `main` and not to `report_rows` — it can
    only be checked at the moment of generation, and it passed in that run.
    """
    frozen = json.loads(SENTENCES.read_text(encoding="utf-8"))["sentences"]
    english = {e["id"]: e["text"] for e in frozen}
    slots = {s["id"]: s for s in
             json.loads(SLOTS.read_text(encoding="utf-8"))["slots"]}

    given = {}
    for path in sorted(XLIT.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        # reference_deva.json lives here too and holds one text rather than
        # seven. Keyed by stem, not by arm: ARM_TEXT maps the arms onto it.
        if "sentences" not in data:
            continue
        given[path.stem] = {s["id"]: s["devanagari"] for s in data["sentences"]}

    rows = []
    for directory in sorted(d for d in OUT.iterdir() if d.is_dir()):
        # main() writes these as label.replace("/", "_"), so "deva_hand/s0"
        # becomes "deva_hand_s0" and a single-seed arm stays "latin". Only a
        # trailing _s<digits> is a seed; the underscore inside "deva_hand" is
        # part of the arm name.
        seeded = re.fullmatch(r"(.+)_s(\d+)", directory.name)
        arm = seeded.group(1) if seeded else directory.name
        seed = int(seeded.group(2)) if seeded else 0
        label = f"{arm}/s{seed}" if seeded else arm

        for clip in sorted(directory.glob("*.wav")):
            index = int(clip.stem)
            rows.append({
                "arm": arm, "seed": seed, "label": label,
                "index": index,
                "text": english[index],
                "given": arm_generated_text(
                    arm, given.get(ARM_TEXT.get(arm, arm), english))[index],
                "path": clip, "language": "en",
                "actual_s": sf.info(str(clip)).duration,
                "slot_s": slots[index]["duration_s"],
                "in_reference": slots[index].get("in_reference", False),
            })
    return rows


EN_SPEAKER = FIXTURES / "en_speaker"


def floor_rows():
    """
    His own recordings, scored through the identical path, as the `floor` arm.

    A synthesis CER read against zero is not a measurement. Whisper misreads
    his real English too — FINDINGS §12 puts word error at 7.9% on this
    recording — and it misreads it in exactly the places the arms are judged
    on: `fit` heard as `feet`, `to land` as `two land`, "week" as "weak".
    Scoring the real clips here puts that number in the same table instead of
    in a sentence under it telling the reader to go and find it.

    These rows are excluded from the seed spread and the verdict by arm name;
    they are a ruler, not an arm.
    """
    if not EN_SPEAKER.is_dir():
        return []

    frozen = json.loads(SENTENCES.read_text(encoding="utf-8"))["sentences"]
    slots = {s["id"]: s for s in
             json.loads(SLOTS.read_text(encoding="utf-8"))["slots"]}

    rows = []
    for entry in frozen:
        path = EN_SPEAKER / f"{entry['id']:02d}.wav"
        if not path.exists():
            continue
        rows.append({
            "arm": "floor", "seed": 0, "label": "floor (him)",
            "index": entry["id"],
            "text": entry["text"], "given": entry["text"],
            "path": path, "language": "en",
            "actual_s": sf.info(str(path)).duration,
            "slot_s": slots[entry["id"]]["duration_s"],
            "in_reference": slots[entry["id"]].get("in_reference", False),
        })
    return rows


def _provenance():
    """
    What code is actually running, against what is on disk.

    Three runs of this probe reported no content because the kernel held a
    module from before a fix while the working tree held the fix — see
    colab/reimport.py. Nothing about a stale module looks stale, so the two
    are printed side by side in every report and the reader can see the
    mismatch instead of deducing it from a symptom.
    """
    import subprocess

    from colab import indicf5_check

    repo = Path(__file__).resolve().parent.parent
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        head = "unknown"

    unsupported = indicf5_check.unsupported_decode_params()
    lines = [
        f"  on disk     git {head}",
        f"  loaded      ASR_DECODE {indicf5_check.ASR_DECODE}",
    ]
    if unsupported:
        lines.append(f"  !! {unsupported} is not accepted by the installed "
                     "Whisper. If that name is not in ASR_DECODE on disk, this")
        lines.append("     kernel is stale — from colab.reimport import fresh")
    return lines


def _hydrate(rows):
    """
    Refill from the fixtures anything a row is missing.

    Every field here has a failure that looks like a model problem. Without
    `text` the row is skipped and scores nan; without `slot_s` every pace
    number is wrong; without `in_reference` two rows the model was shown the
    answer to get read as evidence. None of them raise.
    """
    frozen = json.loads(SENTENCES.read_text(encoding="utf-8"))["sentences"]
    english = {e["id"]: e["text"] for e in frozen}
    slots = {s["id"]: s for s in
             json.loads(SLOTS.read_text(encoding="utf-8"))["slots"]}

    for row in rows:
        index = row["index"]
        row.setdefault("language", "en")
        if not row.get("text"):
            row["text"] = english[index]
        if not row.get("slot_s"):
            row["slot_s"] = slots[index]["duration_s"]
        if "in_reference" not in row:
            row["in_reference"] = slots[index].get("in_reference", False)
        if not row.get("ref_text") and row.get("arm") == "deva_ref":
            # Rebuilt from disk, so the conditioning is not recorded on the
            # row. It is recoverable for this arm and only this arm, because
            # the fixture is what main() read.
            deva, _ = load_reference_deva()
            if deva is not None:
                row["ref_text"] = plain(deva["devanagari"])
        if not row.get("actual_s") and Path(row["path"]).exists():
            row["actual_s"] = sf.info(str(row["path"])).duration
    return rows


def diagnose_asr(rows, index=0):
    """
    Try one clip three ways, when the report says nothing was transcribed.

    The three calls separate the causes that produce the same empty string: a
    clip Whisper genuinely hears nothing in, a language hint that suppresses
    the output, and a decode parameter that does. Prints what each returns
    rather than deciding.

        probe.diagnose_asr(rows)
    """
    from transformers import pipeline

    from colab.indicf5_check import ASR_DECODE, ASR_MODEL

    row = rows[index]
    path = Path(row["path"])
    print(f"clip     {path}")
    print(f"exists   {path.exists()}")
    if not path.exists():
        return
    info = sf.info(str(path))
    wave, _ = sf.read(str(path), dtype="float32")
    print(f"audio    {info.duration:.2f}s, {info.samplerate} Hz, "
          f"peak {float(np.abs(wave).max()):.4f}, "
          f"rms {float(np.sqrt((wave ** 2).mean())):.4f}")
    print(f"asked    {row.get('text')!r}")

    device = 0 if torch.cuda.is_available() else -1
    asr = pipeline("automatic-speech-recognition", model=ASR_MODEL,
                   device=device,
                   torch_dtype=torch.float16 if device == 0 else torch.float32)

    attempts = [
        ("bare", {}),
        ("language only", {"language": "en", "task": "transcribe"}),
        ("as the probe calls it",
         {"language": "en", "task": "transcribe", **ASR_DECODE}),
    ]
    for name, kwargs in attempts:
        try:
            out = asr(str(path), generate_kwargs=kwargs) if kwargs else asr(str(path))
            print(f"  {name:<22} {((out or {}).get('text') or '').strip()!r}")
        except Exception as exc:
            print(f"  {name:<22} {type(exc).__name__}: {exc}")


def rescore():
    """
    Re-read and re-score clips already on disk. No GPU synthesis, no IndicF5.

    Use it when the audio is good and the numbers are not — which is exactly
    what the first phase 0 run produced. Loads only Whisper, so it costs a
    couple of minutes rather than most of an hour.

        import colab.indicf5_xlit_probe as probe
        rows = probe.rescore()
        probe.listen(rows)
    """
    _lines.clear()

    if not OUT.is_dir():
        print(f"!! no clips under {OUT} — nothing to rescore. Run main().")
        return []

    if ROWS.exists():
        saved = json.loads(ROWS.read_text(encoding="utf-8"))
        rows = [dict(row, path=OUT / row["label"].replace("/", "_")
                     / f"{row['index']:02d}.wav") for row in saved]
    else:
        rows = _rebuild_rows()

    # rows.json is a cache; the fixtures are the truth. A row missing the
    # intended text is not scored and not reported as unscored — it is simply
    # passed over — so anything the cache lacks is refilled from the source
    # rather than trusted to be there.
    rows = _hydrate(rows)

    if not rows:
        print(f"!! {OUT} holds no clips. Run main().")
        return []

    section("RESCORE — existing clips, no synthesis")
    p(f"  {len(rows)} clips under {OUT}")
    p(f"  rows.json {'found' if ROWS.exists() else 'absent, rebuilt from disk'}")
    for line in _provenance():
        p(line)
    p("")

    # A stale `heard` from an earlier scoring attempt must not survive into
    # this one. It is the field every content number is computed from, and a
    # leftover value is indistinguishable from a fresh transcript in the
    # report — which is how an exception string from a previous run ended up
    # being read as an empty transcription three times.
    for row in rows:
        row.pop("heard", None)
        row.pop("asr_error", None)
        row.pop("asr_looped", None)
        for key in ("cer", "extra", "missing", "lead", "lead_chars"):
            row.pop(key, None)
    p("")
    p("  fix_duration is not re-checked here. It can only be observed at the")
    p("  moment of generation, so it belongs to main(); this run inherits")
    p("  whatever that one reported.")

    rows = floor_rows() + rows
    transcribe_outputs(rows)
    return report_rows(rows)


def main(only=DEFAULT_ARMS):
    """
    Synthesize and report. `only` selects which arms run; see DEFAULT_ARMS.

    Existing clips for arms not selected are left on disk untouched, so a
    scoped run does not destroy a previous one — but the report covers what
    this call generated, and `rescore()` is what reads everything present.
    """
    _lines.clear()
    only = tuple(only)

    for path in (SENTENCES, SLOTS):
        if not path.exists():
            p(f"!! missing {path}")
            p("   Run scripts.slice_english_sentences locally, commit, then pull.")
            return []

    OUT.mkdir(parents=True, exist_ok=True)

    frozen = json.loads(SENTENCES.read_text(encoding="utf-8"))["sentences"]
    slot_data = json.loads(SLOTS.read_text(encoding="utf-8"))
    slots = {s["id"]: s for s in slot_data["slots"]}
    transcripts = json.loads((FIXTURES / "reference_text.json").read_text(encoding="utf-8"))

    reference_path = FIXTURES / REFERENCE
    reference_seconds = sf.info(str(reference_path)).duration

    # Punctuation stripped and lowercased — FINDINGS §5c, the shipping policy.
    # The one clip in seven that still opened with invented speech stopped
    # doing so when this was applied, and it costs nothing.
    reference_text = plain(transcripts[REFERENCE_KEY]["text"])

    english = {e["id"]: e["text"] for e in frozen}
    hand, refusal = load_arm("deva_hand")
    reference_deva, deva_ref_refusal = load_reference_deva()

    section("SETUP")
    for line in _provenance():
        p(line)
    p("")
    p(f"  reference   {REFERENCE}  {reference_seconds:.2f}s")
    p(f"              {len(reference_text)} chars, "
      f"{len(reference_text.encode('utf-8'))} bytes, punctuation stripped")
    p(f"              {reference_text}")
    if reference_deva is not None:
        deva_reference_text = plain(reference_deva["devanagari"])
        p("")
        p(f"  deva_ref    the same audio, transcript in Devanagari")
        p(f"              {len(deva_reference_text)} chars, "
          f"{len(deva_reference_text.encode('utf-8'))} bytes")
        p(f"              {deva_reference_text}")
    p("")
    p(f"  sentences   {len(frozen)} from {SENTENCES.name}")
    p(f"  slots       his own, sliced from english_speech.wav; "
      f"{slot_data['measured_cps']:.1f} cps against {NATURAL_CPS_EN} on FLEURS")

    inside = [i for i, s in slots.items() if s.get("in_reference")]
    if inside:
        p(f"  !! sentences {inside} fall inside the reference clip. The model is")
        p("     handed that audio and its transcript, so read those rows apart")
        p("     from the rest — an arm that works only there has shown nothing.")

    # (name, generated text per id, seeds, reference transcript)
    arms = []
    if "latin" in only:
        arms.append(("latin", english, CONTROL_SEEDS, reference_text))
    if hand is None:
        p("")
        p(f"  !! deva_hand not run: {refusal}")
        p("     Without it there is no upper bound and this run cannot answer")
        p("     the question it was written for.")
    else:
        if "deva_hand" in only:
            arms.append(("deva_hand", hand, HYPOTHESIS_SEEDS, reference_text))
        wanted = [a for a in ("deva_ref",) + DIAL_ARMS if a in only]
        if wanted and reference_deva is None:
            p("")
            p(f"  -- {wanted} not run: {deva_ref_refusal}")
        elif wanted:
            # Same audio, same generated text, same seeds. Only ref_text moves.
            deva_reference = plain(reference_deva["devanagari"])
            if "deva_ref" in only:
                arms.append(("deva_ref", hand, HYPOTHESIS_SEEDS, deva_reference))
            # Each dial on top of deva_ref, which §4d put at the content floor.
            # They share its reference, so deva_ref is their paired baseline.
            for dial in (d for d in DIAL_ARMS if d in only):
                arms.append((dial, arm_generated_text(dial, hand),
                             DIAL_SEEDS, deva_reference))

            if any(d in only for d in DIAL_ARMS) and "deva_ref" not in only:
                p("")
                p("  !! a dial is running without deva_ref, which is the only")
                p("     baseline it can be read against. Every CER difference")
                p("     in this run will be against an arm that is not here.")

    section("TEXT GATES — before any synthesis")
    p(f"{'arm':<12}{'id':>3}{'chars':>7}{'bytes':>7}{'deva':>7}{'slot':>7}"
      f"{'asked cps':>11}")
    p("")
    blocked = False
    for arm, texts, _, _ in arms:
        for entry in frozen:
            text = texts[entry["id"]]
            slot = slots[entry["id"]]["duration_s"]
            problems = check_text(text) if arm != "latin" else []
            p(f"{arm:<12}{entry['id']:>3}{len(text):>7}"
              f"{len(text.encode('utf-8')):>7}"
              f"{devanagari_fraction(text):>6.0%}{slot:>6.2f}s"
              f"{len(text) / slot:>11.1f}"
              + ("   <-- " + "; ".join(problems) if problems else ""))
            blocked = blocked or bool(problems)

    p("")
    p("  Bytes against characters is the check that matters. Devanagari costs")
    p("  about 2.6 bytes a character and Latin one, which is the whole reason")
    p("  the byte-ratio duration formula cannot be trusted across scripts and")
    p("  fix_duration is used instead.")

    dialled = [(a, t) for a, t, _, _ in arms if a in DIAL_ARMS]
    if dialled and hand is not None:
        section("WHAT THE DIALS CHANGED — read this before spending the GPU")
        p("  The dial arms are not hand-written. They are src/text/en_to_deva.py")
        p("  applied to the deva_hand you reviewed, so what the model is asked")
        p("  to say is printed here rather than frozen into a file nobody reads.")
        p("")
        p("  Expect a content cost. deva_hand spells the loanwords the way Hindi")
        p("  text spells them — प्रोजेक्ट, वीडियो, सिस्टम — because that is the")
        p("  distribution IndicF5 was trained on, and `dental` destroys exactly")
        p("  that. An arm that reaches accent 6 and takes cer from 0.048 to 0.3")
        p("  has not solved the problem.")
        for arm, texts in dialled:
            p(f"\n--- {arm}")
            for entry in frozen:
                before, after = hand[entry["id"]], texts[entry["id"]]
                if before == after:
                    p(f"  [{entry['id']}] unchanged")
                    continue
                p(f"  [{entry['id']}] {before}")
                p(f"      -> {after}")

    if blocked:
        p("")
        p("!! a text gate failed. Fix the transliteration before spending GPU.")
        return []

    install_patches()
    model = load_indicf5()

    rows = []
    for arm, texts, seeds, arm_reference_text in arms:
        for seed in seeds:
            label = arm if len(seeds) == 1 else f"{arm}/s{seed}"
            p(f"\n--- {label}")

            for entry in frozen:
                sentence_id = entry["id"]
                text = texts[sentence_id]
                slot = slots[sentence_id]["duration_s"]

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
                                  ref_text=arm_reference_text)
                except Exception as exc:
                    p(f"    [{sentence_id}] FAILED {type(exc).__name__}: {exc}")
                    continue

                wave = as_float_wave(audio)
                path = OUT / label.replace("/", "_") / f"{sentence_id:02d}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(path), wave, SAMPLE_RATE, subtype="PCM_16")

                call = dict(_calls[-1]) if _calls else {}
                duration = len(wave) / SAMPLE_RATE
                rows.append({
                    "arm": arm, "seed": seed, "label": label,
                    "index": sentence_id,
                    # Scored against the English, never the Devanagari fed to
                    # the model: the question is whether a listener hears the
                    # sentence, not whether the model read the spelling back.
                    "text": english[sentence_id],
                    "given": text,
                    # deva_hand and deva_ref differ in this field and nothing
                    # else, so a row that cannot say which it used cannot be
                    # attributed to an arm after the fact.
                    "ref_text": arm_reference_text,
                    "path": path, "language": "en",
                    "actual_s": duration,
                    "slot_s": slot,
                    "natural_s": len(english[sentence_id]) / NATURAL_CPS_EN,
                    "requested_s": call.get("requested_s", float("nan")),
                    "in_reference": slots[sentence_id].get("in_reference", False),
                    "instrumented": bool(_calls),
                    "wall_s": time.perf_counter() - started,
                })
                p(f"    [{sentence_id}] {duration:5.2f}s  slot {slot:5.2f}s  "
                  f"asked {rows[-1]['requested_s']:5.2f}s  "
                  f"{duration / slot:.2f}x slot")

    if not rows:
        p("\n!! nothing generated")
        return rows

    if not any(r["instrumented"] for r in rows):
        section("INSTRUMENTATION FAILED")
        p("  The patched infer_batch_process was never called, so IndicF5 does")
        p("  not reach generation through f5_tts.infer.utils_infer on this")
        p("  install. fix_duration was therefore never applied and every")
        p("  duration below came from the byte formula. Nothing here is a")
        p("  measurement. Find the real call path before reading on.")
        return rows

    section("DID fix_duration TAKE EFFECT?")
    p("  The generated span against the slot it was asked for. A parameter that")
    p("  silently does nothing is this repo's known failure mode, so it is")
    p("  checked rather than assumed.")
    p("")
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


def report_rows(rows):
    """
    Everything downstream of synthesis, over rows whose audio already exists.

    Split out so `rescore` can run it against clips already on disk. The first
    phase 0 run produced sound audio and void numbers — the ASR raised on every
    clip (FINDINGS §13) — and re-synthesizing 28 clips to fix a transcription
    bug would have spent most of an hour of GPU re-making files that were never
    wrong.
    """
    # ------------------------------------------------------- did the ASR run?
    # The same guard the instrumentation gets, for the same reason. A run where
    # nothing transcribed still prints a full CONTENT table, a full SEED SPREAD
    # and a VERDICT — every cell nan, every threshold comparison False, so
    # every arm shows zero bad clips and the `latin` control comes back
    # "clean". That is a report that reads as a result.
    scored = [r for r in rows if np.isfinite(r.get("cer", float("nan")))]
    if not scored:
        section("NOTHING WAS TRANSCRIBED")

        # Counts, not a conclusion. The first version of this guard said
        # "every transcript came back empty" when what it had actually
        # observed was the absence of a number, and the two are different
        # facts: a row the loop skipped has no `heard` key at all, a row the
        # ASR returned nothing for has an empty one, and a row that raised has
        # `asr_error`. Guessing between them cost a round trip.
        errors = [r for r in rows if r.get("asr_error")]
        skipped = [r for r in rows if "heard" not in r]
        empty = [r for r in rows
                 if "heard" in r and not (r.get("heard") or "").strip()]
        looped = [r for r in rows if r.get("asr_looped")]

        p(f"  {len(rows)} rows: {len(errors)} raised, {len(skipped)} never "
          f"reached the ASR, {len(empty)} transcribed to nothing, "
          f"{len(looped)} looped")
        p("")

        # The transcript itself, verbatim. Every previous version of this guard
        # printed a count or a category and left the reader to infer the rest,
        # and each time the inference was wrong. If all four counts above are
        # zero then every row has a transcript and the scoring is what failed,
        # and the only way to see which is to look at one.
        sample = rows[0]
        p(f"  first row: {sample['label']} [{sample['index']}]")
        p(f"    asked: {sample.get('text')!r}")
        p(f"    heard: {(sample.get('heard') or '')[:200]!r}")
        p(f"    cer:   {sample.get('cer', 'never set')!r}")
        p("")
        for line in _provenance():
            p(line)
        p("")
        if errors:
            p(f"  first error: {errors[0]['asr_error']}")
        if skipped:
            row = skipped[0]
            p("  A skipped row is one transcribe_outputs passed over, which it")
            p("  does when the row carries no intended text to score against.")
            p(f"  first skipped: {row['label']} [{row['index']}] "
              f"text={row.get('text')!r}")
        if empty:
            row = empty[0]
            exists = Path(row["path"]).exists()
            p(f"  first empty: {row['label']} [{row['index']}] "
              f"path exists {exists}, "
              f"{row.get('actual_s', float('nan')):.2f}s, "
              f"language {row.get('language')!r}")
            p("  Run probe.diagnose_asr(rows) to try that clip three ways.")
        p("")
        p("  The audio is on disk and is worth listening to — synthesis")
        p("  succeeded and fix_duration was applied. But no content number")
        p("  can be computed, so none is printed rather than printed as nan.")
        p(f"  Clips are under {OUT}.")
        return rows

    # ------------------------------------------------------------ script gate
    section("SCRIPT — is the transcript even in the right alphabet?")
    p("  Whisper's language argument is a hint. Indian-accented English coming")
    p("  back in Devanagari would score as every character wrong, which reads")
    p("  as the model failing when it is the ruler that moved.")
    p("")
    for row in rows:
        heard = row.get("heard") or ""
        row["latin"] = script_ratio(heard, "en") if heard else 0.0
        row["wrong_script"] = bool(heard) and row["latin"] < MIN_LATIN_FRACTION
        if row["wrong_script"]:
            row.update({"cer": float("nan"), "extra": float("nan"),
                        "missing": float("nan"), "lead": float("nan")})

    drifted = [r for r in rows if r.get("wrong_script")]
    p(f"  {len(drifted)} of {len(rows)} transcripts came back under "
      f"{MIN_LATIN_FRACTION:.0%} Latin and are not scored for content.")
    for row in drifted[:5]:
        p(f"    {row['label']} [{row['index']}] {row['latin']:.0%} Latin: "
          f"{(row.get('heard') or '')[:60]!r}")

    # --------------------------------------------------------------- content
    section("CONTENT — the gate, before anything else is reported")
    p(f"{'arm':<14}{'n':>3}{'cer':>8}{'extra':>8}{'missing':>9}{'lead':>7}"
      f"{'got/slot':>10}{'bad':>6}")
    p("")

    table = {}
    for label in dict.fromkeys(r["label"] for r in rows):
        items = [r for r in rows if r["label"] == label]
        scored = [r for r in items if np.isfinite(r.get("cer", np.nan))]
        bad = [r for r in items
               if r.get("wrong_script")
               or r.get("asr_looped")
               or r.get("extra", 0) > MAX_EXTRA
               or r.get("missing", 0) > MAX_MISSING
               or r.get("cer", 0) > MAX_CER
               or r.get("lead", 0) > MAX_LEAD]
        table[label] = {
            "cer": mean(scored, "cer"), "extra": mean(scored, "extra"),
            "missing": mean(scored, "missing"), "lead": mean(scored, "lead"),
            "got": float(np.mean([r["actual_s"] / r["slot_s"] for r in items])),
            "bad": len(bad), "n": len(items),
        }
        p(f"{label:<14}{len(items):>3}{table[label]['cer']:>8.3f}"
          f"{table[label]['extra']:>8.3f}{table[label]['missing']:>9.3f}"
          f"{table[label]['lead']:>7.3f}{table[label]['got']:>10.2f}"
          f"{len(bad):>5}/{len(items)}")

    p("")
    p("  Read every CER against the `floor (him)` row, not against zero. That")
    p("  row is his own recording of the same seven sentences through this")
    p("  same Whisper, so it carries the same disagreements the arms are")
    p("  judged on — `fit` heard as `feet`, `to land` as `two land`. An arm")
    p("  sitting at the floor is as good as this measurement can see.")

    section("BY SENTENCE — and whether it was inside the reference")
    p(f"{'arm':<14}{'id':>3}{'ref':>5}{'cer':>8}{'extra':>8}{'lead':>7}"
      f"{'got/slot':>10}")
    p("")
    for row in rows:
        p(f"{row['label']:<14}{row['index']:>3}"
          f"{'  in' if row['in_reference'] else '   -':>5}"
          f"{row.get('cer', float('nan')):>8.3f}"
          f"{row.get('extra', float('nan')):>8.3f}"
          f"{row.get('lead', float('nan')):>7.3f}"
          f"{row['actual_s'] / row['slot_s']:>10.2f}")

    # ------------------------------------------------------- what it heard
    # A CER is a summary of this, and the summary is the part that can be
    # wrong without looking wrong. Seed 0 only, so it stays readable: the
    # other seeds are in the report for spread, not for reading.
    section("WHAT WHISPER HEARD — seed 0")
    p("  Read it against the intended line. Whisper is a fluency prior and")
    p("  will round a garbled clip into a clean sentence (FINDINGS §3b), so")
    p("  a low CER here is necessary and not sufficient. Your ears decide.")
    for row in [r for r in rows if r["seed"] == 0]:
        p("")
        p(f"  {row['label']} [{row['index']}]  cer "
          f"{row.get('cer', float('nan')):.3f}")
        p(f"    asked: {row['text']}")
        p(f"    heard: {(row.get('heard') or '')[:160]}")

    # ------------------------------------------------------------ seed spread
    section("SEED SPREAD — the noise floor every later comparison needs")
    spreads = {}
    for arm in REPORTED_ARMS:
        hypothesis = [r for r in rows if r["arm"] == arm]
        if len({r["seed"] for r in hypothesis}) <= 1:
            continue
        per_seed = {}
        for seed in sorted({r["seed"] for r in hypothesis}):
            items = [r for r in hypothesis if r["seed"] == seed]
            per_seed[seed] = mean(items, "cer")
            p(f"  {arm:<11} seed {seed}   cer {per_seed[seed]:.3f}   "
              f"{sum(1 for r in items if r.get('cer', 0) > MAX_CER)} over threshold")
        values = [v for v in per_seed.values() if np.isfinite(v)]
        if len(values) > 1:
            spreads[arm] = max(values) - min(values)
            p(f"  {arm:<11} spread {spreads[arm]:.3f} in mean cer across seeds.")
        p("")

    spread = max(spreads.values()) if spreads else float("nan")
    if not spreads:
        p("  Only one seed ran; the later phases have no noise floor to read")
        p("  their differences against.")
    else:
        p("  This is the number that makes phases 1 to 4 readable. FINDINGS §14")
        p("  records a seven-configuration sweep whose entire span was 0.046")
        p("  against a scale spanning 0.87 — meaningless, and it was written")
        p("  down as a finding first. Any arm difference smaller than this")
        p("  spread is not a difference.")

    # ---------------------------------------------------------------- verdict
    section("VERDICT")
    control = table.get("latin")
    if control is None:
        p("  The control did not run, so the content check is unverified in")
        p("  this session and nothing below should be believed.")
    elif control["bad"] == 0:
        p("  !! The `latin` control came back CLEAN. FINDINGS §4 measured this")
        p("     configuration producing speech that is not English at all, so a")
        p("     clean result means the harness is broken, not that the model")
        p("     improved. Stop and fix the content check.")
    else:
        p(f"  Control behaves: latin {control['bad']}/{control['n']} clips bad,")
        p(f"  cer {control['cer']:.3f}. The content check fires in this session.")

    p("")
    floor = table.get("floor (him)")
    if floor is None:
        p("  No floor row: fixtures/en_speaker is missing, so every CER below")
        p("  is being read against zero, which no measurement here supports.")
    else:
        p(f"  Floor: his own recording scores cer {floor['cer']:.3f} through")
        p(f"  this same Whisper, {floor['bad']}/{floor['n']} clips over the")
        p("  thresholds. That is the number an arm is trying to reach, not 0.")

    p("")
    arm_cer = {}
    for arm in REPORTED_ARMS:
        arm_rows = [table[l] for l in table if l.split("/")[0] == arm]
        if not arm_rows:
            continue
        worst = max(r["bad"] for r in arm_rows)
        best = min(r["bad"] for r in arm_rows)
        arm_cer[arm] = float(np.mean([r["cer"] for r in arm_rows
                                      if np.isfinite(r["cer"])] or [np.nan]))
        p(f"  {arm:<11}{best}-{worst} bad clips of {arm_rows[0]['n']} "
          f"per seed, mean cer {arm_cer[arm]:.3f}")

    if not arm_cer:
        p("  deva_hand did not run. This session answered nothing.")
    else:
        if {"deva_hand", "deva_ref"} <= set(arm_cer) and np.isfinite(spread):
            # Same audio, same sentences, same seeds — only ref_text differs,
            # so this gap is the reference pair and nothing else. Read it
            # against the spread before reading it at all.
            gap = abs(arm_cer["deva_ref"] - arm_cer["deva_hand"])
            p("")
            p(f"  deva_ref - deva_hand  {gap:.3f} in mean cer, against a seed")
            p(f"  spread of {spread:.3f}. "
              + ("Bigger than the noise." if gap > spread
                 else "Inside the noise — not a difference."))
            p("  Either way this arm is about accent, which CER does not")
            p("  measure. The content numbers are here to show it did not")
            p("  break intelligibility; the answer is in the listening.")

        baseline = arm_cer.get("deva_ref")
        dials = [a for a in DIAL_ARMS if a in arm_cer]
        if dials and baseline is not None and np.isfinite(baseline):
            p("")
            p("  Accent dials, against deva_ref — the baseline they are built on:")
            for dial in dials:
                cost = arm_cer[dial] - baseline
                p(f"    {dial:<13}cer {arm_cer[dial]:.3f}   "
                  f"{cost:+.3f} against deva_ref's {baseline:.3f}")
            p("")
            p("  That column is the price, not the result. These arms spell the")
            p("  loanwords in a way IndicF5 was never trained on, so some content")
            p("  cost is expected and was written down before the run. What it")
            p("  bought is the accent, and only your ears read that. A dial that")
            p("  reaches 6 by breaking the words has not won anything.")
        p("")
        p("  The gate is the continue-the-probe one, not the ship one: content")
        p("  clean, pace near 1.0, and the arm gap bigger than the seed spread.")
        p("  Accent is not asserted here — nothing in this run measures it.")
        p("")
        p("  0 bad clips, cer near the Whisper floor  -> the hypothesis holds.")
        p("        Build src/text/normalize.py and en_to_deva.py and run the")
        p("        automatic arms against this upper bound.")
        p("  intelligible but not Indian by ear       -> the accent metric is")
        p("        now worth building; §3c before any more arms.")
        p("  not intelligible                         -> stop. FINETUNE_PLAN")
        p("        Route B, and none of the metric machinery gets written.")
        p("")
        p("  Whatever the table says, listen to all of them before deciding.")
        p("  Two of this project's findings were caught by ear and by no")
        p("  metric, and Whisper is a fluency prior that can round-trip")
        p("  garbled audio clean (FINDINGS §3b).")

    REPORT.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nwrote {REPORT}")
    return rows


def listen(rows, arms=None, indexes=None):
    """
    His real reading first, then each arm on the same sentence.

    The real clip is the point: this arm is trying to sound like him speaking
    English, and no number in the report reads accent.
    """
    from IPython.display import Audio, display

    # The floor arm *is* his real recording, and that is already played first
    # for every sentence. Including it here would play the same clip twice and
    # invite it to be heard as a generated one.
    picked = [r for r in rows
              if r["arm"] != "floor"
              and (arms is None or r["arm"] in arms)
              and (indexes is None or r["index"] in indexes)]

    for index in sorted({r["index"] for r in picked}):
        group = [r for r in picked if r["index"] == index]
        print(f"\n{'=' * 70}\n[{index}] {group[0]['text']}")
        print(f"     given: {group[0]['given']}")
        if group[0]["in_reference"]:
            print("     (inside the reference clip — the model was handed this)")
        print("=" * 70)

        real = FIXTURES / "en_speaker" / f"{index:02d}.wav"
        if real.exists():
            print(f"\nhim, really saying it   {sf.info(str(real)).duration:.2f}s")
            display(Audio(str(real)))

        for row in sorted(group, key=lambda r: (r["arm"], r["seed"])):
            notes = []
            if row.get("wrong_script"):
                notes.append(f"WRONG SCRIPT ({row.get('latin', 0):.0%} Latin)")
            if row.get("asr_looped"):
                notes.append("UNREADABLE (ASR looped)")
            if row.get("lead", 0) > MAX_LEAD:
                notes.append("PREFIX")
            if row.get("extra", 0) > MAX_EXTRA:
                notes.append("GIBBERISH")
            if row.get("missing", 0) > MAX_MISSING:
                notes.append("CUT SHORT")
            if row.get("cer", 0) > MAX_CER:
                notes.append("MANGLED")

            print(f"\n{row['label']}   {row['actual_s']:.2f}s  "
                  f"{row['actual_s'] / row['slot_s']:.2f}x slot"
                  + ("   " + "  ".join(notes) if notes else ""))
            heard = row.get("heard") or ""
            if heard:
                print(f"  heard: {heard[:200]}")
            display(Audio(str(row["path"])))


if __name__ == "__main__":
    main()
