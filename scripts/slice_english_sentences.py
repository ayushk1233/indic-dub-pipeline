"""
Cut the speaker's real English reading into one clip per fixture sentence.

The transliteration probe needs three things that every version of the plan
scheduled a fresh recording session for, and all three are already on disk:
fixtures/english_speech.wav is fifty seconds of this speaker reading the whole
scripted paragraph, and the seven fixture sentences are sentences inside it.

    slot            how long *he* takes to say this sentence. That is what
                    fix_duration is set from, so the probe asks IndicF5 for a
                    duration a human actually used rather than one derived from
                    a corpus rate. FINDINGS §10 records that the FLEURS-fitted
                    duration model is about 20% off synthesis anyway.
    ceiling         his voice against his voice on clips of the same length as
                    the generated ones. FINDINGS §2: a ceiling measured on 17s
                    pieces must not be applied to 5s clips, and most of the
                    apparent short-segment collapse in earlier runs was the
                    ruler rather than the model.
    content floor   Whisper's own error transcribing him. Synthesis CER means
                    nothing next to zero; it means something next to this.
                    FINDINGS §12 already put word error at 7.9% normalized on
                    this recording, so the floor is not small.

Word timestamps rather than a forced aligner: the sentence boundaries wanted
here are word boundaries, the text is known, and adding Montreal Forced Aligner
for this would be a conda environment and an Indian English acoustic model that
may not exist.

The decoding settings are transcribe_fixtures.py's, for the same reason they
exist there — the output is committed, and a fixture that changes under its own
re-run is not a fixture.

    ./venv/bin/python -m scripts.slice_english_sentences
"""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import soundfile as sf
import yaml
from faster_whisper import WhisperModel

from scripts.transcribe_fixtures import DECODE
from src.text.numbers import spell_numbers

FIXTURES = Path("fixtures")
SOURCE = FIXTURES / "english_speech.wav"
SENTENCES = FIXTURES / "sentences" / "fixture7.json"
OUTPUT_DIR = FIXTURES / "en_speaker"
SLOTS = OUTPUT_DIR / "slots.json"

LANGUAGE = "en"

# Padding kept around each cut, in seconds. A word timestamp lands on the
# decoder's estimate of the boundary, not on the zero crossing, and clipping a
# consonant off the front of a slice would both shorten the slot and give the
# speaker encoder a different sound than the speaker made.
PAD_S = 0.06

# Below this the match between a fixture sentence and the transcript is not a
# recognition error, it is the wrong sentence. He did not read the script word
# for word — fixtures/scripted_text.json says so — and Whisper adds its own
# error on top, so the bar is well under 1.0. It is not near zero either,
# because silently slicing the wrong span would give every downstream metric a
# plausible, wrong ruler.
MIN_MATCH = 0.60


def words_of(segments):
    """Every word in the take, flattened, with its timestamp."""
    out = []
    for segment in segments:
        for word in segment.words or []:
            text = word.word.strip()
            if text:
                out.append({"text": text, "start": word.start, "end": word.end})
    return out


def _key(text):
    """
    What two spellings of the same speech should agree on.

    Case, punctuation and spacing go. Numbers are spelled out because the two
    sides disagree about them systematically rather than occasionally: the
    script says "forty-seven segments" and "Thirty-one fit perfectly", Whisper
    writes "47 segments" and "31 fit perfectly", and on a letters-only
    projection those share no characters at all. Without this, sentence 5 lost
    its subject and the slice came back at 27 characters per second — faster
    than anything this project has ever measured, synthesized or human.

    This is alignment machinery, not the pipeline's number normaliser. That one
    has to decide how an Indian speaker *says* a number; this one only has to
    make two written forms collide.
    """
    text = spell_numbers(text).lower()
    return "".join(c for c in text if c.isalnum())


def locate(words, sentence, search_from=0, lookahead=30):
    """
    Find the span of `words` that the fixture sentence was read as.

    The best-scoring contiguous run of words, searched over start positions and
    lengths. An earlier version took the first and last matching blocks that
    SequenceMatcher reported over the whole remaining transcript, which is not
    the same thing: scattered incidental matches stretched the span across
    neighbouring sentences, and sentence 6 picked up "needed stretching" from a
    sentence that is deliberately not in fixture7.

    Fuzzy rather than exact because both sides are approximate — the speaker
    paraphrased (fixtures/scripted_text.json says as much) and the ASR has its
    own error. The ratio comes back with the span so the caller can refuse a
    match rather than trust it.

    `search_from` moves forward with each sentence: the paragraph is read in
    order, so constraining the search stops a repeated phrase later in the take
    from matching an earlier sentence. `lookahead` is what lets it step over
    the sentences fixture7 excludes.
    """
    target = _key(sentence)
    if not target or search_from >= len(words):
        return None

    keys = [_key(word["text"]) for word in words]

    # A span much longer than the target cannot be a better reading of it, and
    # cutting the search there is what keeps this from being quadratic in the
    # whole take.
    limit = len(target) * 1.6 + 8

    best = None
    for first in range(search_from, min(search_from + lookahead, len(words))):
        accumulated = ""
        for last in range(first, len(words)):
            accumulated += keys[last]
            ratio = SequenceMatcher(None, accumulated, target,
                                    autojunk=False).ratio()
            if best is None or ratio > best["ratio"]:
                best = {"first": first, "last": last, "ratio": ratio}
            if len(accumulated) > limit:
                break

    return best


def reference_window():
    """
    Where the 10 s reference clip sits inside english_speech.wav.

    build_fixtures.py joins the take's speech spans into `_speech.wav` and cuts
    `_reference_short.wav` from the same spans, so with a single speech span the
    reference is simply the opening seconds of the speech file. Returned in the
    speech file's own coordinates.

    This is not bookkeeping. The probe conditions IndicF5 on that reference and
    then asks it to generate these sentences, so any sentence inside the window
    is one the model was handed the audio *and* the transcript for. If an arm
    succeeds only on those, that is leakage rather than a result.
    """
    metadata = json.loads((FIXTURES / "metadata.json").read_text(encoding="utf-8"))
    english = metadata["english"]

    spans = english.get("reference_short_spans") or []
    if not spans or english.get("num_speech_spans") != 1:
        return None

    offset = english["lead_silence_s"]
    return (spans[0][0] - offset, spans[-1][1] - offset)


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"{SOURCE} — run scripts.build_fixtures first")
    if not SENTENCES.exists():
        raise FileNotFoundError(f"{SENTENCES} — the frozen fixture7 set")

    frozen = json.loads(SENTENCES.read_text(encoding="utf-8"))["sentences"]
    window = reference_window()

    cfg = yaml.safe_load(Path("config/pipeline.yaml").read_text(encoding="utf-8"))
    asr = cfg["asr"]
    model = WhisperModel(asr["model"], device=asr["device"],
                         compute_type=asr["compute_type"])

    segments, _ = model.transcribe(str(SOURCE), language=LANGUAGE,
                                   word_timestamps=True, **DECODE)
    words = words_of(segments)
    print(f"{len(words)} words from {SOURCE}")

    audio, rate = sf.read(str(SOURCE), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    source_duration = float(audio.size / rate)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    slots, cursor = [], 0
    for entry in frozen:
        found = locate(words, entry["text"], cursor)

        if found is None or found["ratio"] < MIN_MATCH:
            ratio = found["ratio"] if found else 0.0
            raise RuntimeError(
                f"sentence {entry['id']} matched the transcript at only "
                f"{ratio:.2f} (floor {MIN_MATCH}): {entry['text'][:60]!r}. "
                "Slicing on that would hand every downstream metric a "
                "plausible but wrong span — check the transcript before "
                "lowering the floor."
            )

        start = max(0.0, words[found["first"]]["start"] - PAD_S)
        end = min(source_duration, words[found["last"]]["end"] + PAD_S)
        cursor = found["last"] + 1

        name = f"{entry['id']:02d}.wav"
        sf.write(str(OUTPUT_DIR / name),
                 audio[int(start * rate):int(end * rate)], rate,
                 subtype="PCM_16")

        heard = " ".join(words[i]["text"]
                         for i in range(found["first"], found["last"] + 1))
        # float()/bool() rather than leaving these as whatever numpy handed
        # back: numpy 2 names its scalar bool "bool", so json's refusal reads
        # as though a plain Python bool were unserializable.
        overlap = 0.0
        if window is not None:
            overlap = float(max(0.0, min(end, window[1]) - max(start, window[0])))

        slots.append({
            "id": entry["id"],
            "file": name,
            "text": entry["text"],
            "heard": heard,
            "match": round(found["ratio"], 3),
            "start_s": round(float(start), 3),
            "end_s": round(float(end), 3),
            "duration_s": round(float(end - start), 3),
            "chars": entry["chars"],
            "cps": (round(float(entry["chars"] / (end - start)), 2)
                    if end > start else None),
            "reference_overlap_s": round(overlap, 3),
            "in_reference": bool(overlap > 0.5),
        })
        flag = "  <-- inside the reference" if slots[-1]["in_reference"] else ""
        print(f"  [{entry['id']}] {start:6.2f}-{end:6.2f}s  "
              f"{end - start:5.2f}s  {slots[-1]['cps']:5.1f} cps  "
              f"match {found['ratio']:.2f}{flag}")
        print(f"        heard: {heard}")

    covered = float(sum(s["duration_s"] for s in slots))
    rate_cps = float(sum(s["chars"] for s in slots) / covered) if covered else 0.0

    out = {
        "_note": (
            "One clip per fixture7 sentence, cut from "
            "fixtures/english_speech.wav on Whisper word timestamps by "
            "scripts/slice_english_sentences.py. duration_s is this speaker's "
            "own slot for that sentence and is what fix_duration is set from; "
            "the wav files are the same-language identity ceiling and the "
            "Whisper content floor. `heard` is what the ASR made of the span "
            "and `match` how well that agrees with the scripted text — he did "
            "not read the script word for word, so neither is 1.00 and both "
            "are recorded rather than assumed. `in_reference` marks sentences "
            "that fall inside the 10s reference clip the probe conditions on: "
            "the model is handed that audio and its transcript, so an arm that "
            "succeeds only on those has not been shown to generalise. "
            "measured_cps is the rate inside slots, with the pauses between "
            "sentences excluded, which is the rate fix_duration needs and runs "
            "higher than a whole-take figure."
        ),
        "source": str(SOURCE),
        "source_duration_s": round(source_duration, 3),
        "language": LANGUAGE,
        "pad_s": PAD_S,
        "reference_window_s": ([round(w, 3) for w in window]
                               if window is not None else None),
        "covered_s": round(covered, 3),
        "measured_cps": round(rate_cps, 2),
        "slots": slots,
    }
    SLOTS.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                     encoding="utf-8")

    print(f"\n{covered:.1f}s of {source_duration:.1f}s covered")
    print(f"his English runs {rate_cps:.2f} cps "
          f"against 13.13 measured on FLEURS")
    print(f"wrote {SLOTS}")


if __name__ == "__main__":
    main()
