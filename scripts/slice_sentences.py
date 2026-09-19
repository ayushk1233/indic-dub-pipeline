"""
Cut a speaker's real reading into one clip per frozen fixture sentence.

Written for English first and generalised when hi -> hi needed the same three
things. Both takes are the same speaker reading the same paragraph in the two
languages, so everything here is one `Take` away from being shared:

    slot            how long *he* takes to say this sentence. That is what
                    fix_duration is set from, so a probe asks IndicF5 for a
                    duration a human actually used rather than one derived from
                    a corpus rate. FINDINGS §10 records that the FLEURS-fitted
                    duration model is about 20% off synthesis anyway.
    ceiling         his voice against his voice on clips of the same length as
                    the generated ones. FINDINGS §2: a ceiling measured on 17s
                    pieces must not be applied to 5s clips, and most of the
                    apparent short-segment collapse in earlier runs was the
                    ruler rather than the model.
    content floor   Whisper's own error transcribing him. Synthesis CER means
                    nothing next to zero; it means something next to this. The
                    floor is not small — FINDINGS §12 puts word error at 7.9%
                    normalized on the English take, and Whisper is weaker on
                    Hindi than on English, so the Hindi floor should be
                    expected to sit higher and must be read before any Hindi
                    arm is called good or bad.

Word timestamps rather than a forced aligner: the sentence boundaries wanted
here are word boundaries, the text is known, and adding Montreal Forced Aligner
for this would be a conda environment and an Indian English acoustic model that
may not exist.

The decoding settings are transcribe_fixtures.py's, for the same reason they
exist there — the output is committed, and a fixture that changes under its own
re-run is not a fixture.

    ./venv/bin/python -m scripts.slice_english_sentences
    ./venv/bin/python -m scripts.slice_hindi_sentences
"""

import json
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import soundfile as sf
import yaml
from faster_whisper import WhisperModel

from scripts.transcribe_fixtures import DECODE
from src.text.numbers import spell_numbers

FIXTURES = Path("fixtures")

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

# What each take's rate is worth reading against, measured on FLEURS.
NATURAL_CPS = {"en": 13.13, "hi": 10.81}


@dataclass(frozen=True)
class Take:
    """One recorded reading, and where its pieces live."""

    language: str
    source: Path
    sentences: Path
    output_dir: Path
    metadata_key: str

    @property
    def slots(self):
        return self.output_dir / "slots.json"


EN = Take(
    language="en",
    source=FIXTURES / "english_speech.wav",
    sentences=FIXTURES / "sentences" / "fixture7.json",
    output_dir=FIXTURES / "en_speaker",
    metadata_key="english",
)

HI = Take(
    language="hi",
    source=FIXTURES / "hindi_speech.wav",
    sentences=FIXTURES / "sentences" / "fixture7_hi.json",
    output_dir=FIXTURES / "hi_speaker",
    metadata_key="hindi",
)

TAKES = {"en": EN, "hi": HI}


def words_of(segments):
    """Every word in the take, flattened, with its timestamp."""
    out = []
    for segment in segments:
        for word in segment.words or []:
            text = word.word.strip()
            if text:
                out.append({"text": text, "start": word.start, "end": word.end})
    return out


def _key(text, language=None):
    """
    What two spellings of the same speech should agree on.

    Case, punctuation and spacing go. Numbers are spelled out because the two
    sides disagree about them systematically rather than occasionally: the
    script says "forty-seven segments" and "Thirty-one fit perfectly", Whisper
    writes "47 segments" and "31 fit perfectly", and on a letters-only
    projection those share no characters at all. Without this, sentence 5 lost
    its subject and the slice came back at 27 characters per second — faster
    than anything this project has ever measured, synthesized or human. Hindi
    has the same disagreement, with no shared root at all between `47` and
    `सैंतालीस`.

    NFC first: the nukta consonants have a precomposed and a decomposed form
    (फ़ is U+095E or फ + U+093C) and Whisper does not always pick the one the
    fixture holds. Two spellings of the same letter would otherwise be scored
    as different letters, on exactly the words this corpus is full of.

    This is alignment machinery, not the pipeline's number normaliser. That one
    has to decide how an Indian speaker *says* a number; this one only has to
    make two written forms collide.
    """
    text = unicodedata.normalize("NFC", text or "")
    text = spell_numbers(text, language).lower()
    return "".join(c for c in text if c.isalnum())


def locate(words, sentence, search_from=0, lookahead=30, language=None):
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
    the sentences the fixture excludes.
    """
    target = _key(sentence, language)
    if not target or search_from >= len(words):
        return None

    keys = [_key(word["text"], language) for word in words]

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


def reference_window(take):
    """
    Where the 10 s reference clip sits inside the take's speech file.

    build_fixtures.py joins the take's speech spans into `_speech.wav` and cuts
    `_reference_short.wav` from the same spans, so with a single speech span the
    reference is simply the opening seconds of the speech file. Returned in the
    speech file's own coordinates.

    This is not bookkeeping. A probe conditions IndicF5 on that reference and
    then asks it to generate these sentences, so any sentence inside the window
    is one the model was handed the audio *and* the transcript for. If an arm
    succeeds only on those, that is leakage rather than a result.
    """
    metadata = json.loads((FIXTURES / "metadata.json").read_text(encoding="utf-8"))
    entry = metadata[take.metadata_key]

    spans = entry.get("reference_short_spans") or []
    if not spans or entry.get("num_speech_spans") != 1:
        return None

    offset = entry["lead_silence_s"]
    return (spans[0][0] - offset, spans[-1][1] - offset)


def run(take) -> None:
    if not take.source.exists():
        raise FileNotFoundError(f"{take.source} — run scripts.build_fixtures first")
    if not take.sentences.exists():
        raise FileNotFoundError(f"{take.sentences} — the frozen sentence set")

    frozen = json.loads(take.sentences.read_text(encoding="utf-8"))["sentences"]
    window = reference_window(take)

    cfg = yaml.safe_load(Path("config/pipeline.yaml").read_text(encoding="utf-8"))
    asr = cfg["asr"]
    model = WhisperModel(asr["model"], device=asr["device"],
                         compute_type=asr["compute_type"])

    segments, _ = model.transcribe(str(take.source), language=take.language,
                                   word_timestamps=True, **DECODE)
    words = words_of(segments)
    print(f"{len(words)} words from {take.source}")

    audio, rate = sf.read(str(take.source), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    source_duration = float(audio.size / rate)

    take.output_dir.mkdir(parents=True, exist_ok=True)

    slots, cursor = [], 0
    for entry in frozen:
        found = locate(words, entry["text"], cursor, language=take.language)

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
        sf.write(str(take.output_dir / name),
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
    natural = NATURAL_CPS.get(take.language)

    out = {
        "_note": (
            f"One clip per sentence in {take.sentences.name}, cut from "
            f"{take.source} on Whisper word timestamps by "
            "scripts/slice_sentences.py. duration_s is this speaker's "
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
            "higher than a whole-take figure. Characters per second is not "
            "comparable across scripts — Devanagari writes a syllable in fewer "
            "characters than Latin does — so read cps against this take's own "
            "natural_cps and never against the other take's."
        ),
        "source": str(take.source),
        "source_duration_s": round(source_duration, 3),
        "language": take.language,
        "pad_s": PAD_S,
        "natural_cps": natural,
        "reference_window_s": ([round(w, 3) for w in window]
                               if window is not None else None),
        "covered_s": round(covered, 3),
        "measured_cps": round(rate_cps, 2),
        "slots": slots,
    }
    take.slots.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")

    print(f"\n{covered:.1f}s of {source_duration:.1f}s covered")
    print(f"his {take.language} runs {rate_cps:.2f} cps "
          f"against {natural} measured on FLEURS")
    print(f"wrote {take.slots}")


def main(language="en") -> None:
    run(TAKES[language])


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "en")
