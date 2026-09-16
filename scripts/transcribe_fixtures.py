"""
Transcribe the reference fixtures.

XTTS conditions on audio alone. IndicF5 conditions on audio together with what
was said in it, so a comparison between them needs a transcript of exactly the
25 seconds each model is handed — not of the whole take, and not the scripted
text, because the reference is a few spans cut out of the middle and the
speaker did not read the script word for word anyway.

Written as a separate script rather than folded into build_fixtures.py so that
rebuilding the audio does not require reloading a 3 GB ASR model, and so the
transcripts stay reviewable as text under version control.

This does not go through FasterWhisperBackend. The pipeline's ASR settings are
tuned for transcribing a whole lecture; what is needed here is a transcript
that comes out the same every time, because the result is committed and a
fixture that changes under its own re-run is not a fixture. The first two runs
of an earlier version of this script disagreed about the tail of the Hindi
clip, which is what prompted the explicit decoding settings below.

    ./venv/bin/python -m scripts.transcribe_fixtures
"""

import json
from pathlib import Path

import yaml
from faster_whisper import WhisperModel


FIXTURES = Path("fixtures")
OUTPUT = FIXTURES / "reference_text.json"

REFERENCES = {
    "english": ("english_reference.wav", "en"),
    "hindi": ("hindi_reference.wav", "hi"),
}

# Greedy, with no temperature fallback ladder. Whisper's default is to retry at
# rising temperatures when a decode trips the compression-ratio guard, and the
# reference clips trip it because they end mid-sentence. Those retries sample,
# so the run is not reproducible and the sampled output is where the
# repetition loops came from in the first place.
DECODE = {
    "beam_size": 5,
    "temperature": 0.0,
    # The clip is one continuous passage, but carrying decoded text forward as
    # a prompt is the other half of what drives a loop: once the model emits a
    # repeated phrase it conditions on that repetition and continues it.
    "condition_on_previous_text": False,
}

# A trailing phrase repeated this many times is a decoder loop, not speech.
REPEAT_RUN = 3

# Longest phrase treated as a possible loop. Whisper loops on short units;
# above roughly four words a repetition is far more likely to be real.
MAX_REPEAT_WORDS = 4


def strip_repetition_tail(text: str) -> str:
    """
    Drop a hallucinated repetition loop from the end of a transcript.

    Whisper ends a clip that stops mid-sentence by looping on a phrase. Two
    runs of this script produced `झाल झाल झाल झाल` and `अजय को अजय को अजय को
    अजय को` on the same audio — a repeated word the first time and a repeated
    pair the second, so matching single tokens is not enough.

    The whole run goes, not just enough of it to break the threshold: a phrase
    emitted four times is hallucination, and leaving two behind would still be
    describing audio that does not contain them. Only the tail is examined, so
    a genuine repetition earlier in the speech is untouched. Longer phrases are
    tried first, because a repeated pair also looks like a repeated word.
    """
    words = text.split()

    for size in range(MAX_REPEAT_WORDS, 0, -1):
        if len(words) < size * REPEAT_RUN:
            continue

        phrase = words[-size:]
        count = 0

        while words[len(words) - (count + 1) * size: len(words) - count * size] == phrase:
            count += 1

        if count >= REPEAT_RUN:
            return " ".join(words[: len(words) - count * size])

    return text


def transcribe(model: WhisperModel, path: Path, language: str) -> str:
    segments, _ = model.transcribe(str(path), language=language, **DECODE)
    joined = " ".join(segment.text.strip() for segment in segments).strip()

    return strip_repetition_tail(joined)


def main() -> None:
    cfg = yaml.safe_load(Path("config/pipeline.yaml").read_text(encoding="utf-8"))
    asr = cfg["asr"]

    model = WhisperModel(
        asr["model"], device=asr["device"], compute_type=asr["compute_type"]
    )

    out = {
        "_note": (
            "ASR transcripts of the reference clips, for models that condition "
            "on audio and text together. Whisper output, so it carries ASR "
            "error; read it before trusting it. Produced by "
            "scripts/transcribe_fixtures.py with greedy decoding, so re-running "
            "it should reproduce this file exactly."
        )
    }

    for name, (filename, language) in REFERENCES.items():
        path = FIXTURES / filename
        if not path.exists():
            raise FileNotFoundError(f"{path} — run scripts.build_fixtures first")

        text = transcribe(model, path, language)
        out[name] = {"file": filename, "language": language, "text": text}

        print(f"{name} ({language}), {len(text)} chars")
        print(f"  {text}\n")

    OUTPUT.write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
