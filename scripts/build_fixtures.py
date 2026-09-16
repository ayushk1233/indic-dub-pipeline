"""
Derive small, committable audio fixtures from a recording take.

The source .mov files are around 100 MB each and gitignored. Colab needs the
audio, not the video, and needs it to arrive with a `git pull` rather than a
manual upload into every fresh runtime. Three files come out of each take:

  <name>_reference.wav   what the model clones from. Built through the same
                         Segmenter and build_reference path the pipeline uses
                         at export, so the English fixture is byte-comparable
                         with what a real run produces.
  <name>_speech.wav      every speech span joined, room tone trimmed. For the
                         Hindi take this is the gold standard: a real human
                         saying the words the dub is trying to say.
  <name>_room_tone.wav   the silence recorded before speaking, at the take's
                         original gain.

The room tone is deliberately the one file that is NOT normalized. Its whole
purpose is to carry the noise floor at the level it was recorded, so gain
applied to it would destroy the measurement it exists for. The gain applied
to the other two is written into metadata.json instead, which is enough to
reconstruct the relationship between them.
"""

import argparse
import json
import subprocess
import wave
from pathlib import Path

import numpy as np

from src.stages.preprocessing.segmentation import Segmenter
from src.stages.reference import (
    REFERENCE_SAMPLE_RATE,
    TARGET_PEAK_DBFS,
    TARGET_REFERENCE_S,
    build_reference,
    measure_peak_dbfs,
)


FIXTURES = Path("fixtures")

# Speech sits around -25 dBFS on these takes and the noise floor around -55,
# so the default -30dB threshold straddles nothing. -45 splits them cleanly
# while still catching the quiet ends of words.
SILENCE_THRESHOLD = "-45dB"
SILENCE_MIN_S = 0.4

# Sensitivity has a cost. At -45 dB the recorded room tone is not one silence
# but five, split by transients a few milliseconds long: the click of the
# record button settling, a breath before the first word. Each gap between
# them then reads as speech, and the first 1.2 seconds of room tone gets
# handed to the reference builder as if the speaker were talking. Anything
# shorter than this between two silences is not a word.
MIN_SPEECH_GAP_S = 0.2

# Coalescing is still not enough on its own. Nothing says the recorded tone
# contains a clean 0.4-second window before the first transient, and when it
# does not, the opening stretch is never marked silent at all and survives as
# a speech span made entirely of room tone. Level settles it: real speech on
# these takes runs 25 dB or more above the floor, so a span sitting near the
# floor is not speech whatever the gap structure says.
SPEECH_FLOOR_MARGIN_DB = 20.0

# Leave the first second of the take out of the room tone. Auto gain control,
# if the device applies any, is still settling there, and the handling bump
# from pressing record lands inside it. Silence found later in the take needs
# no such allowance, only a trim clear of the speech on either side.
ROOM_TONE_SKIP_S = 1.0
ROOM_TONE_EDGE_S = 0.1
ROOM_TONE_MIN_S = 1.0
ROOM_TONE_MAX_S = 6.0

# A second, shorter reference, for models that derive output duration from the
# reference rather than only cloning timbre from it.
#
# IndicF5 estimates how long a generated sentence should be from the ratio of
# its UTF-8 byte length to the reference transcript's, scaled by the reference
# audio's duration. It also clips the reference audio internally, and measured
# against the 25-second clips it used only about 12 to 14 seconds of them while
# still using the whole transcript. That makes it believe the speaker talks
# twice as fast as he does: every Hindi sentence came back at 0.48x the
# duration natural Hindi needs, with a standard deviation of 0.0 across seven
# sentences. Ten seconds sits below any clipping threshold observed, so the
# ratio is computed against audio the model actually used.
TARGET_SHORT_REFERENCE_S = 10.0


def probe_duration(path: Path) -> float:
    process = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(process.stdout.strip())


def coalesce(silences: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    Join silence runs separated by less than MIN_SPEECH_GAP_S.
    """
    merged: list[tuple[float, float]] = []

    for start, end in sorted(silences):
        if merged and start - merged[-1][1] < MIN_SPEECH_GAP_S:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    return merged


def _mono(path: Path) -> np.ndarray:
    """
    Decode the take once, at the fixture rate, for level measurement.
    """
    tmp = FIXTURES / ".probe.wav"
    FIXTURES.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-i", str(path),
         "-vn", "-ac", "1", "-ar", str(REFERENCE_SAMPLE_RATE),
         "-c:a", "pcm_s16le", str(tmp)],
        capture_output=True, text=True, check=True,
    )

    with wave.open(str(tmp), "rb") as handle:
        audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)

    tmp.unlink(missing_ok=True)

    return audio.astype(np.float32) / 32768.0


def _level_db(audio: np.ndarray, start: float, end: float) -> float:
    piece = audio[int(start * REFERENCE_SAMPLE_RATE):int(end * REFERENCE_SAMPLE_RATE)]

    if piece.size == 0:
        return -120.0

    return 20.0 * float(np.log10(max(float(np.sqrt((piece ** 2).mean())), 1e-9)))


def speech_spans(path: Path) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Returns (speech spans, silence runs), both already coalesced and filtered.
    """
    segmenter = Segmenter()
    silences = coalesce(
        segmenter.detect_silences(
            str(path), noise=SILENCE_THRESHOLD, duration=SILENCE_MIN_S
        )
    )
    spans = segmenter.build_segments(silences, probe_duration(path))

    if not spans:
        return [], silences

    audio = _mono(path)
    floor = min(_level_db(audio, s, e) for s, e in silences) if silences else -120.0
    keep = [span for span in spans
            if _level_db(audio, *span) - floor >= SPEECH_FLOOR_MARGIN_DB]

    dropped = len(spans) - len(keep)
    if dropped:
        print(f"  dropped {dropped} span(s) sitting within "
              f"{SPEECH_FLOOR_MARGIN_DB:.0f} dB of the {floor:.1f} dBFS noise floor")

    return keep, silences


def reference_spans(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    Longest first up to the target, then back into time order.

    Same rule as PipelineRunner._reference_spans, with one addition: a span
    longer than the remaining budget is cut to fit rather than taken whole.
    The runner takes it whole, which on a take with one 50-second stretch of
    continuous speech produces a 50-second reference against a 25-second
    target. XTTS would truncate it at max_ref_length anyway, but then the
    truncation point is the model's choice rather than ours, and the file
    carried to the GPU is twice the size it needs to be.
    """
    chosen: list[tuple[float, float]] = []
    total = 0.0

    for start, end in sorted(spans, key=lambda s: s[1] - s[0], reverse=True):
        remaining = TARGET_REFERENCE_S - total

        if remaining <= 0:
            break

        end = min(end, start + remaining)
        chosen.append((start, end))
        total += end - start

    return sorted(chosen)


def cut_raw(source: Path, start: float, duration: float, output: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-y",
         "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(source),
         "-vn", "-ac", "1", "-ar", str(REFERENCE_SAMPLE_RATE),
         "-c:a", "pcm_s16le", str(output)],
        capture_output=True, text=True, check=True,
    )


def truncate_spans(
    spans: list[tuple[float, float]], budget: float
) -> list[tuple[float, float]]:
    """
    Take spans in order until `budget` seconds are covered, cutting the last.
    """
    out: list[tuple[float, float]] = []
    total = 0.0

    for start, end in spans:
        if total >= budget:
            break
        end = min(end, start + (budget - total))
        if end > start:
            out.append((start, end))
            total += end - start

    return out


def build(source: Path, name: str, transcript: str | None) -> dict:
    FIXTURES.mkdir(parents=True, exist_ok=True)

    spans, silences = speech_spans(source)

    if not spans:
        raise SystemExit(f"No speech found in {source} at {SILENCE_THRESHOLD}")

    chosen = reference_spans(spans)

    reference = build_reference(source, chosen, FIXTURES / f"{name}_reference.wav")
    speech = build_reference(source, spans, FIXTURES / f"{name}_speech.wav")

    # Cut from the same spans rather than trimmed off the finished reference,
    # so the short clip is built by the same path and carries the same gain.
    short_spans = truncate_spans(chosen, TARGET_SHORT_REFERENCE_S)
    short = build_reference(
        source, short_spans, FIXTURES / f"{name}_reference_short.wav"
    )

    # Room tone is the longest silence in the take, which on these recordings
    # is the stretch before the first word. Taking the longest rather than
    # assuming it comes first means a take that opens straight into speech
    # still yields usable tone from its tail.
    lead = spans[0][0]
    tone_path = None
    longest = max(silences, key=lambda s: s[1] - s[0], default=None)

    if longest is not None:
        begin = max(longest[0] + ROOM_TONE_EDGE_S, ROOM_TONE_SKIP_S)
        span_s = min(longest[1] - ROOM_TONE_EDGE_S - begin, ROOM_TONE_MAX_S)

        if span_s >= ROOM_TONE_MIN_S:
            tone_path = FIXTURES / f"{name}_room_tone.wav"
            cut_raw(source, begin, span_s, tone_path)

    source_peak = measure_peak_dbfs(source)

    entry = {
        "source": source.name,
        "source_duration_s": round(probe_duration(source), 3),
        "source_peak_dbfs": source_peak,
        "normalized_to_dbfs": TARGET_PEAK_DBFS,
        "gain_applied_db": (
            round(TARGET_PEAK_DBFS - source_peak, 2)
            if source_peak is not None else None
        ),
        "num_speech_spans": len(spans),
        "speech_s": round(sum(e - s for s, e in spans), 2),
        "reference_spans": [[round(s, 3), round(e, 3)] for s, e in chosen],
        "reference_s": round(sum(e - s for s, e in chosen), 2),
        "reference_short_spans": [[round(s, 3), round(e, 3)] for s, e in short_spans],
        "reference_short_s": round(sum(e - s for s, e in short_spans), 2),
        "lead_silence_s": round(lead, 3),
        "files": {
            "reference": reference.name,
            "reference_short": short.name,
            "speech": speech.name,
            "room_tone": tone_path.name if tone_path else None,
        },
        "scripted_text": transcript,
    }

    print(f"\n{name}")
    print(f"  {len(spans)} speech spans, {entry['speech_s']}s of speech")
    print(f"  reference {entry['reference_s']}s from {len(chosen)} spans -> {reference}")
    print(f"  speech    {entry['speech_s']}s -> {speech}")
    print(f"  room tone {'-> ' + str(tone_path) if tone_path else 'none (take opens in speech)'}")

    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--english", default="english.mov")
    parser.add_argument("--hindi", default="hindi.mov")
    parser.add_argument("--transcripts", default="fixtures/scripted_text.json",
                        help="JSON with 'en' and 'hi' keys holding what was read aloud.")
    args = parser.parse_args()

    scripted = {}
    scripted_path = Path(args.transcripts)

    if scripted_path.exists():
        scripted = json.loads(scripted_path.read_text(encoding="utf-8"))

    metadata = {}

    for name, source in (("english", args.english), ("hindi", args.hindi)):
        path = Path(source)
        if not path.exists():
            print(f"skipping {name}: {path} not found")
            continue
        key = "en" if name == "english" else "hi"
        metadata[name] = build(path, name, scripted.get(key))

    out = FIXTURES / "metadata.json"
    out.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
