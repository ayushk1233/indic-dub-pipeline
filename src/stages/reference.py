"""
Build the voice reference the TTS model clones from.

This is a small file with an outsized effect on output quality, and **its
correct settings differ by model** — see REFERENCE_SECONDS and
REFERENCE_CEILING_S below. The history here is XTTS's, because XTTS is what
first exposed each fault; IndicF5 is what ships (FINDINGS §9). The first real
GPU run cloned from a 16 kHz ASR chunk peaking at 0.21 and scored a mean
speaker similarity of 0.478, against a floor of 0.75. Two things were wrong
with that file and both are fixed here rather than on the GPU side, because
the reference is built locally and travels inside the bundle.

Sample rate: XTTS loads conditioning audio at 22.05 kHz. A 16 kHz reference is
upsampled to get there, so everything above 8 kHz is missing. Sibilance and
much of what makes a voice recognizable live in that band.

Level: conditioning is not level-invariant, and a clip peaking at a fifth of
full scale is a quiet, noisy input to the speaker encoder.
"""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path


# XTTS conditions at 22.05 kHz and generates at 24 kHz. Cutting the reference
# at 24 kHz means no upsampling anywhere in the chain.
REFERENCE_SAMPLE_RATE = 24000

# Leave a little room below full scale. Normalizing to exactly 0 dBFS invites
# clipping in anything downstream that resamples.
TARGET_PEAK_DBFS = -1.0

# How much reference audio to aim for. XTTS conditions on what it is handed,
# and 8 seconds proved far too little. Above roughly 30 seconds the returns
# flatten and max_ref_length would truncate it anyway.
TARGET_REFERENCE_S = 25.0

# **This length is model-specific and 25 s is actively wrong for IndicF5.**
#
# IndicF5 clips reference audio longer than 15 s inside
# `preprocess_ref_audio_text` and **never truncates `ref_text` to match**
# (FINDINGS §5d). A 25 s reference therefore hands the model 15 s of audio
# described by 25 s of transcript, which is the exact mismatch FINDINGS §5
# identifies behind the original gibberish. §1's shipping configuration is a
# 10 s clip for precisely this reason, and it is measured: 93% of the
# calibrated identity scale, 0 prefixes, pace 1.00x.
#
# Keyed on the same string as `tts.model` in config/pipeline.yaml, lowercased.
REFERENCE_SECONDS = {
    "coqui/xtts_v2": TARGET_REFERENCE_S,
    "ai4bharat/indicf5": 10.0,
}


# A hard limit, as opposed to the target above. Exceeding the target wastes
# reference; exceeding this is *wrong*.
#
# IndicF5 clips reference audio past 15 s and never truncates `ref_text` to
# match (FINDINGS §5d), so an over-long clip arrives described by a transcript
# covering audio the model cannot hear — §5's gibberish. 14 s leaves room for
# SPAN_PADDING_S and for ffmpeg landing a frame long.
#
# XTTS has no such cliff: `max_ref_length` truncates conditioning cleanly and
# the transcript is not used at all. So it has no ceiling here.
REFERENCE_CEILING_S = {
    "ai4bharat/indicf5": 14.0,
}

# Pad the span outward. Segment boundaries sit at silence edges, and clipping
# a reference tight to the first phoneme loses the speaker's onset. Every span
# costs 2 * this in the finished file, which is why span selection has to
# account for it before it can respect a ceiling.
SPAN_PADDING_S = 0.25


def reference_ceiling(model: str | None) -> float | None:
    """The length past which this model's reference is wrong, not just long."""
    return REFERENCE_CEILING_S.get((model or "").strip().lower())


def reference_seconds(model: str | None) -> float:
    """
    How much reference audio to cut, for the model that will consume it.

    Unknown models get the XTTS default rather than an error: a reference of
    the wrong length degrades output, while refusing to build one at all
    fails the export. The caller that cares should pass a known model.
    """
    return REFERENCE_SECONDS.get((model or "").strip().lower(), TARGET_REFERENCE_S)

_MAX_VOLUME_RE = re.compile(r"max_volume:\s*(-?[0-9.]+)\s*dB")


def _run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True)


def measure_peak_dbfs(path: Path) -> float | None:
    """
    Peak level of a file in dBFS, via FFmpeg's volumedetect.

    Returns None when FFmpeg does not report one, which is not an error worth
    failing an export over.
    """
    process = _run(
        [
            "ffmpeg", "-hide_banner", "-nostdin",
            "-i", str(path),
            "-af", "volumedetect",
            "-f", "null", "-",
        ]
    )

    match = _MAX_VOLUME_RE.search(process.stderr)

    return float(match.group(1)) if match else None


def build_reference(
    source_media: str | Path,
    spans: tuple[float, float] | list[tuple[float, float]],
    output_path: Path,
    sample_rate: int = REFERENCE_SAMPLE_RATE,
) -> Path:
    """
    Cut `spans` out of `source_media` into one normalized mono reference clip.

    Several spans are concatenated in the order given. A single tuple is
    accepted for the one-span case.

    Length is the reason this takes a list. XTTS conditions on what it is
    given, and a short clip carries little of a speaker's range. The first two
    GPU runs cloned from 7.96s, of which a third was silence, and scored a
    mean speaker similarity of 0.53 against a floor of 0.75 with every listener
    complaint naming the same thing: the voice has no richness.

    Peak normalization rather than loudness normalization, applied once to the
    joined clip: it is a single constant gain, so it cannot alter the dynamics
    the speaker encoder reads, and doing it after the join keeps the relative
    level between spans intact.
    """
    source_media = Path(source_media)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(spans, tuple) and len(spans) == 2 and not isinstance(spans[0], tuple):
        spans = [spans]

    if not spans:
        raise ValueError("No reference spans given")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        pieces = []

        for index, (start, end) in enumerate(spans):
            begin = max(start - SPAN_PADDING_S, 0.0)
            duration = (end + SPAN_PADDING_S) - begin

            if duration <= 0:
                raise ValueError(f"Empty reference span: {(start, end)}")

            piece = tmp_dir / f"piece_{index:03d}.wav"
            process = _cut(source_media, begin, duration, piece, sample_rate)

            if process.returncode != 0:
                raise RuntimeError(
                    f"Could not cut a reference from {source_media}: "
                    f"{process.stderr[-500:]}"
                )

            pieces.append(piece)

        joined = tmp_dir / "joined.wav"
        _concatenate(pieces, joined, sample_rate)

        peak = measure_peak_dbfs(joined)
        gain = None if peak is None or peak >= TARGET_PEAK_DBFS else TARGET_PEAK_DBFS - peak

        process = _run(
            [
                "ffmpeg", "-hide_banner", "-nostdin", "-y",
                "-i", str(joined),
                "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le",
            ]
            + (["-af", f"volume={gain:.2f}dB"] if gain is not None else [])
            + [str(output_path)]
        )

        if process.returncode != 0:
            raise RuntimeError(
                f"Could not normalize the reference: {process.stderr[-500:]}"
            )

    return output_path


def _cut(source, start, duration, output_path, sample_rate):
    return _run(
        [
            "ffmpeg", "-hide_banner", "-nostdin", "-y",
            "-ss", f"{start:.3f}",
            "-t", f"{duration:.3f}",
            "-i", str(source),
            "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le",
            str(output_path),
        ]
    )


def _concatenate(pieces: list[Path], output_path: Path, sample_rate: int) -> None:
    """
    Join cut pieces end to end.

    The concat demuxer rather than the concat filter, because every piece was
    written here with identical parameters, so a stream copy is exact.
    """
    if len(pieces) == 1:
        shutil.copyfile(pieces[0], output_path)
        return

    listing = output_path.parent / "pieces.txt"
    listing.write_text(
        "".join(f"file '{p.name}'\n" for p in pieces), encoding="utf-8"
    )

    process = _run(
        [
            "ffmpeg", "-hide_banner", "-nostdin", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(listing),
            "-c", "copy",
            str(output_path),
        ]
    )

    if process.returncode != 0:
        raise RuntimeError(f"Could not join reference spans: {process.stderr[-500:]}")
