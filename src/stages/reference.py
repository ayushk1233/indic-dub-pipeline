"""
Build the voice reference XTTS clones from.

This is a small file with an outsized effect on output quality. The first real
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
import subprocess
from pathlib import Path


# XTTS conditions at 22.05 kHz and generates at 24 kHz. Cutting the reference
# at 24 kHz means no upsampling anywhere in the chain.
REFERENCE_SAMPLE_RATE = 24000

# Leave a little room below full scale. Normalizing to exactly 0 dBFS invites
# clipping in anything downstream that resamples.
TARGET_PEAK_DBFS = -1.0

# Pad the span outward. Segment boundaries sit at silence edges, and clipping
# a reference tight to the first phoneme loses the speaker's onset.
SPAN_PADDING_S = 0.25

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
    span: tuple[float, float],
    output_path: Path,
    sample_rate: int = REFERENCE_SAMPLE_RATE,
) -> Path:
    """
    Cut `span` out of `source_media` as a normalized mono reference clip.

    Peak normalization rather than loudness normalization: it is a single
    constant gain, so it cannot alter the dynamics the speaker encoder reads.
    """
    source_media = Path(source_media)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start, end = span
    start = max(start - SPAN_PADDING_S, 0.0)
    duration = (end + SPAN_PADDING_S) - start

    if duration <= 0:
        raise ValueError(f"Empty reference span: {span}")

    def cut(filters: str | None) -> subprocess.CompletedProcess:
        command = [
            "ffmpeg", "-hide_banner", "-nostdin", "-y",
            "-ss", f"{start:.3f}",
            "-t", f"{duration:.3f}",
            "-i", str(source_media),
            "-vn",
            "-ac", "1",
            "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
        ]
        if filters:
            command += ["-af", filters]
        command.append(str(output_path))
        return _run(command)

    process = cut(None)

    if process.returncode != 0:
        raise RuntimeError(
            f"Could not cut a reference from {source_media}: {process.stderr[-500:]}"
        )

    peak = measure_peak_dbfs(output_path)

    if peak is None or peak >= TARGET_PEAK_DBFS:
        return output_path

    process = cut(f"volume={TARGET_PEAK_DBFS - peak:.2f}dB")

    if process.returncode != 0:
        raise RuntimeError(
            f"Could not normalize the reference: {process.stderr[-500:]}"
        )

    return output_path
