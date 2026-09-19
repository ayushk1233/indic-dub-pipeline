"""
The short video the media-dependent tests run against, built outside the repo.

Six test modules need a file with a video stream, an audio stream and known
timings. That used to be `test.mp4` in the repo root — untracked, covered by
`.gitignore`'s `*.mp4`, and therefore never actually part of the repository.
It worked until the file was deleted, at which point fourteen tests across
preprocessing, reference building and remux failed with ffprobe errors that
named neither the missing file nor each other.

So the suite builds its own, **in the system temp directory rather than the
project**: 20 seconds, five quiet 2-second tones separated by 2 seconds of
silence.

Three details are load-bearing:

  - *Alternating*, because `Segmenter` finds speech by inverting FFmpeg's
    `silencedetect`. A continuous tone yields no segments and the
    preprocessing tests would fail differently rather than pass.
  - *Quiet* (about -14 dBFS), because `build_reference` peak-normalizes to
    TARGET_PEAK_DBFS. A full-scale tone makes that a no-op and the test
    asserting it would pass without exercising anything.
  - *Outside the repo*, so no stray video reappears in the project on every
    test run. The two recordings that matter are english.mov and hindi.mov.

It is cached across runs and rebuilt only if missing. This is a floor under
the suite, not a fixture worth measuring anything on — nothing in FINDINGS
should ever be read off it.
"""

import subprocess
import tempfile
from pathlib import Path

MEDIA_DIR = Path(tempfile.gettempdir()) / "indic_dub_test_media"
TEST_VIDEO = str(MEDIA_DIR / "test.mp4")

DURATION_S = 20
TONE_HZ = 220
TONE_AMPLITUDE = 0.2
PERIOD_S = 4
BURST_S = 2


def build(path: Path = None) -> Path:
    """Build the clip if it is not already there. Returns its path."""
    path = Path(path or TEST_VIDEO)

    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)

    process = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostdin", "-y",
            "-f", "lavfi",
            "-i", f"color=c=black:s=320x240:r=10:d={DURATION_S}",
            "-f", "lavfi",
            "-i",
            f"aevalsrc='{TONE_AMPLITUDE}*sin(2*PI*{TONE_HZ}*t)"
            f"*lt(mod(t\\,{PERIOD_S})\\,{BURST_S})':s=44100:d={DURATION_S}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ac", "1", "-shortest",
            str(path),
        ],
        capture_output=True,
        text=True,
    )

    if process.returncode != 0 or not path.exists():
        raise RuntimeError(
            f"Could not build the test clip (is ffmpeg on PATH?): "
            f"{process.stderr[-400:]}"
        )

    return path
