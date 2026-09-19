"""
Guarantee the fixture media the suite has always assumed was lying around.

Six test modules hard-code `test.mp4` in the repo root, and `.gitignore` has
`*.mp4`, so it has never been in the repository. That worked for as long as
one particular local copy survived. When it was deleted, fourteen tests
across preprocessing, reference building and remux failed with ffprobe errors
that name neither the missing file nor each other.

So the suite now builds its own stand-in when the real file is absent: a
20-second clip whose audio is five 2-second tones separated by 2 seconds of
silence. The alternation matters — `Segmenter` finds speech by inverting
FFmpeg's `silencedetect`, so a continuous tone or a silent track yields no
segments and the preprocessing tests fail differently rather than passing.

A real recording is still better where one exists, and a local `test.mp4` is
used untouched if present. This is a floor under the suite, not a fixture
worth measuring anything on.
"""

import subprocess
from pathlib import Path

import pytest

TEST_VIDEO = Path(__file__).resolve().parent.parent / "test.mp4"

DURATION_S = 20
TONE_HZ = 220
# Quiet on purpose, about -14 dBFS. `build_reference` peak-normalizes to
# TARGET_PEAK_DBFS and a full-scale tone is already past it, so a loud
# stand-in would make that normalization a no-op and the test that asserts it
# would pass on silence-of-evidence.
TONE_AMPLITUDE = 0.2
# Tone for the first half of every 4-second period: five bursts, five gaps.
PERIOD_S = 4
BURST_S = 2


def _build_stand_in(path: Path) -> bool:
    process = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostdin", "-y",
            "-f", "lavfi",
            "-i", f"color=c=black:s=320x240:r=10:d={DURATION_S}",
            "-f", "lavfi",
            "-i",
            f"aevalsrc='{TONE_AMPLITUDE}*sin(2*PI*{TONE_HZ}*t)"
            f"*lt(mod(t\\,{PERIOD_S})\\,{BURST_S})'"
            f":s=44100:d={DURATION_S}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ac", "1", "-shortest",
            str(path),
        ],
        capture_output=True,
        text=True,
    )

    return process.returncode == 0 and path.exists()


@pytest.fixture(scope="session", autouse=True)
def fixture_media():
    """Build `test.mp4` if it is missing. Never replaces a real one."""
    if TEST_VIDEO.exists():
        return TEST_VIDEO

    if not _build_stand_in(TEST_VIDEO):
        pytest.skip(
            f"{TEST_VIDEO.name} is absent and no stand-in could be built "
            "(is ffmpeg on PATH?). The media-dependent tests cannot run.",
            allow_module_level=True,
        )

    print(f"\nBuilt a synthetic {TEST_VIDEO.name}: "
          f"{DURATION_S}s, five {BURST_S}s tones. See tests/conftest.py.")

    return TEST_VIDEO
