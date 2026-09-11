from pathlib import Path
import json

from src.stages.preprocessing.audio import AudioProcessor
from src.stages.preprocessing.manifest import ManifestWriter
from src.stages.preprocessing.segmentation import Segmenter
from src.stages.preprocessing.validator import validate_media


TEST_AUDIO = "test.mp4"


def test_validate_media():
    assert validate_media(TEST_AUDIO)
    assert not validate_media("does_not_exist.wav")


def test_audio_processor(tmp_path):
    processor = AudioProcessor(
        sample_rate=16000,
        channels=1,
    )

    audio_path, latency = processor.extract(
        TEST_AUDIO,
        tmp_path,
    )

    assert audio_path.exists()
    assert latency > 0
    assert processor.duration(audio_path) > 0


def test_build_segments():
    segmenter = Segmenter()

    segments = segmenter.build_segments(
        [(2.0, 3.0), (5.0, 6.0)],
        8.0,
    )

    assert segments == [
        (0.0, 2.0),
        (3.0, 5.0),
        (6.0, 8.0),
    ]


def test_manifest_writer(tmp_path):
    writer = ManifestWriter()

    manifest = writer.write(
        chunk_paths=[
            Path("chunk_0000.wav"),
            Path("chunk_0001.wav"),
        ],
        segments=[
            (0.0, 1.0),
            (2.0, 3.5),
        ],
        output_path=tmp_path / "manifest.json",
    )

    with open(manifest) as f:
        data = json.load(f)

    assert len(data) == 2
    assert data[0]["start_ts"] == 0.0
    assert data[1]["end_ts"] == 3.5


# Real FFmpeg stderr from a clip that opens in silence. The leading
# silence_start is negative, which is what the old two-findall parser could
# not match.
FFMPEG_SILENCE_STDERR = """\
[silencedetect @ 0x6000021f0000] silence_start: -0.003013
[silencedetect @ 0x6000021f0000] silence_end: 1.2 | silence_duration: 1.203013
[silencedetect @ 0x6000021f0000] silence_start: 5.5
[silencedetect @ 0x6000021f0000] silence_end: 6.75 | silence_duration: 1.25
size=N/A time=00:00:08.00 bitrate=N/A speed=1.0x
"""


def test_negative_silence_start_is_parsed():
    silences = Segmenter._parse_silences(FFMPEG_SILENCE_STDERR)

    # Both intervals survive, in order, with the negative start intact.
    assert silences == [(-0.003013, 1.2), (5.5, 6.75)]

    # Every interval must move forward in time. The old parser produced
    # (5.5, 1.2) here, which inverts and overlaps the real speech.
    assert all(end > start for start, end in silences)


def test_segments_from_a_clip_that_opens_in_silence():
    segmenter = Segmenter()

    segments = segmenter.build_segments(
        Segmenter._parse_silences(FFMPEG_SILENCE_STDERR),
        8.0,
    )

    # Speech runs between the two silences, then to the end of the clip.
    assert segments == [(1.2, 5.5), (6.75, 8.0)]


def test_dangling_silence_start_is_discarded():
    # FFmpeg logs no silence_end when the file ends mid-silence. That start
    # must not be paired with an unrelated end.
    stderr = (
        "[silencedetect] silence_start: 1.0\n"
        "[silencedetect] silence_end: 2.0 | silence_duration: 1.0\n"
        "[silencedetect] silence_start: 7.5\n"
    )

    assert Segmenter._parse_silences(stderr) == [(1.0, 2.0)]
