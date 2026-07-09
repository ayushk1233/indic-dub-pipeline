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
