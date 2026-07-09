import json
from pathlib import Path

import yaml

from src.stages.preprocess import FFmpegPreprocessStage


TEST_AUDIO = "/System/Library/Sounds/Glass.aiff"


def load_cfg():
    with open("config/pipeline.yaml") as f:
        return yaml.safe_load(f)


def test_preprocess_end_to_end(tmp_path):
    cfg = load_cfg()

    stage = FFmpegPreprocessStage()

    result = stage.run(
        TEST_AUDIO,
        "integration-pytest",
        cfg,
    )

    assert result.status.value == "done"

    manifest_path = Path(result.output_path)

    assert manifest_path.exists()

    with open(manifest_path) as f:
        manifest = json.load(f)

    assert len(manifest) >= 1

    for entry in manifest:
        assert Path(entry["chunk_path"]).exists()
        assert entry["end_ts"] > entry["start_ts"]


def test_invalid_media():
    cfg = load_cfg()

    stage = FFmpegPreprocessStage()

    assert not stage.validate_input(
        "this_file_does_not_exist.wav"
    )
