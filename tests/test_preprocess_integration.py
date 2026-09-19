import json
from pathlib import Path

import yaml

from src.stages.preprocess import FFmpegPreprocessStage


from fixture_media import TEST_VIDEO as TEST_AUDIO


def load_cfg(root=None):
    """
    The pipeline config, with the artifact root pointed somewhere disposable.

    Without the root, `FFmpegPreprocessStage` writes into the repo's own
    `artifacts/` and the suite leaves real job directories behind on every
    run. The only recordings that belong in this project are english.mov and
    hindi.mov.
    """
    with open("config/pipeline.yaml") as f:
        cfg = yaml.safe_load(f)

    if root is not None:
        cfg["artifacts"] = {"root": str(root)}

    return cfg


def test_preprocess_end_to_end(tmp_path):
    cfg = load_cfg(tmp_path)

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
