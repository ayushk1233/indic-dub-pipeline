from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from src.stages.preprocess import FFmpegPreprocessStage


from fixture_media import TEST_VIDEO as TEST_AUDIO


def load_cfg(root=None):
    """See tests/test_preprocess_integration.py: keep jobs out of the repo."""
    with open("config/pipeline.yaml") as f:
        cfg = yaml.safe_load(f)

    if root is not None:
        cfg["artifacts"] = {"root": str(root)}

    return cfg


def run_job(job_id: str, root=None):
    stage = FFmpegPreprocessStage()

    result = stage.run(
        TEST_AUDIO,
        job_id,
        load_cfg(root),
    )

    return result


def test_parallel_preprocessing(tmp_path):
    job_ids = [
        f"parallel-{i}"
        for i in range(5)
    ]

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(
            executor.map(lambda j: run_job(j, tmp_path), job_ids)
        )

    for result, job_id in zip(results, job_ids):
        assert result.status.value == "done"

        manifest = Path(result.output_path)

        assert manifest.exists()

        assert str(manifest).startswith(
            str(tmp_path / job_id)
        )
