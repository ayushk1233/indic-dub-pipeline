from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from src.stages.preprocess import FFmpegPreprocessStage


TEST_AUDIO = "/System/Library/Sounds/Glass.aiff"


def load_cfg():
    with open("config/pipeline.yaml") as f:
        return yaml.safe_load(f)


def run_job(job_id: str):
    stage = FFmpegPreprocessStage()

    result = stage.run(
        TEST_AUDIO,
        job_id,
        load_cfg(),
    )

    return result


def test_parallel_preprocessing():
    job_ids = [
        f"parallel-{i}"
        for i in range(5)
    ]

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(
            executor.map(run_job, job_ids)
        )

    for result, job_id in zip(results, job_ids):
        assert result.status.value == "done"

        manifest = Path(result.output_path)

        assert manifest.exists()

        assert str(manifest).startswith(
            f"artifacts/{job_id}"
        )
