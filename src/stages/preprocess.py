from pathlib import Path

from src.orchestrator.models import StageResult
from src.stages.base import PipelineStage


class FFmpegPreprocessStage(PipelineStage):
    """
    Phase 1 preprocessing stage.

    This step currently implements only lightweight validation.
    FFmpeg probing and extraction will be added in later atomic steps.
    """

    def validate_input(self, input_path: str) -> bool:
        path = Path(input_path)

        return (
            path.exists()
            and path.is_file()
            and path.stat().st_size > 0
        )

    def run(
        self,
        input_path: str,
        job_id: str,
        cfg: dict,
    ) -> StageResult:
        raise NotImplementedError(
            "run() will be implemented in the next step."
        )