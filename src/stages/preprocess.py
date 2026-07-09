from pathlib import Path

import ffmpeg

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

        if not (
            path.exists()
            and path.is_file()
            and path.stat().st_size > 0
        ):
            return False

        try:
            probe = ffmpeg.probe(str(path))
        except ffmpeg.Error:
            return False

        audio_streams = [
            stream
            for stream in probe.get("streams", [])
            if stream.get("codec_type") == "audio"
        ]

        if not audio_streams:
            return False

        duration = float(probe.get("format", {}).get("duration", 0))

        return duration > 0

    def run(
        self,
        input_path: str,
        job_id: str,
        cfg: dict,
    ) -> StageResult:
        raise NotImplementedError(
            "run() will be implemented in the next step."
        )