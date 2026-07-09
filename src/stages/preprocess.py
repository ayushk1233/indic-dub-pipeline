import time
from pathlib import Path

import ffmpeg

from src.orchestrator.models import StageResult, StageStatus
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
        if not self.validate_input(input_path):
            return StageResult(
                stage_name="preprocess",
                status=StageStatus.FAILED,
                error="Input validation failed.",
            )

        output_dir = Path("artifacts") / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        output_audio = output_dir / "audio.wav"

        start = time.perf_counter()

        (
            ffmpeg
            .input(input_path)
            .output(
                str(output_audio),
                ar=cfg["audio"]["sample_rate"],
                ac=cfg["audio"]["channels"],
                format="wav",
            )
            .overwrite_output()
            .run(quiet=True)
        )

        latency_ms = (time.perf_counter() - start) * 1000

        return StageResult(
            stage_name="preprocess",
            status=StageStatus.DONE,
            output_path=str(output_audio),
            metrics={
                "latency_ms": latency_ms,
                "sample_rate": cfg["audio"]["sample_rate"],
                "channels": cfg["audio"]["channels"],
            },
        )