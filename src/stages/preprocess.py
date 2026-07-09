import json
import re
import subprocess
import time
from pathlib import Path

import ffmpeg

from src.orchestrator.models import StageResult, StageStatus
from src.stages.base import PipelineStage
from src.stages.preprocessing.audio import AudioProcessor
from src.stages.preprocessing.segmentation import Segmenter
from src.stages.preprocessing.validator import validate_media


class FFmpegPreprocessStage(PipelineStage):
    """
    Phase 1 preprocessing stage.

    This step currently implements only lightweight validation.
    FFmpeg probing and extraction will be added in later atomic steps.
    """

    def validate_input(self, input_path: str) -> bool:
        return validate_media(input_path)



    def write_manifest(
        self,
        chunk_paths: list[Path],
        segments: list[tuple[float, float]],
        output_path: Path,
    ) -> Path:
        """
        Write a JSON manifest describing generated speech chunks.
        """

        manifest = []

        for chunk_path, (start, end) in zip(chunk_paths, segments):
            manifest.append(
                {
                    "chunk_path": str(chunk_path),
                    "start_ts": start,
                    "end_ts": end,
                }
            )

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return output_path

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

        start = time.perf_counter()

        audio_processor = AudioProcessor(
            sample_rate=cfg["audio"]["sample_rate"],
            channels=cfg["audio"]["channels"],
        )

        output_audio, latency_ms = audio_processor.extract(
            input_path=input_path,
            output_dir=output_dir,
        )

        total_duration = audio_processor.duration(output_audio)

        segmenter = Segmenter()

        silences = segmenter.detect_silences(str(output_audio))

        segments = segmenter.build_segments(
            silences,
            total_duration,
        )

        chunk_paths = segmenter.extract_segments(
            str(output_audio),
            segments,
            output_dir / "chunks",
        )

        manifest_path = self.write_manifest(
            chunk_paths,
            segments,
            output_dir / "manifest.json",
        )

        latency_ms = (time.perf_counter() - start) * 1000

        return StageResult(
            stage_name="preprocess",
            status=StageStatus.DONE,
            output_path=str(manifest_path),
            metrics={
                "latency_ms": latency_ms,
                "sample_rate": cfg["audio"]["sample_rate"],
                "channels": cfg["audio"]["channels"],
                "num_segments": len(chunk_paths),
            },
        )