import json
import re
import subprocess
import time
from pathlib import Path

import ffmpeg

from src.orchestrator.models import StageResult, StageStatus
from src.stages.base import PipelineStage
from src.stages.preprocessing.audio import (
    extract_normalized_audio,
    get_audio_duration,
)
from src.stages.preprocessing.validator import validate_media


class FFmpegPreprocessStage(PipelineStage):
    """
    Phase 1 preprocessing stage.

    This step currently implements only lightweight validation.
    FFmpeg probing and extraction will be added in later atomic steps.
    """

    MIN_SEGMENT_DURATION_S = 0.25

    def validate_input(self, input_path: str) -> bool:
        return validate_media(input_path)

    def detect_silences(
        self,
        audio_path: str,
        noise: str = "-30dB",
        duration: float = 0.5,
    ) -> list[tuple[float, float]]:
        """
        Returns a list of (silence_start, silence_end) tuples.
        """

        command = [
            "ffmpeg",
            "-i",
            audio_path,
            "-af",
            f"silencedetect=noise={noise}:d={duration}",
            "-f",
            "null",
            "-"
        ]

        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )

        output = process.stderr

        starts = [
            float(x)
            for x in re.findall(
                r"silence_start:\s*([0-9.]+)",
                output,
            )
        ]

        ends = [
            float(x)
            for x in re.findall(
                r"silence_end:\s*([0-9.]+)",
                output,
            )
        ]

        return list(zip(starts, ends))

    def build_segments(
        self,
        silences: list[tuple[float, float]],
        total_duration: float,
    ) -> list[tuple[float, float]]:
        """
        Convert silence intervals into speech segments.

        Returns:
            [(speech_start, speech_end), ...]
        """

        segments = []
        current_start = 0.0

        for silence_start, silence_end in silences:
            duration = silence_start - current_start

            if duration >= self.MIN_SEGMENT_DURATION_S:
                segments.append((current_start, silence_start))
            current_start = silence_end

        duration = total_duration - current_start

        if duration >= self.MIN_SEGMENT_DURATION_S:
            segments.append((current_start, total_duration))

        return segments

    def extract_segments(
        self,
        audio_path: str,
        segments: list[tuple[float, float]],
        output_dir: Path,
    ) -> list[Path]:
        """
        Extract each speech segment into an individual WAV file.
        """

        output_dir.mkdir(parents=True, exist_ok=True)

        chunk_paths: list[Path] = []

        for index, (start, end) in enumerate(segments):
            chunk_path = output_dir / f"chunk_{index:04d}.wav"

            (
                ffmpeg
                .input(audio_path, ss=start, to=end)
                .output(str(chunk_path))
                .overwrite_output()
                .run(quiet=True)
            )

            chunk_paths.append(chunk_path)

        return chunk_paths

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

        output_audio, _ = extract_normalized_audio(
            input_path=input_path,
            output_dir=output_dir,
            sample_rate=cfg["audio"]["sample_rate"],
            channels=cfg["audio"]["channels"],
        )

        total_duration = get_audio_duration(output_audio)

        silences = self.detect_silences(str(output_audio))

        segments = self.build_segments(
            silences,
            total_duration,
        )

        chunk_paths = self.extract_segments(
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