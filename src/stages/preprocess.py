import re
import subprocess
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
            if silence_start > current_start:
                segments.append((current_start, silence_start))
            current_start = silence_end

        if current_start < total_duration:
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