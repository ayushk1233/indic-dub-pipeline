import re
import subprocess
from pathlib import Path

import ffmpeg


class Segmenter:
    MIN_SEGMENT_DURATION_S = 0.25

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
