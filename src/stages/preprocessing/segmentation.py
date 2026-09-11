import re
import subprocess
from pathlib import Path

import ffmpeg


class Segmenter:
    MIN_SEGMENT_DURATION_S = 0.25
    FALLBACK_WINDOW_SECONDS = 30.0
    FALLBACK_OVERLAP_SECONDS = 1.0

    # Matches either marker in one pass, in the order FFmpeg emits them, and
    # accepts a leading '-' because a clip that opens in silence logs a
    # negative silence_start (e.g. "silence_start: -0.003013"). The old code
    # used two independent findall() calls with a pattern that could not
    # match that minus sign: the start was dropped from its list but the
    # matching end was not, which desynchronized every pair from that point
    # on when the two lists were zipped.
    _SILENCE_EVENT_RE = re.compile(r"silence_(start|end):\s*(-?[0-9.]+)")

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

        return self._parse_silences(process.stderr)

    @classmethod
    def _parse_silences(cls, ffmpeg_stderr: str) -> list[tuple[float, float]]:
        """
        Pair silence_start/silence_end markers in the order they appear,
        rather than trusting two separately-collected lists to line up.

        A clip that is still in silence when FFmpeg stops analyzing logs a
        silence_start with no matching silence_end; that dangling start is
        discarded rather than paired with the wrong end.
        """
        silences: list[tuple[float, float]] = []
        pending_start: float | None = None

        for kind, value in cls._SILENCE_EVENT_RE.findall(ffmpeg_stderr):
            if kind == "start":
                pending_start = float(value)
            elif kind == "end" and pending_start is not None:
                silences.append((pending_start, float(value)))
                pending_start = None

        return silences

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

    def build_fixed_window_segments(
        self,
        total_duration: float,
    ) -> list[tuple[float, float]]:
        """
        Generate fixed-duration segments when silence detection
        does not produce usable speech segments.
        """

        segments = []

        start = 0.0

        while start < total_duration:
            end = min(
                start + self.FALLBACK_WINDOW_SECONDS,
                total_duration,
            )

            segments.append((start, end))

            if end >= total_duration:
                break

            start = end - self.FALLBACK_OVERLAP_SECONDS

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
