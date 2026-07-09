from pathlib import Path
import time

import ffmpeg


class AudioProcessor:
    """
    Handles extraction and normalization of audio inputs.
    """

    def __init__(
        self,
        sample_rate: int,
        channels: int,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels

    def extract(
        self,
        input_path: str,
        output_dir: Path,
    ) -> tuple[Path, float]:
        """
        Extract and normalize audio to WAV.

        Returns:
            (output_audio_path, latency_ms)
        """

        output_dir.mkdir(parents=True, exist_ok=True)

        output_audio = output_dir / "audio.wav"

        start = time.perf_counter()

        (
            ffmpeg
            .input(input_path)
            .output(
                str(output_audio),
                ar=self.sample_rate,
                ac=self.channels,
                format="wav",
            )
            .overwrite_output()
            .run(quiet=True)
        )

        latency_ms = (time.perf_counter() - start) * 1000

        return output_audio, latency_ms

    def duration(
        self,
        audio_path: Path,
    ) -> float:
        """
        Return duration in seconds.
        """

        probe = ffmpeg.probe(str(audio_path))

        return float(probe["format"]["duration"])
