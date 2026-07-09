from pathlib import Path
import time

import ffmpeg


def extract_normalized_audio(
    input_path: str,
    output_dir: Path,
    sample_rate: int,
    channels: int,
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
            ar=sample_rate,
            ac=channels,
            format="wav",
        )
        .overwrite_output()
        .run(quiet=True)
    )

    latency_ms = (time.perf_counter() - start) * 1000

    return output_audio, latency_ms


def get_audio_duration(audio_path: Path) -> float:
    """
    Return duration of normalized audio in seconds.
    """

    probe = ffmpeg.probe(str(audio_path))
    return float(probe["format"]["duration"])
