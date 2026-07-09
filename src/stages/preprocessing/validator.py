from pathlib import Path

import ffmpeg


def validate_media(input_path: str) -> bool:
    """
    Validate that the input exists, is non-empty,
    contains an audio stream, and has a duration > 0.
    """

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

    duration = float(
        probe.get("format", {}).get("duration", 0)
    )

    return duration > 0
