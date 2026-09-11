"""
Put the dubbed audio track back onto the original video.

The only real risk here is drift. If the muxer is left to decide how long the
output runs, a dubbed track that is a fraction of a second short or long
against the video produces a file whose audio slides out of sync toward the
end, and the error is invisible until someone watches the last minute. So the
duration is measured from the video and pinned explicitly, with silence padded
or the tail cut to match. Video is stream-copied, never re-encoded, because
re-encoding the picture to change the soundtrack is pure loss.
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


# Output audio settings. AAC because it plays everywhere; 192 kbps because
# speech does not benefit from more and the file should stay shareable.
AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"

# Video and audio duration may differ by at most this much in the output
# before it counts as a drift failure rather than container rounding.
DURATION_TOLERANCE_S = 0.1


@dataclass
class RemuxResult:
    output_path: str
    video_duration_s: float
    audio_duration_s: float
    had_original_audio: bool

    @property
    def drift_s(self) -> float:
        return abs(self.video_duration_s - self.audio_duration_s)

    @property
    def in_sync(self) -> bool:
        return self.drift_s <= DURATION_TOLERANCE_S


def probe_duration(media_path: Path) -> float:
    """
    Duration of a media file in seconds, from the container.
    """
    import ffmpeg

    probe = ffmpeg.probe(str(media_path))

    return float(probe["format"]["duration"])


def probe_stream_durations(media_path: Path) -> dict[str, float]:
    """
    Per-stream durations, used to verify the muxed output rather than trusting
    that the command did what was asked.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,duration",
            "-of",
            "json",
            str(media_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    durations: dict[str, float] = {}

    for stream in json.loads(result.stdout).get("streams", []):
        codec_type = stream.get("codec_type")
        duration = stream.get("duration")

        if codec_type and duration is not None:
            durations[codec_type] = float(duration)

    return durations


def has_video_stream(media_path: Path) -> bool:
    import ffmpeg

    try:
        probe = ffmpeg.probe(str(media_path))
    except Exception:
        return False

    return any(
        stream.get("codec_type") == "video"
        for stream in probe.get("streams", [])
    )


def remux(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    keep_original_audio_at: float | None = None,
) -> RemuxResult:
    """
    Replace the video's soundtrack with the dubbed track.

    `keep_original_audio_at` mixes the source audio back in at that linear
    gain, which is how dubbed documentaries keep ambient sound under the
    narration. Leave it None to replace the audio outright.
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not audio_path.exists():
        raise FileNotFoundError(f"Dubbed audio not found: {audio_path}")

    if not has_video_stream(video_path):
        raise ValueError(
            f"{video_path} has no video stream; there is nothing to remux onto."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = probe_duration(video_path)
    source_has_audio = _has_audio_stream(video_path)

    command = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
    ]

    if keep_original_audio_at is not None and source_has_audio:
        # Duck the original under the dub, then pad so the mix cannot end
        # early. amix would renormalize, so the gains are set explicitly.
        filtergraph = (
            f"[0:a]volume={keep_original_audio_at:.3f}[bed];"
            f"[1:a]volume=1.0[dub];"
            f"[bed][dub]amix=inputs=2:duration=longest:normalize=0[mixed];"
            f"[mixed]apad[out]"
        )
        command += ["-filter_complex", filtergraph, "-map", "0:v:0", "-map", "[out]"]
    else:
        command += ["-af", "apad", "-map", "0:v:0", "-map", "1:a:0"]

    command += [
        "-c:v",
        "copy",
        "-c:a",
        AUDIO_CODEC,
        "-b:a",
        AUDIO_BITRATE,
        # Pin the output length to the video. apad above guarantees there is
        # always audio to reach this point, so this both pads and truncates.
        "-t",
        f"{duration:.6f}",
        "-movflags",
        "+faststart",
        "-y",
        str(output_path),
    ]

    process = subprocess.run(command, capture_output=True, text=True)

    if process.returncode != 0:
        raise RuntimeError(
            f"ffmpeg remux failed ({process.returncode}):\n{process.stderr.strip()}"
        )

    streams = probe_stream_durations(output_path)

    return RemuxResult(
        output_path=str(output_path),
        video_duration_s=streams.get("video", duration),
        audio_duration_s=streams.get("audio", duration),
        had_original_audio=source_has_audio,
    )


def _has_audio_stream(media_path: Path) -> bool:
    import ffmpeg

    try:
        probe = ffmpeg.probe(str(media_path))
    except Exception:
        return False

    return any(
        stream.get("codec_type") == "audio"
        for stream in probe.get("streams", [])
    )
