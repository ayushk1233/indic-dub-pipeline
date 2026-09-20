"""
Tests for putting the dubbed track back onto the video.

These run against a short generated clip (tests/fixture_media.py), because the
failure this guards
against — audio and video disagreeing about how long the file is — only shows
up in a real container.
"""

import numpy as np
import pytest

from src.stages.assemble import write_wav_samples
from src.stages.remux import (
    DURATION_TOLERANCE_S,
    has_video_stream,
    probe_duration,
    probe_stream_durations,
    remux,
)


from fixture_media import TEST_VIDEO
SAMPLE_RATE = 24000


def tone(duration_s, amplitude=0.2):
    n = int(duration_s * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def test_video_stream_detection():
    assert has_video_stream(TEST_VIDEO)


def test_a_short_track_is_padded_to_the_video_length(tmp_path):
    video_duration = probe_duration(TEST_VIDEO)

    # Deliberately far too short: a quarter of the video.
    audio = tmp_path / "dubbed.wav"
    write_wav_samples(audio, tone(video_duration / 4), SAMPLE_RATE)

    result = remux(TEST_VIDEO, audio, tmp_path / "out.mp4")

    assert result.in_sync, f"drifted by {result.drift_s:.3f}s"
    assert abs(result.video_duration_s - video_duration) < DURATION_TOLERANCE_S
    assert abs(result.audio_duration_s - video_duration) < DURATION_TOLERANCE_S


def test_a_long_track_is_truncated_to_the_video_length(tmp_path):
    video_duration = probe_duration(TEST_VIDEO)

    audio = tmp_path / "dubbed.wav"
    write_wav_samples(audio, tone(video_duration + 10.0), SAMPLE_RATE)

    result = remux(TEST_VIDEO, audio, tmp_path / "out.mp4")

    assert result.in_sync
    assert result.audio_duration_s < video_duration + DURATION_TOLERANCE_S


def test_the_output_carries_both_streams(tmp_path):
    audio = tmp_path / "dubbed.wav"
    write_wav_samples(audio, tone(5.0), SAMPLE_RATE)

    output = tmp_path / "out.mp4"
    remux(TEST_VIDEO, audio, output)

    streams = probe_stream_durations(output)

    assert "video" in streams
    assert "audio" in streams


def test_ducking_the_original_still_produces_one_audio_track(tmp_path):
    audio = tmp_path / "dubbed.wav"
    write_wav_samples(audio, tone(5.0), SAMPLE_RATE)

    result = remux(
        TEST_VIDEO,
        audio,
        tmp_path / "out.mp4",
        keep_original_audio_at=0.15,
    )

    assert result.had_original_audio
    assert result.in_sync


def test_a_missing_input_is_reported_clearly(tmp_path):
    audio = tmp_path / "dubbed.wav"
    write_wav_samples(audio, tone(1.0), SAMPLE_RATE)

    with pytest.raises(FileNotFoundError, match="Video not found"):
        remux(tmp_path / "nope.mp4", audio, tmp_path / "out.mp4")

    with pytest.raises(FileNotFoundError, match="Dubbed audio not found"):
        remux(TEST_VIDEO, tmp_path / "nope.wav", tmp_path / "out.mp4")


def test_remuxing_onto_something_with_no_video_is_refused(tmp_path):
    audio = tmp_path / "dubbed.wav"
    write_wav_samples(audio, tone(1.0), SAMPLE_RATE)

    with pytest.raises(ValueError, match="no video stream"):
        remux(audio, audio, tmp_path / "out.mp4")


# -- the pin is the video stream, not the container --------------------------


def test_the_output_is_pinned_to_the_video_stream_not_the_container():
    """
    QuickTime .mov files carry a container duration longer than either stream:
    hindi.mov reports 73.812s at container level against a video stream of
    73.693s. Pinning audio to the container while `-c:v copy` copies the video
    at its own length leaves the two output streams 0.118s apart, which the
    drift check reports as a sync failure on a file that is not out of sync.
    """
    from unittest.mock import patch

    from src.stages.remux import probe_video_duration

    streams = {"video": 73.693333, "audio": 73.700136}

    with patch("src.stages.remux.probe_stream_durations", return_value=streams):
        with patch("src.stages.remux.probe_duration", return_value=73.811814):
            assert probe_video_duration("hindi.mov") == 73.693333


def test_a_file_without_a_video_stream_duration_falls_back_to_the_container():
    """
    Not every container reports per-stream durations. Falling back keeps the
    old behaviour rather than pinning to zero and truncating the whole track.
    """
    from unittest.mock import patch

    from src.stages.remux import probe_video_duration

    with patch("src.stages.remux.probe_stream_durations", return_value={"audio": 10.0}):
        with patch("src.stages.remux.probe_duration", return_value=12.5):
            assert probe_video_duration("odd.mkv") == 12.5
