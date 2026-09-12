"""
Tests for building the voice reference.

The reference is the single input that decides whether cloning works, and the
first real GPU run failed on it: a 16 kHz chunk peaking at 0.21 scored 0.478
mean speaker similarity against a 0.75 floor. These tests pin the two
properties that were wrong.
"""

import wave

import pytest

from src.stages.reference import (
    REFERENCE_SAMPLE_RATE,
    TARGET_PEAK_DBFS,
    build_reference,
    measure_peak_dbfs,
)


TEST_VIDEO = "test.mp4"


def read(path):
    with wave.open(str(path), "rb") as handle:
        return {
            "rate": handle.getframerate(),
            "channels": handle.getnchannels(),
            "width": handle.getsampwidth(),
            "frames": handle.getnframes(),
        }


def test_the_reference_is_cut_at_the_synthesis_sample_rate(tmp_path):
    out = build_reference(TEST_VIDEO, (2.0, 6.0), tmp_path / "reference.wav")

    info = read(out)

    # 24 kHz mono 16-bit. Anything lower means XTTS upsamples and clones a
    # voice with no energy above half that rate.
    assert info["rate"] == REFERENCE_SAMPLE_RATE
    assert info["channels"] == 1
    assert info["width"] == 2


def test_the_span_is_padded_on_both_sides(tmp_path):
    out = build_reference(TEST_VIDEO, (2.0, 6.0), tmp_path / "reference.wav")

    info = read(out)
    duration = info["frames"] / info["rate"]

    # 4s of span plus 0.25s of padding at each end.
    assert 4.4 < duration < 4.6


def test_a_quiet_reference_is_brought_up_to_the_target_peak(tmp_path):
    out = build_reference(TEST_VIDEO, (2.0, 6.0), tmp_path / "reference.wav")

    peak = measure_peak_dbfs(out)

    assert peak is not None
    # Normalized to just under full scale, not left wherever the source sat.
    assert TARGET_PEAK_DBFS - 0.5 <= peak <= TARGET_PEAK_DBFS + 0.5


def test_an_inverted_span_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Empty reference span"):
        build_reference(TEST_VIDEO, (5.0, 3.0), tmp_path / "reference.wav")


def test_a_zero_length_span_survives_on_padding_alone(tmp_path):
    # Not a good reference, but the padding makes it a real file rather than a
    # crash. The manifest never produces spans this short.
    out = build_reference(TEST_VIDEO, (3.0, 3.0), tmp_path / "reference.wav")

    assert read(out)["frames"] > 0


def test_a_missing_source_is_an_explicit_error(tmp_path):
    with pytest.raises(RuntimeError, match="Could not cut a reference"):
        build_reference("no_such_file.mp4", (0.0, 1.0), tmp_path / "reference.wav")
