"""
Tests for the fitting cascade and track assembly.

The cascade is a sequence of escalating compromises, so what matters is that
each segment gets the *cheapest* fix that works, and that a segment which
cannot be fixed is reported rather than mangled.
"""

import wave

import numpy as np

from src.eval.tts_metrics import MAX_TEMPO
from src.stages.assemble import (
    DRIFT_RESET_GAP_S,
    apply_fades,
    assemble,
    time_stretch,
    trim_trailing_silence,
    write_wav_samples,
)
from src.stages.tts.models import (
    SynthesisRequest,
    SynthesisResult,
    SynthesisSegment,
    SynthesizedSegment,
)


SAMPLE_RATE = 24000


def tone(duration_s, trailing_silence_s=0.0, amplitude=0.3):
    n = int(duration_s * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    signal = amplitude * np.sin(2 * np.pi * 220.0 * t)
    silence = np.zeros(int(trailing_silence_s * SAMPLE_RATE))
    return np.concatenate([signal, silence]).astype(np.float32)


class FakeImported:
    """
    Stand-in for an ImportedBundle, so assembly can be tested without a
    round trip through the exporter and a GPU.
    """

    def __init__(self, request, result, audio_paths):
        self.request = request
        self.result = result
        self.audio_paths = audio_paths


def build(tmp_path, specs, total_duration, sample_rate=SAMPLE_RATE):
    """
    specs: list of (segment_id, start_ts, end_ts, audio_or_None)
    """
    segments = []
    produced = []
    audio_paths = {}

    for segment_id, start, end, audio in specs:
        segments.append(
            SynthesisSegment(
                segment_id=segment_id,
                chunk_id=0,
                start_ts=start,
                end_ts=end,
                text="क" * 12,
                reference_audio="request/reference.wav",
            )
        )

        if audio is None:
            continue

        path = tmp_path / f"seg_{segment_id:05d}.wav"
        write_wav_samples(path, audio, sample_rate)
        audio_paths[segment_id] = path
        produced.append(
            SynthesizedSegment(
                segment_id=segment_id,
                chunk_id=0,
                audio_path=path.name,
                duration=audio.size / sample_rate,
                num_samples=audio.size,
            )
        )

    return FakeImported(
        SynthesisRequest(
            job_id="t",
            language="hi",
            output_sample_rate=sample_rate,
            segments=segments,
        ),
        SynthesisResult(job_id="t", sample_rate=sample_rate, segments=produced),
        audio_paths,
    )


def read_track(path):
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        rate = handle.getframerate()
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0, rate


def test_trailing_silence_is_trimmed_but_a_tail_is_kept():
    samples = tone(1.0, trailing_silence_s=1.0)

    trimmed = trim_trailing_silence(samples, SAMPLE_RATE, keep_s=0.05)

    duration = trimmed.size / SAMPLE_RATE
    assert 1.0 < duration < 1.12


def test_trimming_leaves_clean_audio_alone():
    samples = tone(1.0)
    assert trim_trailing_silence(samples, SAMPLE_RATE).size == samples.size


def test_fades_zero_the_edges():
    faded = apply_fades(tone(0.5), SAMPLE_RATE)

    assert abs(float(faded[0])) < 1e-6
    assert abs(float(faded[-1])) < 1e-6

    # The middle is untouched. Check the peak of the middle third rather than
    # one sample, since a single sample can land on a zero crossing.
    third = faded.size // 3
    assert float(np.abs(faded[third : 2 * third]).max()) > 0.01


def test_time_stretch_shortens_by_the_requested_factor():
    stretched = time_stretch(tone(2.0), SAMPLE_RATE, 1.25)

    duration = stretched.size / SAMPLE_RATE
    assert abs(duration - 1.6) < 0.05


def test_a_segment_that_already_fits_is_placed_untouched(tmp_path):
    imported = build(
        tmp_path,
        [(0, 0.0, 2.0, tone(1.5))],
        total_duration=3.0,
    )

    result = assemble(imported, 3.0, tmp_path / "dubbed.wav")
    placed = result.segments[0]

    assert placed.tier == "placed"
    assert placed.tempo == 1.0
    assert result.num_placed == 1
    assert result.num_stretched == 0


def test_a_segment_uses_the_following_pause_before_being_stretched(tmp_path):
    # 2.6s of audio in a 2.0s slot, but nothing starts until 4.0s.
    imported = build(
        tmp_path,
        [(0, 0.0, 2.0, tone(2.6)), (1, 4.0, 5.0, tone(0.5))],
        total_duration=6.0,
    )

    result = assemble(imported, 6.0, tmp_path / "dubbed.wav")
    placed = result.segments[0]

    assert placed.tier == "gap-absorbed"
    assert placed.tempo == 1.0


def test_a_segment_too_long_for_its_budget_is_stretched_within_the_cap(tmp_path):
    # 2.3s of audio for a 2.0s slot with no pause after it.
    imported = build(
        tmp_path,
        [(0, 0.0, 2.0, tone(2.3)), (1, 2.0, 4.0, tone(1.0))],
        total_duration=4.0,
    )

    result = assemble(imported, 4.0, tmp_path / "dubbed.wav")
    placed = result.segments[0]

    assert placed.tier == "stretched"
    assert 1.0 < placed.tempo <= MAX_TEMPO
    # Stretching actually shortened it toward the budget.
    assert placed.final_duration_s < placed.trimmed_duration_s


def test_stretching_never_exceeds_the_cap(tmp_path):
    # 5s of audio for a 2s slot needs 2.5x, far past what speech tolerates.
    imported = build(
        tmp_path,
        [(0, 0.0, 2.0, tone(5.0)), (1, 2.0, 4.0, tone(1.0))],
        total_duration=4.0,
    )

    result = assemble(imported, 4.0, tmp_path / "dubbed.wav")
    placed = result.segments[0]

    assert placed.tier == "drifted"
    assert placed.tempo == MAX_TEMPO
    assert result.max_tempo_used == MAX_TEMPO


def test_drift_is_handed_back_at_a_real_pause(tmp_path):
    imported = build(
        tmp_path,
        [
            (0, 0.0, 2.0, tone(5.0)),            # overruns, creates drift
            (1, 2.0, 3.0, tone(0.5)),            # followed by a long pause
            (2, 8.0, 9.0, tone(0.5)),            # starts on time again
        ],
        total_duration=10.0,
    )

    result = assemble(imported, 10.0, tmp_path / "dubbed.wav")

    assert result.segments[0].tier == "drifted"
    # The gap after segment 1 is 5s, well past the reset threshold.
    assert DRIFT_RESET_GAP_S < 5.0
    assert result.segments[2].drift_s == 0.0
    assert result.segments[2].actual_start_s == 8.0


def test_a_missing_segment_leaves_silence_and_is_counted(tmp_path):
    imported = build(
        tmp_path,
        [(0, 0.0, 2.0, None), (1, 2.0, 4.0, tone(1.0))],
        total_duration=4.0,
    )

    result = assemble(imported, 4.0, tmp_path / "dubbed.wav")

    assert result.num_missing == 1
    assert result.segments[0].tier == "missing"
    assert result.segments[0].final_duration_s == 0.0

    track, _ = read_track(tmp_path / "dubbed.wav")
    # The first slot is silent, the second is not.
    assert np.abs(track[: int(1.5 * SAMPLE_RATE)]).max() < 0.01
    assert np.abs(track[int(2.1 * SAMPLE_RATE) : int(2.8 * SAMPLE_RATE)]).max() > 0.01


def test_the_track_is_exactly_the_requested_duration(tmp_path):
    imported = build(tmp_path, [(0, 0.0, 2.0, tone(1.0))], total_duration=7.5)

    assemble(imported, 7.5, tmp_path / "dubbed.wav")
    track, rate = read_track(tmp_path / "dubbed.wav")

    assert rate == SAMPLE_RATE
    assert abs(track.size / rate - 7.5) < 0.01


def test_a_segment_lands_at_its_exact_sample_offset(tmp_path):
    imported = build(tmp_path, [(0, 3.0, 5.0, tone(1.0))], total_duration=8.0)

    assemble(imported, 8.0, tmp_path / "dubbed.wav")
    track, rate = read_track(tmp_path / "dubbed.wav")

    start = int(3.0 * rate)

    # Silence right up to the placement point, audio just after it.
    assert np.abs(track[: start - 1]).max() < 0.01
    assert np.abs(track[start : start + int(0.5 * rate)]).max() > 0.01


def test_a_clipping_track_is_normalized(tmp_path):
    # Two loud overlapping segments would sum past full scale.
    imported = build(
        tmp_path,
        [
            (0, 0.0, 1.0, tone(2.0, amplitude=0.9)),
            (1, 0.5, 1.5, tone(2.0, amplitude=0.9)),
        ],
        total_duration=3.0,
    )

    assemble(imported, 3.0, tmp_path / "dubbed.wav")
    track, _ = read_track(tmp_path / "dubbed.wav")

    assert np.abs(track).max() <= 1.0
