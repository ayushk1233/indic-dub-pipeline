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


from fixture_media import TEST_VIDEO


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


# -- model-specific reference length -----------------------------------------


def test_indicf5_gets_a_reference_under_its_own_clipping_threshold():
    """
    FINDINGS §5d: IndicF5 clips reference audio past 15s inside
    `preprocess_ref_audio_text` and never truncates `ref_text` to match, so a
    25s clip hands it a transcript describing audio it cannot hear. That is
    the mismatch §5 blames for the original gibberish, and §1's shipping
    configuration is a 10s clip because of it.
    """
    from src.stages.reference import reference_seconds

    assert reference_seconds("ai4bharat/indicf5") < 15.0


def test_xtts_keeps_the_length_it_was_tuned_for():
    from src.stages.reference import TARGET_REFERENCE_S, reference_seconds

    assert reference_seconds("coqui/xtts_v2") == TARGET_REFERENCE_S


def test_an_unknown_model_degrades_rather_than_failing_the_export():
    from src.stages.reference import TARGET_REFERENCE_S, reference_seconds

    assert reference_seconds(None) == TARGET_REFERENCE_S
    assert reference_seconds("something/else") == TARGET_REFERENCE_S


def test_the_shipping_config_selects_a_length_indicf5_can_use():
    """
    The config string and the lookup table have to agree, and nothing else
    checks that they do — a typo in either would silently fall through to
    XTTS's 25s and reproduce §5.
    """
    import yaml

    from src.stages.reference import reference_seconds

    with open("config/pipeline.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert reference_seconds(cfg["tts"]["model"]) < 15.0


# -- span selection must not overshoot the ceiling ---------------------------


def _runner(model="ai4bharat/indicf5", entries=None):
    """A runner with just enough state for _reference_spans()."""
    import json
    import tempfile
    from pathlib import Path

    from src.pipeline.paths import JobPaths
    from src.pipeline.runner import PipelineRunner

    tmp = Path(tempfile.mkdtemp())
    paths = JobPaths(job_id="t", root=tmp)
    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.manifest.write_text(json.dumps(entries or []), encoding="utf-8")

    return PipelineRunner(cfg={"tts": {"model": model}}, paths=paths)


def _spans(*pairs):
    return [{"start_ts": a, "end_ts": b} for a, b in pairs]


def test_selection_never_overshoots_indicf5s_ceiling():
    """
    The old loop appended while `total < target`, so the finished clip could
    overshoot by a whole span — two 9s spans against a 10s target give 18s,
    past the 15s at which IndicF5 clips the audio and keeps the transcript
    (FINDINGS §5d). Padding counts toward it too.
    """
    from src.stages.reference import SPAN_PADDING_S, reference_ceiling

    runner = _runner(entries=_spans((0.0, 9.0), (20.0, 29.0)))

    chosen = runner._reference_spans()
    predicted = sum(e - s for s, e in chosen) + 2 * SPAN_PADDING_S * len(chosen)

    assert predicted <= reference_ceiling("ai4bharat/indicf5")


def test_a_shorter_span_is_still_taken_when_a_long_one_does_not_fit():
    """Skipping the span that overshoots should not end the search."""
    runner = _runner(entries=_spans((0.0, 9.0), (20.0, 28.0), (40.0, 43.0)))

    chosen = runner._reference_spans()

    assert (40.0, 43.0) in chosen


def test_a_single_over_long_span_is_trimmed_rather_than_shipped():
    """
    With nothing else to choose from, the clip still must not exceed the
    ceiling — the transcript already describes more than the model will hear.
    """
    from src.stages.reference import SPAN_PADDING_S, reference_ceiling

    ceiling = reference_ceiling("ai4bharat/indicf5")
    runner = _runner(entries=_spans((0.0, 60.0)))

    chosen = runner._reference_spans()
    predicted = sum(e - s for s, e in chosen) + 2 * SPAN_PADDING_S * len(chosen)

    assert len(chosen) == 1
    assert predicted <= ceiling


def test_xtts_has_no_ceiling_and_keeps_its_longer_reference():
    """
    XTTS truncates conditioning cleanly with max_ref_length and ignores the
    transcript, so capping it would only throw reference away.
    """
    runner = _runner(model="coqui/xtts_v2",
                     entries=_spans((0.0, 9.0), (20.0, 29.0), (40.0, 49.0)))

    chosen = runner._reference_spans()

    assert sum(e - s for s, e in chosen) > 14.0
