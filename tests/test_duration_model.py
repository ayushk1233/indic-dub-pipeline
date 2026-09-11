"""
Tests for the spoken-duration model.

The important guarantee is not that the model is accurate, it is that a model
which fails to beat the constant it replaces is never used. These tests pin
that behaviour down.
"""

import math

import pytest

from src.data.fleurs import SpeechSample
from src.eval.duration_model import (
    DurationModelSet,
    LanguageDurationModel,
    fit_language,
    feature_vector,
    text_features,
)


def sample(sentence_id, language, text, duration_s):
    return SpeechSample(
        sentence_id=sentence_id,
        language=language,
        text=text,
        duration_s=duration_s,
    )


def test_combining_marks_are_counted_separately():
    # "क्या" is four codepoints but one spoken syllable: two base consonants
    # plus a virama and a vowel sign that attach to them.
    features = text_features("क्या")

    assert features["chars"] == 4.0
    assert features["base_chars"] == 2.0
    assert features["combining"] == 2.0


def test_latin_text_has_no_combining_marks():
    features = text_features("Are you all right?")

    assert features["combining"] == 0.0
    assert features["punctuation"] == 1.0
    assert features["words"] == 4.0
    # Whitespace and punctuation are excluded from the base count.
    assert features["base_chars"] == 14.0


def test_feature_vector_matches_declared_order():
    assert feature_vector("क्या आप ठीक हैं?") == [7.0, 5.0, 4.0, 1.0]


def build_linear_samples(language, n=80, seconds_per_char=0.08, overhead=1.5):
    """
    Sentences whose duration follows base_chars exactly, so a correct fit is
    recoverable and a constant rate provably is not.
    """
    samples = []

    for i in range(n):
        base_chars = 10 + (i % 40) * 5
        text = "क" * base_chars
        samples.append(
            sample(i, language, text, seconds_per_char * base_chars + overhead)
        )

    return samples


def test_fit_beats_the_constant_when_there_is_fixed_overhead():
    model = fit_language(build_linear_samples("hi"), "hi")

    assert model.num_train > 0
    assert model.num_test > 0
    assert model.beats_baseline

    # The fixed overhead is exactly what a constant rate cannot express.
    assert math.isclose(model.intercept, 1.5, abs_tol=0.2)
    assert model.model_score.mae_s < model.baseline_score.mae_s


def test_a_losing_model_is_not_used_for_prediction():
    model = LanguageDurationModel(
        language="hi",
        coefficients=[99.0, 0.0, 0.0, 0.0],  # deliberately absurd
        intercept=500.0,
        fallback_cps=10.0,
    )

    # With no scores recorded, beats_baseline is False, so the constant is used.
    assert not model.beats_baseline
    assert math.isclose(model.predict("क" * 100), 10.0, rel_tol=1e-6)


def test_prediction_is_never_zero_or_negative():
    model = LanguageDurationModel(language="hi", fallback_cps=10.0)

    assert model.predict("") > 0
    assert model.predict("   ") > 0


def test_too_little_data_yields_an_unfitted_model():
    model = fit_language([sample(0, "hi", "क" * 10, 1.0)], "hi")

    assert model.coefficients == []
    assert not model.beats_baseline
    # Still usable, via the constant.
    assert model.predict("क" * 13) > 0


def test_model_set_round_trips_through_disk(tmp_path):
    original = DurationModelSet(
        models={"hi": fit_language(build_linear_samples("hi"), "hi")},
        source="test",
    )

    path = original.save(tmp_path / "duration_model.json")
    restored = DurationModelSet.load(path)

    assert restored.source == "test"
    assert math.isclose(
        restored.predict("क" * 50, "hi"),
        original.predict("क" * 50, "hi"),
        rel_tol=1e-9,
    )


def test_unknown_language_falls_back_rather_than_raising():
    model_set = DurationModelSet()

    assert model_set.predict("hello there", "xx") > 0


def test_loading_a_model_with_a_different_feature_order_is_refused(tmp_path):
    import json

    path = tmp_path / "stale.json"
    path.write_text(
        json.dumps({"feature_order": ["chars"], "models": {}}),
        encoding="utf-8",
    )

    try:
        DurationModelSet.load(path)
    except ValueError as exc:
        assert "feature order" in str(exc)
    else:
        raise AssertionError("stale coefficients must not load silently")


def test_overhead_is_excluded_from_prediction_by_default():
    """
    The fitted intercept is FLEURS recording silence, which does not exist in
    synthesized speech placed on a timeline. Including it made eight
    characters of Hindi predict 2.6 seconds.
    """
    model = fit_language(build_linear_samples("hi", overhead=1.5), "hi")

    assert model.beats_baseline
    assert math.isclose(model.intercept, 1.5, abs_tol=0.2)

    short = "क" * 10

    without = model.predict(short)
    with_overhead = model.predict(short, include_overhead=True)

    assert with_overhead - without == pytest.approx(model.intercept, abs=1e-6)
    # The default must not be dominated by a constant the deployment data
    # does not have.
    assert without < with_overhead / 2


def test_short_text_prediction_stays_in_the_right_order_of_magnitude():
    model = fit_language(build_linear_samples("hi", overhead=1.5), "hi")

    # Roughly one second of speech should not predict several seconds.
    predicted = model.predict("क" * 13)

    assert 0.5 < predicted < 2.5


def test_prediction_grows_with_length():
    model = fit_language(build_linear_samples("hi"), "hi")

    lengths = [10, 40, 90, 150]
    predictions = [model.predict("क" * n) for n in lengths]

    assert predictions == sorted(predictions)
