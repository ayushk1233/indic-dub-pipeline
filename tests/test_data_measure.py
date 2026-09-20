"""
Tests for the FLEURS measurement layer.

Nothing here touches the network. The loader is exercised elsewhere; what
matters for correctness is the arithmetic that turns samples into the
constants the feasibility check depends on.
"""

import math

from src.data.fleurs import SpeechSample, pair_by_sentence
from src.data.measure import (
    _linear_fit,
    _percentile,
    measure_expansion,
    measure_rate,
)


def sample(sentence_id, language, text, duration_s):
    return SpeechSample(
        sentence_id=sentence_id,
        language=language,
        text=text,
        duration_s=duration_s,
    )


def test_sample_derives_rate_from_text_and_duration():
    s = sample(1, "hi", "क" * 100, 10.0)

    assert s.chars == 100
    assert s.cps == 10.0

    # A zero-length clip must not divide by zero.
    assert sample(2, "hi", "abc", 0.0).cps == 0.0


def test_words_counts_whitespace_separated_tokens():
    assert sample(1, "en", "hello there friend", 1.0).words == 3


def test_percentile_picks_order_statistics():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert _percentile(values, 0.0) == 1.0
    assert _percentile(values, 1.0) == 5.0
    assert _percentile([], 0.5) == 0.0


def test_linear_fit_recovers_a_known_line():
    # duration = 0.08 * chars + 1.5, which is the shape the real data takes:
    # a per-character articulation cost plus fixed recording overhead.
    chars = [50.0, 100.0, 150.0, 200.0]
    durations = [0.08 * c + 1.5 for c in chars]

    slope, intercept = _linear_fit(chars, durations)

    assert math.isclose(slope, 0.08, rel_tol=1e-9)
    assert math.isclose(intercept, 1.5, rel_tol=1e-9)


def test_linear_fit_is_safe_on_degenerate_input():
    assert _linear_fit([], []) == (0.0, 0.0)
    # Every x identical means no slope is recoverable; fall back to the mean.
    slope, intercept = _linear_fit([5.0, 5.0], [2.0, 4.0])
    assert slope == 0.0
    assert intercept == 3.0


def test_measure_rate_separates_articulation_from_overhead():
    samples = [
        sample(i, "hi", "क" * chars, 0.08 * chars + 1.5)
        for i, chars in enumerate([50, 100, 150, 200])
    ]

    stats = measure_rate(samples)

    assert stats.language == "hi"
    assert stats.num_samples == 4
    assert math.isclose(stats.seconds_per_char, 0.08, rel_tol=1e-6)
    assert math.isclose(stats.intercept_s, 1.5, rel_tol=1e-6)

    # The clip rate is lower than the articulation rate, because the fixed
    # overhead is counted against every clip.
    assert stats.median_cps < 1.0 / stats.seconds_per_char


def test_pairing_joins_on_sentence_id_only():
    source = [sample(1, "en", "one", 1.0), sample(2, "en", "two", 2.0)]
    target = [sample(2, "hi", "दो", 3.0), sample(9, "hi", "nine", 1.0)]

    pairs = pair_by_sentence(source, target)

    assert len(pairs) == 1
    assert pairs[0][0].sentence_id == 2
    assert pairs[0][1].language == "hi"


def test_expansion_measures_the_same_sentence_twice():
    source = [
        sample(1, "en", "a" * 100, 10.0),
        sample(2, "en", "a" * 100, 10.0),
    ]
    target = [
        sample(1, "hi", "क" * 100, 12.0),
        sample(2, "hi", "क" * 120, 13.0),
    ]

    stats = measure_expansion(source, target)

    assert stats.num_pairs == 2
    assert stats.source_language == "en"
    assert stats.target_language == "hi"
    # Ratios are 1.2 and 1.3, so the median sits between them.
    assert 1.2 <= stats.median_duration_ratio <= 1.3
    assert 1.0 <= stats.median_char_ratio <= 1.2


def test_expansion_on_no_overlap_is_empty_not_an_error():
    stats = measure_expansion(
        [sample(1, "en", "x", 1.0)],
        [sample(7, "hi", "य", 1.0)],
    )

    assert stats.num_pairs == 0
    assert stats.median_duration_ratio == 0.0
