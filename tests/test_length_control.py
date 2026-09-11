"""
Tests for choosing a translation that fits the time available.

The policy trades two quantities against each other, so these tests pin down
what it refuses to trade: it must never ship a short candidate that lost the
meaning just because it fits.
"""

import pytest

from src.stages.translation.length_control import (
    FIDELITY_FLOOR,
    Candidate,
    choose_translation,
    score_candidates,
    select,
)


class FixedRateModel:
    """
    Duration model stand-in: a flat characters-per-second rate.
    """

    def __init__(self, cps=10.0):
        self.cps = cps

    def predict(self, text, language):
        return max(len(text.strip()) / self.cps, 0.05)


def candidate(text, fit_ratio, fidelity=None, budget=2.0):
    return Candidate(
        text=text,
        predicted_duration_s=fit_ratio * budget,
        budget_s=budget,
        fit_ratio=fit_ratio,
        fidelity=fidelity,
    )


def test_scoring_predicts_duration_against_the_budget():
    candidates = score_candidates(
        ["क" * 20, "क" * 40],
        budget_s=2.0,
        language="hi",
        duration_model=FixedRateModel(cps=10.0),
    )

    # 20 chars at 10 cps is 2.0s, exactly the budget.
    assert candidates[0].predicted_duration_s == pytest.approx(2.0)
    assert candidates[0].fit_ratio == pytest.approx(1.0)
    assert candidates[0].fits

    # 40 chars needs twice the budget.
    assert candidates[1].fit_ratio == pytest.approx(2.0)
    assert not candidates[1].fits


def test_fidelity_is_attached_positionally():
    candidates = score_candidates(
        ["a", "b"],
        budget_s=1.0,
        language="hi",
        duration_model=FixedRateModel(),
        fidelities=[0.9, 0.7],
    )

    assert candidates[0].fidelity == 0.9
    assert candidates[1].fidelity == 0.7


def test_a_fitting_faithful_candidate_wins():
    selection = select(
        [
            candidate("long but faithful", 1.6, fidelity=0.95),
            candidate("short and faithful", 0.8, fidelity=0.90),
        ]
    )

    assert selection.chosen.text == "short and faithful"
    assert selection.reason == "fits the budget"


def test_among_fitting_candidates_the_most_faithful_wins():
    # Both fit comfortably, so there is no reason to sacrifice meaning.
    selection = select(
        [
            candidate("less faithful", 0.5, fidelity=0.84),
            candidate("more faithful", 0.52, fidelity=0.97),
        ]
    )

    assert selection.chosen.text == "more faithful"


def test_an_unfaithful_short_candidate_is_refused():
    # This is the failure the policy exists to prevent: a candidate that fits
    # beautifully because it threw away half the sentence.
    selection = select(
        [
            candidate("full meaning", 1.4, fidelity=0.95),
            candidate("gutted", 0.4, fidelity=0.55),
        ]
    )

    assert selection.chosen.text == "full meaning"
    assert "nothing fit" in selection.reason


def test_when_nothing_fits_the_closest_faithful_candidate_wins():
    selection = select(
        [
            candidate("way too long", 2.0, fidelity=0.93),
            candidate("only a bit long", 1.1, fidelity=0.88),
        ]
    )

    assert selection.chosen.text == "only a bit long"
    assert "closest" in selection.reason


def test_when_nothing_is_faithful_the_most_faithful_is_kept():
    selection = select(
        [
            candidate("bad", 0.5, fidelity=0.4),
            candidate("less bad", 0.6, fidelity=0.7),
        ]
    )

    assert selection.chosen.text == "less bad"
    assert "fidelity floor" in selection.reason
    # A correctness problem must not be disguised as a solved timing problem.
    assert selection.chosen.fidelity < FIDELITY_FLOOR


def test_without_fidelity_scores_length_decides_alone():
    selection = select([candidate("long", 1.5), candidate("short", 0.7)])

    assert selection.chosen.text == "short"
    assert selection.chosen.fidelity is None


def test_the_baseline_is_the_first_candidate():
    selection = select(
        [
            candidate("beam search favourite", 1.5, fidelity=0.96),
            candidate("shorter alternative", 0.9, fidelity=0.89),
        ]
    )

    assert selection.baseline.text == "beam search favourite"
    assert selection.chosen.text == "shorter alternative"

    # The tradeoff is measurable in both directions.
    assert selection.fit_gain == pytest.approx(0.6)
    assert selection.fidelity_cost == pytest.approx(0.07)


def test_choosing_the_baseline_costs_nothing():
    selection = select(
        [
            candidate("already fits", 0.8, fidelity=0.95),
            candidate("worse", 0.9, fidelity=0.81),
        ]
    )

    assert selection.chosen is selection.baseline
    assert selection.fidelity_cost == 0.0
    assert selection.fit_gain == 0.0


def test_end_to_end_selection_from_raw_texts():
    selection = choose_translation(
        ["क" * 40, "क" * 18],
        budget_s=2.0,
        language="hi",
        duration_model=FixedRateModel(cps=10.0),
        fidelities=[0.95, 0.88],
    )

    assert selection.chosen.text == "क" * 18
    assert selection.num_candidates == 2


def test_selecting_from_nothing_is_an_error():
    with pytest.raises(ValueError, match="at least one candidate"):
        select([])
