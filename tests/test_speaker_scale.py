"""
The scale is the instrument every voice result in this project is read on, and
it has been wrong three times: once on language, once on clip duration, and
once by quoting a target that sat inside its own noise. Its arithmetic is pure
and gets pinned here even though building it needs a GPU.
"""

import math

import pytest

from colab.speaker_scale import Scale, ceiling_for, position

# duration in seconds, ceiling — the shape actually measured on this speaker
CURVE = [(3.6, 0.847), (5.0, 0.881), (8.3, 0.923), (16.7, 0.968)]


def test_a_clip_is_judged_against_its_own_length():
    assert ceiling_for(5.1, CURVE) == 0.881
    assert ceiling_for(16.0, CURVE) == 0.968


def test_a_clip_outside_the_measured_range_takes_the_nearest_point():
    assert ceiling_for(0.5, CURVE) == 0.847
    assert ceiling_for(60.0, CURVE) == 0.968


def test_the_floor_is_zero_percent_and_the_ceiling_is_one_hundred():
    assert position(0.095, 5.0, CURVE, floor=0.095) == pytest.approx(0.0)
    assert position(0.881, 5.0, CURVE, floor=0.095) == pytest.approx(100.0)


def test_the_same_score_places_higher_on_a_short_clip():
    """
    The error that inflated the four-arm result: a short clip scored against a
    ceiling measured on 17 second pieces looks worse than it is.
    """
    short = position(0.80, 3.6, CURVE, floor=0.095)
    long = position(0.80, 16.7, CURVE, floor=0.095)
    assert short > long


def test_an_empty_curve_is_not_an_answer():
    assert math.isnan(ceiling_for(5.0, []))
    assert math.isnan(position(0.8, 5.0, [], floor=0.095))


def test_a_floor_above_the_ceiling_refuses_rather_than_inverting():
    assert math.isnan(position(0.8, 5.0, CURVE, floor=0.99))


def test_a_missing_score_propagates():
    assert math.isnan(position(float("nan"), 5.0, CURVE, floor=0.095))


def test_the_dataclass_routes_to_the_named_curve():
    scale = Scale(floor=0.095, curves={"same": CURVE, "cross": [(5.0, 0.825)]})
    assert scale.ceiling(5.0, "same") == 0.881
    assert scale.ceiling(5.0, "cross") == 0.825
    assert scale.position(0.881, 5.0, "same") == pytest.approx(100.0)
