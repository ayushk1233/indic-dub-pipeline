"""
Two spellings of the same number must not be scored as an error.

This has now cost two separate measurements.

  - The slicer matched "Thirty-one fit perfectly." to `fit perfectly,` alone,
    because Whisper wrote "31" and a letters-only key shares nothing with
    "Thirty-one". It cut a 0.92 s clip: 27 characters per second, faster than
    anything this project has measured from a human or a model.
  - The transliteration probe scored `31 feet perfectly.` at CER 0.458 against
    "Thirty-one fit perfectly." The only thing the model got wrong is `feet`
    for `fit`; the rest of that 0.458 is the numeral. Four of the seven probe
    sentences contain a number, so it moved an arm's headline score.

The two callers want different shapes — the slicer strips to alphanumerics,
the scorer keeps word boundaries — so the shared piece is the spelling and
nothing else.
"""

import pytest

numbers = pytest.importorskip("src.text.numbers")


def flat(text):
    return " ".join(numbers.spell_numbers(text).split())


def test_the_two_failures_that_actually_happened():
    assert flat("31 feet perfectly.") == "thirty one feet perfectly."
    assert flat("47 segments") == "forty seven segments"


def test_percent_becomes_a_word():
    """Whisper writes `20%` where the script writes `twenty percent`."""
    assert flat("about 20% longer") == "about twenty percent longer"


def test_a_number_already_in_words_is_untouched():
    """Both sides must land on the same string, from either direction."""
    assert flat("Thirty-one fit") == "Thirty-one fit"


def test_the_scorer_and_the_script_agree_after_normalising():
    """The end-to-end property, through the scorer's own normaliser."""
    check = pytest.importorskip("colab.indicf5_check")

    assert check.normalize("Thirty-one fit perfectly.") == \
        check.normalize("31 fit perfectly")
    assert check.normalize("Seven were impossible, and we had to rewrite them.") == \
        check.normalize("7 were impossible and we had to rewrite them.")
    assert check.normalize("about twenty percent longer") == \
        check.normalize("about 20% longer")


def test_a_real_error_still_scores_as_one():
    """
    The guard against over-normalising. `feet` for `fit` is the model, and it
    must survive: a normaliser that made this pair equal would be hiding the
    only real error in that clip.
    """
    check = pytest.importorskip("colab.indicf5_check")

    assert check.normalize("Thirty-one fit perfectly.") != \
        check.normalize("31 feet perfectly.")

    scored = check.score_text("Thirty-one fit perfectly.", "31 feet perfectly.")
    assert 0 < scored["cer"] < 0.2, "the numeral was most of the old 0.458"


def test_numbers_past_the_table_are_left_alone():
    assert flat("4096 things") == "4096 things"


def test_hundreds_are_spelled_both_ways_the_same():
    assert flat("100") == flat("one hundred")
    assert flat("247") == "two hundred forty seven"
