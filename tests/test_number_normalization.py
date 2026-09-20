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


def key(text):
    """
    What a scorer compares: numbers spelled out, case and punctuation gone.

    Written here against `jiwer` rather than borrowed from a scoring module,
    so this file depends only on what the pipeline itself ships.
    """
    import re

    spelled = numbers.spell_numbers(text)

    return " ".join(re.sub(r"[^\w\s]", " ", spelled.lower()).split())


def test_the_scorer_and_the_script_agree_after_normalising():
    """The end-to-end property: both spellings land on one string."""
    assert key("Thirty-one fit perfectly.") == key("31 fit perfectly")
    assert key("Seven were impossible, and we had to rewrite them.") == \
        key("7 were impossible and we had to rewrite them.")
    assert key("about twenty percent longer") == key("about 20% longer")


def test_a_real_error_still_scores_as_one():
    """
    The guard against over-normalising. `feet` for `fit` is the model, and it
    must survive: a normaliser that made this pair equal would be hiding the
    only real error in that clip.
    """
    import jiwer

    assert key("Thirty-one fit perfectly.") != key("31 feet perfectly.")

    cer = jiwer.cer(key("Thirty-one fit perfectly."), key("31 feet perfectly."))

    assert 0 < cer < 0.2, "the numeral was most of the old 0.458"


def test_numbers_past_a_hundred_are_spelled_too():
    """
    This asserted `flat("4096 things") == "4096 things"` while the table
    stopped at 99. Both sides of a CER comparison went through the same
    function, so leaving digits alone was self-consistent — it just meant
    Whisper's `4096` never collided with a script's "four thousand ninety
    six".
    """
    assert flat("4096 things") == "four thousand ninety six things"
    assert flat("4096 things") == flat("four thousand ninety six things")


def test_a_year_collides_with_the_way_a_script_writes_it():
    """
    The point of the year rule, from the alignment side: a transcript writes
    "nineteen forty-seven" and Whisper writes "1947".
    """
    assert flat("1947") == flat("nineteen forty seven")
    assert flat("in 1900") == "in nineteen hundred"


def test_indian_grouping_survives_the_commas():
    """
    `1,500` used to spell the `1` and leave the `500`, giving "one , 500" —
    wrong on both halves, and it failed at export blaming the 500.
    """
    assert flat("1,500 rupees") == "one thousand five hundred rupees"


def test_a_comma_is_evidence_that_a_quantity_was_meant():
    """
    The two spellings deliberately diverge. A bare 1500 is read as the year
    form, fifteen hundred; a grouped 1,500 carries its author's own evidence
    that a quantity was meant.

    The cost is recorded rather than hidden: a script writing "one thousand
    five hundred" does not collide with Whisper's `1500`. That is the price of
    reading years the way people say them, and years are far commoner in
    speech than four-digit quantities written without a comma.
    """
    assert flat("1500") == "fifteen hundred"
    assert flat("1,500") == "one thousand five hundred"


def test_two_numbers_separated_by_a_comma_stay_two_numbers():
    """The grouping rule must not glue `item 1,2` into twelve."""
    assert flat("item 1,2") == "item one , two"


def test_a_decimal_is_read_the_way_it_is_spoken():
    assert flat("3.5 times") == "three point five times"
    assert flat("3.14") == "three point one four"


def test_hundreds_are_spelled_both_ways_the_same():
    assert flat("100") == flat("one hundred")
    assert flat("247") == "two hundred forty seven"


# -- rescued from tests/test_hindi_probe.py -----------------------------------
#
# The Hindi cardinal table lives in src/text/numbers.py and is called by the
# shipping normaliser, but its tests were written inside the hi -> hi probe's
# file. These four came across when the research-probe tests were removed.


def test_the_table_covers_zero_to_a_hundred_without_repeating_itself():
    from src.text.numbers import _HI

    assert len(_HI) == 101
    assert len(set(_HI)) == 101


def test_every_number_word_in_the_script_is_in_the_table():
    """
    The collision guard. A wrong or missing entry does not raise — it scores
    the number as a substitution and inflates the CER of whichever sentence
    contains it, on a GPU, hours later.
    """
    import json
    from pathlib import Path

    from src.text.numbers import _HI

    text = json.loads(
        Path("fixtures/metadata.json").read_text(encoding="utf-8")
    )["hindi"]["scripted_text"]

    for mark in "—।,?":
        text = text.replace(mark, " ")

    found = sorted({t for t in text.split() if t in _HI})

    assert found == ["इकतीस", "एक", "छह", "तीन", "नौ", "बीस", "सात", "सैंतालीस"]


def test_the_language_is_read_off_the_intended_text_not_each_string():
    """
    Inferring per string would spell the reference in English and a
    Devanagari transcript of it in Hindi, and score every number as a
    substitution — the error this normaliser exists to remove.
    """
    from src.text.numbers import script_language, spell_numbers

    assert script_language("इकतीस बिल्कुल") == "hi"
    assert script_language("Thirty-one fit") == "en"
    # digits alone carry no script, so the caller's choice has to win
    assert spell_numbers("31", "hi").strip() == "इकतीस"
    assert spell_numbers("31", "en").strip() == "thirty one"


def test_each_language_uses_its_own_grouping_past_the_table():
    """
    Hindi counts in लाख and करोड़, English in thousands and millions, and the
    multiplier recurses so there is no ceiling above करोड़.
    """
    from src.text.numbers import number_words

    assert number_words(101, "hi") == "एक सौ एक"
    assert number_words(1000, "en") == "one thousand"
    assert number_words(250000, "hi") == "दो लाख पचास हज़ार"
    assert number_words(10_000_000, "hi") == "एक करोड़"
    assert number_words(250000, "en") == "two hundred fifty thousand"
    assert number_words(120_000_000, "hi") == "बारह करोड़"
