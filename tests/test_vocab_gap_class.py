"""
Which missing vocabulary tokens are allowed to stop a synthesis run.

The first real run of `check_arms` refused to synthesize sentence 1 of both
arms because the em dash is outside IndicF5's 2545-token vocabulary. That
refusal was wrong, and it was wrong in the expensive direction: a GPU session
was already open, and the rule as written would also have refused the shipping
`en -> hi` gen text, which carries six em dashes and scored 93% of scale.

The rule it replaced was "any missing token blocks". The rule it did not become
is "punctuation is fine" — an apostrophe is punctuation too, and losing it turns
"isn't" into "isn t", which is a pause inside a word and exactly the failure the
check exists for.

What actually separates them is whether the character already stands between
spaces. An unknown token maps to index 0, which is the space; substituting a
space for something already surrounded by spaces changes a pause into a slightly
different pause. Substituting one inside a word splits the word.

The hyphen appears here because it was the character expected to fail and did
not: `फोर्टी-सेवन` tokenizes at 100%. It is kept as a test because if a future
vocabulary drops it, the answer must be "block" — it is word-internal — and not
"allow, it is punctuation".

No vocabulary file, no f5_tts and no GPU: `stands_alone` is pure.
"""

import pytest

vc = pytest.importorskip("colab.vocab_check")

EM_DASH = "—"

# The real rows, verbatim from the fixtures, so these do not drift apart.
SENTENCE_1_EN = ("You give it a video — a lecture, an interview, anything with "
                 "clear speech — and it gives you back the same video speaking "
                 "Hindi.")
SENTENCE_1_DEVA = ("यू गिव इट अ वीडियो — अ लेक्चर, ऐन इंटरव्यू, एनीथिंग विद क्लियर स्पीच — "
                   "एंड इट गिव्ज़ यू बैक द सेम वीडियो स्पीकिंग हिंदी।")
SENTENCE_5_DEVA = "थर्टी-वन फिट परफेक्टली।"


def test_the_em_dash_that_stopped_the_first_run_does_not_stop_it():
    assert vc.stands_alone(EM_DASH, SENTENCE_1_EN)
    assert vc.stands_alone(EM_DASH, SENTENCE_1_DEVA)


def test_a_word_internal_hyphen_blocks_even_though_it_is_punctuation():
    """
    The case that stops `stands_alone` from collapsing into a category check.
    This hyphen is in vocabulary today, so nothing is blocked now; the test is
    here for the vocabulary that drops it.
    """
    assert not vc.stands_alone("-", SENTENCE_5_DEVA)


def test_an_apostrophe_blocks():
    """`isn't` -> `isn t` is a pause inside a word, which is the real failure."""
    assert not vc.stands_alone("'", "It isn't.")


def test_one_word_internal_occurrence_condemns_the_character():
    """
    A character is judged over the whole string, not per occurrence, because
    the substitution is global. An em dash used correctly twice and jammed
    against a word once still splits that word.
    """
    assert not vc.stands_alone(EM_DASH, "a — b, c—d")


def test_a_character_the_text_does_not_contain_is_vacuously_alone():
    """
    Tokens come from convert_char_to_pinyin, which can emit something the raw
    text does not hold. Such a token must not be treated as word-internal on
    the strength of not being found — the loop simply has nothing to judge.
    """
    assert vc.stands_alone("z", SENTENCE_5_DEVA)


def test_the_danda_blocks_because_it_closes_a_word():
    """Sentence-final `।` sits against the last letter, not between spaces."""
    assert not vc.stands_alone("।", SENTENCE_5_DEVA)


def test_a_character_at_the_very_edges_counts_as_flanked():
    """Start and end of string are treated as whitespace; nothing is split."""
    assert vc.stands_alone(EM_DASH, "— a b —")
