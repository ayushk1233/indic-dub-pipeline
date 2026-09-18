"""
The accent dials must change the stops and nothing else.

FINDINGS §4d ruled out the reference pair as the cause of the accent overshoot:
transliterating the reference transcript reached the Whisper content floor and
moved the accent not at all. That leaves the orthography, and these are the
dials on it — retroflex to dental, and word-initial aspiration where English
aspirates.

The dials are blanket character substitutions over reviewed text, which is only
safe because the corpus is transliterated English throughout. That makes the
risk over-application rather than under-application: a rule that also fires on a
cluster onset, on an unstressed function word, or on a character that means
something else produces text that is still 100% Devanagari, still passes the
gate, still synthesizes, and is simply the wrong sound. Every test here pins a
position the dial must NOT fire in.

No audio, no model, no GPU.
"""

import json
import unicodedata

import pytest

from colab.indicf5_english import devanagari_fraction
from src.text.en_to_deva import (
    ASPIRATE,
    DENTAL,
    DIALS,
    UNSTRESSED,
    aspirate_initial,
    dental,
    soft,
)

XLIT = "fixtures/xlit/deva_hand.json"


@pytest.fixture
def reviewed():
    """The seven sentences the speaker actually signed off on."""
    data = json.loads(open(XLIT, encoding="utf-8").read())
    return {s["id"]: s["devanagari"] for s in data["sentences"]}


# --------------------------------------------------------------- dental


def test_dental_moves_every_retroflex_stop(reviewed):
    """41 ट and 18 ड in the fixtures — this is most of the consonants."""
    after = dental(" ".join(reviewed.values()))
    assert not set(after) & set(DENTAL)


def test_dental_leaves_everything_else_alone(reviewed):
    """Same length, same vowels, same word count: only stops move."""
    for text in reviewed.values():
        after = dental(text)
        assert len(after) == len(text)
        assert len(after.split()) == len(text.split())
        assert devanagari_fraction(after) == devanagari_fraction(text)
        # every position either held a retroflex stop or is untouched
        for before_char, after_char in zip(text, after):
            if before_char in DENTAL:
                assert after_char == DENTAL[before_char]
            else:
                assert after_char == before_char


def test_dental_is_reversible_in_count(reviewed):
    """
    Every substituted character has exactly one replacement, so the arm's
    duration arithmetic is unchanged — `fix_duration` is set from the slot, but
    the text gate reports chars and bytes and a silent length change there
    would look like a drafting error.
    """
    text = reviewed[0]
    assert len(dental(text).encode("utf-8")) == len(text.encode("utf-8"))


# ----------------------------------------------------------- aspiration


def test_a_stressed_word_initial_stop_is_aspirated():
    """English `tell` is [tʰɛl]; Hindi's plain ट is not."""
    assert aspirate_initial("टेल") == "ठेल"
    assert aspirate_initial(dental("टेल")) == "थेल"


def test_a_cluster_onset_is_left_alone():
    """
    `क्लियर`, `ट्वेंटी`. English reduces aspiration in /kl/ and /tw/ onsets to
    devoicing of the following sonorant, which Devanagari cannot write at all,
    so the unaspirated spelling is the closer of the two available.
    """
    assert aspirate_initial("क्लियर") == "क्लियर"
    assert aspirate_initial("ट्वेंटी") == "ट्वेंटी"


def test_an_unstressed_function_word_keeps_its_plain_stop():
    """
    `to` is [tə] and never [tʰuː]. Position alone cannot see stress, so this
    is the exception list doing its job — and `टू` occurs four times in the
    seven sentences, so getting it wrong is not a corner case.
    """
    assert aspirate_initial("टू") == "टू"
    assert aspirate_initial(dental("टू")) == "तू"
    assert soft("टू स्पीक") == "तू स्पीक"


def test_a_stop_inside_a_word_is_left_alone():
    """`सिस्टम`, `स्पीच` — English does not aspirate /t/ after /s/ either."""
    assert "ठ" not in aspirate_initial("सिस्टम")
    assert aspirate_initial(dental("सिस्टम")) == dental("सिस्टम")


def test_p_is_never_aspirated():
    """
    The one omission that is a finding rather than an oversight. फ is read as
    /f/ in modern Hindi, not /pʰ/, so aspirating `project` through it would
    come back as `froject` — a worse error than the unaspirated stop it
    replaced. There is no way to write an aspirated /p/ this model will read
    as one.
    """
    assert "प" not in ASPIRATE
    assert soft("प्रोजेक्ट") == dental("प्रोजेक्ट")
    # word-initial and not a cluster — the one position the rule would fire in
    assert soft("परफेक्टली") == dental("परफेक्टली")
    assert soft("परफेक्टली").startswith("प")


def test_aspiration_does_not_cross_punctuation():
    """
    Words are split on spaces, the em dash, the comma and the danda, all of
    which appear in the fixtures. A `—` treated as a word character would make
    the following word non-initial and silently disable the dial there.
    """
    assert soft("एंड — टेक") == "एंद — थेक"
    assert soft("वन, टेक") == "वन, थेक"
    assert soft("टेक।") == "थेक।"


# ------------------------------------------------------- over the corpus


@pytest.mark.parametrize("name", sorted(DIALS))
def test_every_dial_keeps_the_text_synthesizable(reviewed, name):
    """
    A dial produces text that goes straight to the model, so it has to clear
    the same gate a hand transliteration does. A dial that emitted a Latin
    character or broke NFC would fail at synthesis, not here.
    """
    probe = pytest.importorskip("colab.indicf5_xlit_probe")

    for text in reviewed.values():
        after = DIALS[name](text)
        assert devanagari_fraction(after) == 1.0
        assert unicodedata.normalize("NFC", after) == after
        assert probe.check_text(after) == []


@pytest.mark.parametrize("name", sorted(DIALS))
def test_every_dial_actually_changes_something(reviewed, name):
    """
    A dial that silently did nothing would run as an arm, cost a third of a
    session, and report a difference of exactly zero that reads as `the
    orthography does not matter`.
    """
    changed = [i for i, text in reviewed.items() if DIALS[name](text) != text]
    assert len(changed) == len(reviewed), f"{name} left {set(reviewed) - set(changed)} alone"


def test_the_probe_derives_dial_arms_from_the_reviewed_fixture(reviewed):
    """
    The arms are text plus a named function, not a second hand-maintained JSON
    file that can drift from the first. This is that contract.
    """
    probe = pytest.importorskip("colab.indicf5_xlit_probe")

    for name in DIALS:
        assert probe.ARM_TEXT[name] == "deva_hand"
        assert probe.arm_generated_text(name, reviewed) == {
            i: DIALS[name](t) for i, t in reviewed.items()}

    # and an arm that is not a dial passes its text through untouched
    assert probe.arm_generated_text("deva_ref", reviewed) == reviewed


def test_the_dials_are_reported_and_not_mistaken_for_the_floor():
    """
    SEED SPREAD and the VERDICT read REPORTED_ARMS. A dial missing from it
    would synthesize, appear in the CONTENT table, and then be left out of
    every comparison that decides anything.
    """
    probe = pytest.importorskip("colab.indicf5_xlit_probe")

    for name in DIALS:
        assert name in probe.REPORTED_ARMS
    assert "floor" not in probe.REPORTED_ARMS
    assert "latin" not in probe.REPORTED_ARMS


def test_unstressed_entries_cover_both_spellings():
    """
    The dials compose in either order in principle, so an exception listed
    only in its retroflex form would leak through `soft`, which dentalises
    first.
    """
    assert {dental(word) for word in UNSTRESSED} <= UNSTRESSED
