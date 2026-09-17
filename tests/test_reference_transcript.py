"""
The first probe reported that script caused the prefix. It had not changed the
script: Whisper, asked to read English "in Hindi", returned English in Roman
letters, and the arm was labelled en_deva and compared against a control it
differed from only in punctuation.

132 Latin characters and 132 Devanagari characters look identical in a report.
The bytes do not.
"""

from colab.indicf5_english import devanagari_fraction, plain

WHISPER_OUTPUT = ("so let me tell you what this project actually does you get "
                  "a video lecture and interview anything with clear speech")
HINDI = "मैं आपको बताता हूँ कि यह प्रोजेक्ट असल में क्या करता है"


def test_the_transcript_that_fooled_the_first_probe():
    assert devanagari_fraction(WHISPER_OUTPUT) == 0.0


def test_real_devanagari_passes():
    assert devanagari_fraction(HINDI) == 1.0


def test_code_mixing_lands_between():
    """hi_hi's own transcript came back with 'lecture' and 'interview' in Latin."""
    mixed = devanagari_fraction("कोई lecture कोई interview कुछ भी")
    assert 0.0 < mixed < 1.0


def test_digits_and_punctuation_do_not_count_as_script():
    assert devanagari_fraction("47 — , . 123") == 0.0
    assert devanagari_fraction("सात 47 — ,") == 1.0


def test_an_empty_string_is_not_devanagari():
    assert devanagari_fraction("") == 0.0


def test_byte_length_is_what_separates_them():
    """The check the report now prints, stated as the property it relies on."""
    assert len(WHISPER_OUTPUT.encode("utf-8")) == len(WHISPER_OUTPUT)
    assert len(HINDI.encode("utf-8")) > 2 * len(HINDI)


def test_plain_strips_punctuation_and_case():
    assert plain("So let me tell you. You get a video, lecture and interview.") \
        == "so let me tell you you get a video lecture and interview"


def test_plain_changes_nothing_else():
    assert plain("so let me tell you") == "so let me tell you"


def test_plain_leaves_devanagari_alone():
    assert plain(HINDI) == HINDI.lower()
