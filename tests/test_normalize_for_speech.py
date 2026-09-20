"""
Tests for the text normaliser that runs before synthesis.

These pin a measured failure, not a style preference. A clean GPU run — 30/30
segments, every clip on budget, assembly tempo 1.00 — produced two videos with
gibberish in them, and transcribing the output located it exactly:

    asked  लगभग 20% अधिक समय        heard  लगभग लतक अधिक समय
    asked  उसी 3 सेकंड में           heard  उसी इदस सेकंड में
    asked  ने 47 खंडों को            heard  ने आत्तखंडो को

Nine of the ten segments containing a digit were corrupted; of the segments
without one, all but a single unexplained row were clean. Two accidental
controls in the same run settle the mechanism: `तीन सिकंड` and `छह अलग-अलग`
reached the model spelled as words and came back perfect. IndicF5 cannot say a
digit, and the fix belongs in the text.

Digits also take the segment head with them — `यू`, `ते`, `एंड` appeared before
the first real word on exactly the digit-bearing rows — because a token the
model cannot align desynchronises the unanchored `ref_audio_len` slice
(§5). So these tests guard audio quality in two places at once.

"""

import pytest

from src.text.normalize import (
    normalize_for_speech,
    unspeakable_digits,
)


# -- the numbers that were actually corrupted ---------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("लगभग 20% अधिक समय", "लगभग बीस प्रतिशत अधिक समय"),
        ("उसी 3 सेकंड में", "उसी तीन सेकंड में"),
        ("पिछले हफ्ते सिस्टम ने 47 खंडों को संसाधित किया",
         "पिछले हफ्ते सिस्टम ने सैंतालीस खंडों को संसाधित किया"),
        ("7 असंभव थे", "सात असंभव थे"),
        ("9 को खिंचना पड़ा", "नौ को खिंचना पड़ा"),
        ("हम हर वाके के लिए 6 अलग अनुवाद बनाते हैं",
         "हम हर वाके के लिए छह अलग अनुवाद बनाते हैं"),
    ],
)
def test_every_corrupted_number_from_the_run_is_spelled_out(text, expected):
    assert normalize_for_speech(text, "hi") == expected


def test_two_numbers_in_one_segment_are_both_spelled():
    """en_dub 11 lost both of its numbers to a single `एंड फिट`."""
    out = normalize_for_speech("31 फीट 9 पूरी तरह से खींचने की जरूरत है ।", "hi")

    assert "इकतीस" in out and "नौ" in out
    assert not unspeakable_digits(out)


# -- the controls, which must not move ----------------------------------------


@pytest.mark.parametrize("text", ["तीन सिकंड में आने चाहिए", "छह अलग अलग अनुवाद"])
def test_a_number_already_spelled_as_a_word_is_untouched(text):
    """
    Both of these synthesized perfectly in the same run that mangled every
    digit. Rewriting them would risk a measured-good result.
    """
    assert normalize_for_speech(text, "hi") == text


def test_text_with_no_digits_is_returned_unchanged():
    text = "और ये आपको वही वीडियो देता है जो हिंदी में बोल रहा है ।"

    assert normalize_for_speech(text, "hi") == text


def test_the_danda_survives():
    """Sentence-final punctuation is not this function's business."""
    assert normalize_for_speech("सात असंभव थे ।", "hi").endswith("।")


# -- shape --------------------------------------------------------------------


def test_the_loose_spacing_spell_numbers_leaves_is_collapsed():
    """
    `spell_numbers` pads its replacements and documents that the caller
    tidies up. Double spaces are a second token boundary to the model.
    """
    out = normalize_for_speech("लगभग 20% अधिक", "hi")

    assert "  " not in out
    assert not out.startswith(" ") and not out.endswith(" ")


def test_a_standalone_hyphen_is_dropped():
    """
    IndicTrans2 emits `-` as a clause separator — `सरल सही -`, `अलग - अलग`.
    A lone dash between spaces is not something a speaker says.
    """
    assert normalize_for_speech("सरल सही -", "hi") == "सरल सही"
    assert normalize_for_speech("छह अलग - अलग अनुवाद", "hi") == "छह अलग अलग अनुवाद"


def test_a_hyphen_inside_a_word_is_kept():
    """
    `fixtures/xlit/deva_hand.json` mirrors the English hyphen in number
    compounds — फोर्टी-सेवन — and the speaker reviewed that spelling. Only a
    free-standing dash goes.
    """
    assert normalize_for_speech("फोर्टी-सेवन सेगमेंट", "hi") == "फोर्टी-सेवन सेगमेंट"


def test_normalizing_twice_changes_nothing():
    once = normalize_for_speech("लगभग 20% अधिक समय ।", "hi")

    assert normalize_for_speech(once, "hi") == once


def test_empty_and_none_are_survivable():
    assert normalize_for_speech("", "hi") == ""
    assert normalize_for_speech(None, "hi") == ""


def test_devanagari_digits_are_spelled_too():
    """Whisper writes a Hindi number either way depending on the clip."""
    assert normalize_for_speech("४७ खंडों", "hi") == "सैंतालीस खंडों"


# -- the guard ----------------------------------------------------------------


def test_a_number_past_a_hundred_is_spoken_rather_than_refused():
    """
    This used to assert the opposite: 150 came back as `["150"]` and the
    export refused the bundle. That was correct while `numbers.py` stopped at
    99 — a refusal beats a mispronunciation — but it meant one price or one
    year in an hour of video blocked the whole job.
    """
    out = normalize_for_speech("150 सेगमेंट", "hi")

    assert out == "एक सौ पचास सेगमेंट"
    assert not unspeakable_digits(out)


@pytest.mark.parametrize(
    "text,expected",
    [
        # The shapes a real video actually contains.
        ("150 सेगमेंट", "एक सौ पचास सेगमेंट"),
        ("2,50,000 लोग", "दो लाख पचास हज़ार लोग"),
        ("1,500 रुपये", "एक हज़ार पाँच सौ रुपये"),
        ("10000000 रुपये", "एक करोड़ रुपये"),
        ("3.5 गुना तेज", "तीन दशमलव पाँच गुना तेज"),
        ("1947 में", "उन्नीस सौ सैंतालीस में"),
        ("2024 में", "दो हज़ार चौबीस में"),
        ("1500 मीटर", "पंद्रह सौ मीटर"),
    ],
)
def test_the_numbers_a_real_video_contains(text, expected):
    assert normalize_for_speech(text, "hi") == expected


def test_a_long_run_is_read_out_not_totalled():
    """
    Nobody reads a phone number as नौ सौ सतासी करोड़. Past
    IDENTIFIER_DIGITS every digit is spoken on its own.
    """
    out = normalize_for_speech("कॉल करें 9876543210", "hi")

    assert out == "कॉल करें नौ आठ सात छह पाँच चार तीन दो एक शून्य"


def test_a_round_number_stays_a_quantity_however_long_it_is():
    """
    The length rule alone would read एक करोड़ out digit by digit. Trailing
    zeros are what separate a crore from a phone number.
    """
    assert normalize_for_speech("10000000 रुपये", "hi") == "एक करोड़ रुपये"
    assert normalize_for_speech("1,00,00,000 रुपये", "hi") == "एक करोड़ रुपये"


def test_a_leading_zero_is_never_a_quantity():
    assert normalize_for_speech("007 एजेंट", "hi") == "शून्य शून्य सात एजेंट"


def test_normalisation_is_still_idempotent_past_the_old_ceiling():
    """
    Three code paths produce a translation and each normalises separately, so
    applying it twice must not change the answer. Worth re-pinning here
    because the new rules read the *written* form, which the first pass
    rewrites.
    """
    once = normalize_for_speech("1,500 रुपये और 1947 में", "hi")

    assert normalize_for_speech(once, "hi") == once


def test_the_guard_is_quiet_on_clean_text():
    assert unspeakable_digits("सात असंभव थे ।") == []
    assert unspeakable_digits("") == []


def test_the_guard_reports_every_offender_not_just_the_first():
    assert unspeakable_digits("150 और 200 सेगमेंट") == ["150", "200"]
