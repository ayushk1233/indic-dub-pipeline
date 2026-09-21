"""
English words, written in Devanagari, for the reference transcript.

The reference clip on an `en -> hi` job is the speaker's own English, so the
transcript that describes it has to carry English words in the target script.
That is a third operation the pipeline did not have: translation changes the
words, transcription keeps the script, and only transliteration changes the
letters alone.

The assembly tests below need no g2p and no network — they drive
`phonemes_to_devanagari` with phoneme lists directly, which is where all the
abugida logic lives. Tests that need CMUdict are marked and skip without it.
"""

import pytest

from src.text.transliterate import (
    ANUSVARA,
    VIRAMA,
    phonemes_to_devanagari,
    transliterate_to_devanagari,
)


# -- the abugida rules, which is where this can go wrong ----------------------


def test_a_consonant_carries_its_inherent_vowel():
    """`but` is बट, not बअट. A bare consonant already says the schwa."""
    assert phonemes_to_devanagari(["B", "AH1", "T"]) == "बट"


def test_a_vowel_after_a_consonant_becomes_a_matra():
    assert phonemes_to_devanagari(["M", "IY1"]) == "मी"


def test_a_word_initial_vowel_uses_the_independent_letter():
    """A matra has nothing to hang off at the start of a word."""
    assert phonemes_to_devanagari(["AE1", "T"]).startswith("ऐ")


def test_a_consonant_cluster_takes_a_virama():
    """`speak` is स्पीक — the virama is what makes /sp/ read as a cluster."""
    assert phonemes_to_devanagari(["S", "P", "IY1", "K"]) == "स्पीक"


def test_a_final_consonant_is_written_bare():
    """Hindi writes प्रोजेक्ट, not प्रोजेक्ट्."""
    out = phonemes_to_devanagari(["P", "R", "AA1", "JH", "EH0", "K", "T"])

    assert not out.endswith(VIRAMA)
    assert out == "प्राजेक्ट"


def test_stress_digits_are_dropped():
    """Devanagari does not write stress, and `AA1` is not a phoneme."""
    assert phonemes_to_devanagari(["B", "AH1", "T"]) == \
        phonemes_to_devanagari(["B", "AH", "T"])


def test_the_velar_nasal_keeps_its_g_unless_another_velar_follows():
    """
    `meaning` is मीनिंग and `think` is थिंक. The anusvara takes its place of
    articulation from what follows, so a following velar already supplies it
    and a written ग would be one consonant too many.
    """
    assert phonemes_to_devanagari(["M", "IY1", "N", "IH0", "NG"]) == "मीनिंग"
    assert phonemes_to_devanagari(["TH", "IH1", "NG", "K"]) == "थिंक"


def test_english_t_and_d_take_the_retroflex_series():
    """
    The loanword convention, and deliberate. It is also what makes a cloned
    English accent sound the way it does — but the reference transcript has to
    match how the model reads Devanagari, not what a phonetician would prefer.
    """
    assert phonemes_to_devanagari(["T", "AH0"]) == "ट"
    assert phonemes_to_devanagari(["D", "AH0"]) == "ड"


def test_an_unknown_phoneme_is_skipped_rather_than_raising():
    """g2p emits a stress-only or punctuation token now and then."""
    assert phonemes_to_devanagari(["B", "???", "AH1", "T"]) == "बट"


def test_no_phonemes_is_empty_not_an_error():
    assert phonemes_to_devanagari([]) == ""


# -- the text level -----------------------------------------------------------


g2p = pytest.importorskip("g2p_en", reason="g2p_en not installed")


def test_the_real_reference_sentence_comes_back_in_devanagari():
    """
    The exact sentence that blocked every en -> hi run, and the reason this
    module exists. It must clear the export guard's script threshold.
    """
    from src.eval.translation_metrics import script_ratio
    from src.stages.tts.bundle.exporter import MIN_REFERENCE_SCRIPT_RATIO

    out = transliterate_to_devanagari(
        "So let me tell you what this project actually does.")

    assert script_ratio(out, "hi") >= MIN_REFERENCE_SCRIPT_RATIO
    assert out.startswith("सो लेट मी टेल यू")


def test_spacing_and_punctuation_survive():
    out = transliterate_to_devanagari("you get a video, a lecture.")

    assert out.endswith(".")
    assert "," in out
    assert len(out.split()) == 6


def test_devanagari_already_present_is_left_alone():
    """A transcript can be code-mixed before it reaches here."""
    assert transliterate_to_devanagari("मैं project") .startswith("मैं ")


def test_empty_and_none_are_survivable():
    assert transliterate_to_devanagari("") == ""
    assert transliterate_to_devanagari(None) == ""


def test_the_output_is_about_as_long_as_the_input():
    """
    Not a style check — the export refuses a reference transcript over 25
    characters per second, so a transliterator that inflated the text would
    trade one refusal for another.
    """
    source = ("So let me tell you what this project actually does. "
              "You get a video, a lecture or an interview.")

    assert len(transliterate_to_devanagari(source)) < len(source) * 1.15


# -- a nasal before a stop is an anusvara, not a conjunct ---------------------
#
# This was wrong until it was measured against a real reference transcript:
# ANUSVARA was reachable only from the NG branch, so every other nasal fell
# through to the generic consonant path and took a virama. `and` came out
# अन्ड, `number` नम्बर, `content` कान्टेन्ट. The reference transcript is the
# one input IndicF5 conditions on, and it was being handed an orthography
# Hindi does not use.


def test_a_nasal_before_a_stop_becomes_an_anusvara():
    """
    `and` is एंड and `number` is नंबर. Hindi writes the nasal as an anusvara on
    the syllable before it rather than spelling the cluster out, and English
    loanwords follow that convention.
    """
    assert phonemes_to_devanagari(["AH0", "N", "D"]) == "अंड"
    assert phonemes_to_devanagari(["N", "AH1", "M", "B", "ER0"]) == "नंबर"
    assert VIRAMA not in phonemes_to_devanagari(["P", "OY1", "N", "T"])


def test_the_anusvara_also_applies_before_fricatives():
    """`answer` and `month` take it too — not only the stops."""
    assert phonemes_to_devanagari(["M", "AH1", "N", "TH"]) == "मंथ"
    assert ANUSVARA in phonemes_to_devanagari(["AE1", "N", "S", "ER0"])


def test_a_nasal_before_another_nasal_keeps_its_own_letter():
    """
    `unknown` is अननोन, never अंनोन. An anusvara takes its place of
    articulation from what follows, and a following nasal has nothing
    distinct to give it.
    """
    out = phonemes_to_devanagari(["AH0", "N", "N", "OW1", "N"])

    assert ANUSVARA not in out


def test_a_nasal_before_a_semivowel_or_h_keeps_its_own_letter():
    """
    `only`, `annual`, `convert` and `inherit` are ओनली, ऐन्युअल, कन्वर्ट and
    इनहेरिट — य र ल व and ह do not take a preceding anusvara.
    """
    for phones in (
        ["OW1", "N", "L", "IY0"],      # only
        ["AE1", "N", "Y", "UW0"],      # annual, first syllables
        ["K", "AH0", "N", "V", "ER1"], # convert, first syllables
        ["IH0", "N", "HH", "EH1"],     # inherit, first syllables
    ):
        assert ANUSVARA not in phonemes_to_devanagari(phones)


def test_a_nasal_before_a_vowel_is_an_ordinary_consonant():
    """`common` is कामन: the rule is about clusters, not about nasals."""
    out = phonemes_to_devanagari(["K", "AA1", "M", "AH0", "N"])

    assert ANUSVARA not in out
    assert VIRAMA not in out


def test_a_word_initial_nasal_has_no_syllable_to_carry_the_anusvara():
    """
    An anusvara rides the syllable already written. With nothing written yet
    there is nothing to ride, so the nasal keeps its letter rather than
    producing a stray mark at the start of the word.
    """
    out = phonemes_to_devanagari(["N", "T", "AA1"])

    assert not out.startswith(ANUSVARA)
    assert out.startswith("न")


def test_the_velar_nasal_is_untouched_by_the_new_rule():
    """NG has its own branch and the regression would be silent."""
    assert phonemes_to_devanagari(["M", "IY1", "N", "IH0", "NG"]) == "मीनिंग"
    assert phonemes_to_devanagari(["TH", "IH1", "NG", "K"]) == "थिंक"
