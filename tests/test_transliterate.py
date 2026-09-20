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
