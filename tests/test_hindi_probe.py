"""
The hi -> hi probe must measure Hindi, against Hindi, with a Hindi ruler.

It was written by reading the transliteration probe, which conditions on
English audio, slices the English take and scores against an English floor.
Every one of those is a line that still runs if it is left pointing at the
wrong language: the model synthesizes, the report fills in, and the numbers
describe a cross-lingual clone nobody asked for. These tests pin the four
places that would not raise.

The other half is the number table. Hindi and English number words share no
characters at all, so `47` against `सैंतालीस` scores as a total substitution
rather than a match — the same failure that once put `31 feet perfectly.` at
CER 0.458 in English, with nothing in Hindi to soften it.

No audio, no model, no GPU.
"""

import json
import unicodedata
from pathlib import Path

import pytest

from colab.indicf5_check import normalize, score_text
from src.eval.translation_metrics import script_ratio
from src.text.numbers import _HI, number_words, script_language, spell_numbers

FIXTURE_HI = Path("fixtures/sentences/fixture7_hi.json")
FIXTURE_EN = Path("fixtures/sentences/fixture7.json")
SLOTS_HI = Path("fixtures/hi_speaker/slots.json")
METADATA = Path("fixtures/metadata.json")


@pytest.fixture
def hindi():
    return json.loads(FIXTURE_HI.read_text(encoding="utf-8"))


@pytest.fixture
def english():
    return json.loads(FIXTURE_EN.read_text(encoding="utf-8"))


# ------------------------------------------------- the comparability contract


def test_the_two_fixtures_are_the_same_sentences(hindi, english):
    """
    Selected by source index, never by re-running the character filter.

    fixture7 was cut with a 25-character minimum. 'इकतीस बिल्कुल फ़िट हुए।' is
    23 characters where 'Thirty-one fit perfectly.' is 25, so re-applying that
    filter to Hindi keeps the English sentence and drops its own translation —
    and then row 5 of one report and row 5 of the other are different
    sentences. Comparing hi -> hi against en -> en sentence for sentence is
    the entire reason the Hindi fixture exists.
    """
    assert ([s["source_index"] for s in hindi["sentences"]]
            == [s["source_index"] for s in english["sentences"]])
    assert len(hindi["sentences"]) == len(english["sentences"]) == 7


def test_every_hindi_row_carries_the_english_it_pairs_with(hindi, english):
    """So the pairing is readable in the file rather than inferred from an index."""
    by_index = {s["source_index"]: s["text"] for s in english["sentences"]}
    for row in hindi["sentences"]:
        assert row["english"] == by_index[row["source_index"]]


def test_the_short_sentence_survived_the_selection(hindi):
    """The one the character filter would have dropped. If it is gone, the
    fixture was rebuilt by filter and the two reports no longer line up."""
    short = [s for s in hindi["sentences"] if s["chars"] < 25]
    assert short, "the 23-character sentence is missing"
    assert short[0]["source_index"] == 8


def test_the_text_is_devanagari_and_nfc(hindi):
    for row in hindi["sentences"]:
        assert script_ratio(row["text"], "hi") == 1.0
        assert unicodedata.normalize("NFC", row["text"]) == row["text"]


def test_bytes_are_recorded_and_are_not_characters(hindi):
    """132 Latin characters and 132 Devanagari characters are indistinguishable
    in a report; their byte counts are not — FINDINGS §13."""
    for row in hindi["sentences"]:
        assert row["bytes"] == len(row["text"].encode("utf-8"))
        assert row["bytes"] > row["chars"]


# ------------------------------------------------------- the Hindi ruler


def test_the_probe_points_at_the_hindi_take_everywhere():
    """
    The four lines that would still run if left pointing at English.

    A probe conditioned on english_reference_short.wav while generating Hindi
    is a cross-lingual clone, and it produces a complete, plausible, wrong
    report.
    """
    probe = pytest.importorskip("colab.indicf5_hindi_probe")

    assert probe.REFERENCE == "hindi_reference_short.wav"
    assert probe.REFERENCE_KEY == "hindi_short"
    assert probe.SENTENCES.name == "fixture7_hi.json"
    assert probe.SLOTS.parts[-2] == "hi_speaker"


def test_the_floor_is_his_hindi_and_is_not_an_arm():
    """
    The floor is a ruler. If it were counted as an arm it would flatter every
    average it appears in, and if it were missing the report would read a
    synthesis CER against zero.
    """
    probe = pytest.importorskip("colab.indicf5_hindi_probe")

    rows = probe.floor_rows()
    assert len(rows) == 7
    assert {r["arm"] for r in rows} == {"floor"}
    assert probe.ARM != "floor"
    assert all(r["language"] == "hi" for r in rows)


def test_the_slots_are_the_hindi_recording():
    data = json.loads(SLOTS_HI.read_text(encoding="utf-8"))
    assert data["language"] == "hi"
    assert data["source"].endswith("hindi_speech.wav")
    assert data["natural_cps"] == 10.81
    assert len(data["slots"]) == 7


def test_the_slots_lie_inside_the_recording():
    data = json.loads(SLOTS_HI.read_text(encoding="utf-8"))
    for slot in data["slots"]:
        assert 0.0 <= slot["start_s"] < slot["end_s"] <= data["source_duration_s"]
        assert slot["duration_s"] > 0.5
    assert data["covered_s"] <= data["source_duration_s"]


def test_the_reference_window_is_marked():
    """Sentences the model is handed the audio and transcript for. An arm that
    works only on these has shown nothing, so the report has to be able to
    separate them."""
    data = json.loads(SLOTS_HI.read_text(encoding="utf-8"))
    inside = [s["id"] for s in data["slots"] if s["in_reference"]]
    assert inside, "no sentence marked in_reference — leakage cannot be read"
    assert data["reference_window_s"] is not None


# ---------------------------------------------------------- Hindi numbers


def test_every_number_word_in_the_script_is_in_the_table():
    """
    The collision guard. A wrong or missing entry does not raise — it scores
    the number as a substitution and inflates the CER of whichever sentence
    contains it, on a GPU, hours later.
    """
    text = json.loads(METADATA.read_text(encoding="utf-8"))["hindi"]["scripted_text"]
    for mark in "—।,?":
        text = text.replace(mark, " ")
    found = sorted({t for t in text.split() if t in _HI})
    assert found == ["इकतीस", "एक", "छह", "तीन", "नौ", "बीस", "सात", "सैंतालीस"]


def test_the_table_covers_zero_to_a_hundred_without_repeating_itself():
    assert len(_HI) == 101
    assert len(set(_HI)) == 101


def test_the_two_spellings_of_a_hindi_number_collide():
    """What Whisper writes against what the script says."""
    assert normalize("इकतीस बिल्कुल फ़िट हुए।") == normalize("31 बिल्कुल फ़िट हुए")
    assert normalize("सैंतालीस सेगमेंट") == normalize("47 सेगमेंट")
    assert normalize("लगभग बीस प्रतिशत") == normalize("लगभग 20%")


def test_a_real_hindi_error_still_scores_as_one():
    """The normaliser removes a notation difference, not a mistake."""
    scored = score_text("सात नामुमकिन थे", "साथ नामुमकिन थे")
    assert scored["cer"] > 0


def test_the_language_is_read_off_the_intended_text_not_each_string():
    """
    Inferring per string would spell the reference in English and a
    Devanagari transcript of it in Hindi, and score every number as a
    substitution — the error this normaliser exists to remove.
    """
    assert script_language("इकतीस बिल्कुल") == "hi"
    assert script_language("Thirty-one fit") == "en"
    # digits alone carry no script, so the caller's choice has to win
    assert spell_numbers("31", "hi").strip() == "इकतीस"
    assert spell_numbers("31", "en").strip() == "thirty one"


def test_english_numbers_are_untouched_by_the_hindi_table():
    """The English probe's numbers must not move — FINDINGS §4c is read
    against them."""
    assert normalize("Thirty-one fit perfectly.") == normalize("31 fit perfectly.")
    assert number_words(47) == "forty seven"


def test_a_number_past_the_table_is_left_as_digits():
    assert number_words(101, "hi") == "101"
    assert number_words(1000, "en") == "1000"


# ----------------------------------------------------------------- nukta


def test_the_two_forms_of_a_nukta_letter_are_the_same_word():
    """
    फ़ is U+095E precomposed or फ + U+093C decomposed, and Whisper does not
    always emit the form the fixture holds. Without NFC the report would score
    a spelling of the same letter as a different letter — on फ़िट, ज़्यादा and
    साफ़, which is most of the words this corpus argues about.
    """
    precomposed = "\u095e\u093f\u091f"          # फ़ + ि + ट
    decomposed = "\u092b\u093c\u093f\u091f"     # फ + ़ + ि + ट
    assert precomposed != decomposed

    # U+0958..U+095F are composition exclusions, so NFC does not build U+095E
    # out of the parts — it takes the precomposed one apart instead. Both
    # spellings land on the decomposed form and collide.
    assert unicodedata.normalize("NFC", precomposed) == decomposed

    assert normalize(precomposed) == normalize(decomposed)
    assert score_text(f"बिल्कुल {precomposed} हुए",
                      f"बिल्कुल {decomposed} हुए")["cer"] == 0
