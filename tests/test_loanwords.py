"""
The loanword fold makes two written forms of one word collide.

Hindi ASR writes English loanwords in Latin and Hindi script writes them in
Devanagari, and Whisper does not pick the same one twice: hi_dub's own
reference transcript came back as `... कि ये project असल में ...`, a Latin
island inside a Devanagari transcript. That is the script boundary §16e measures as the cause of the leading prefix, and the fold is what removes
it before the model sees the text.

Rescued from `tests/test_hindi_probe.py` when the research-probe tests were
removed. These three are here because they guard `src/text/loanwords.py`,
which the shipping worker and the export guard both call — the probe was only
where they happened to be written.
"""

import json
import unicodedata
from pathlib import Path

from src.text.loanwords import fold_loanwords, latin_words, table


def test_the_table_only_holds_words_the_fixtures_contain():
    """
    The guard that stops a collision table from becoming a transliterator.

    A transliterator decides how a word is written from how it sounds, which
    is this project's open question (§4f). This table only makes two written
    forms of a word the speaker actually said collide, and the difference is
    the whole reason it is allowed to run unreviewed.
    """
    corpus = unicodedata.normalize("NFC", "\n".join([
        json.loads(Path("fixtures/scripted_text.json").read_text(encoding="utf-8"))["hi"],
        json.loads(Path("fixtures/reference_text.json").read_text(
            encoding="utf-8"))["hindi"]["text"],
    ]))

    for latin, devanagari in table("hi").items():
        assert devanagari in corpus, f"{latin} -> {devanagari} is not in the fixtures"


def test_an_unknown_latin_word_stays_latin_and_stays_visible():
    """
    Anything the table does not recognise must remain detectable as the wrong
    script rather than being quietly absorbed — the worker reports it instead
    of guessing a Devanagari spelling.
    """
    folded = fold_loanwords("कोई lecture कोई bananagram", "hi")

    assert "लेक्चर" in folded
    assert latin_words(folded) == ["bananagram"]


def test_the_english_route_is_untouched_by_the_hindi_table():
    assert fold_loanwords("31 fit perfectly", "en") == "31 fit perfectly"


def test_the_table_is_stored_nfc_so_a_nukta_letter_has_one_spelling():
    """
    फ़ is U+095E precomposed, or फ + U+093C decomposed, and the two are
    different byte sequences and therefore different token sequences.

    U+0958..U+095F are Unicode composition exclusions, so NFC does not build
    U+095E out of its parts — it takes the precomposed form apart instead.
    `table()` normalises on load (src/text/loanwords.py:50) so every entry
    lands on the decomposed form whatever the JSON file holds. Without that a
    fold would emit one spelling while the transcript held the other, on
    फ़िट, ज़्यादा and साफ़ — most of the words this corpus argues about.
    """
    precomposed = "फ़िट"          # फ़ + ि + ट
    decomposed = "फ़िट"     # फ + ़ + ि + ट

    assert precomposed != decomposed
    assert unicodedata.normalize("NFC", precomposed) == decomposed

    for devanagari in table("hi").values():
        assert unicodedata.normalize("NFC", devanagari) == devanagari
