"""
Two spellings of one loanword, made to collide.

Hindi writes English loanwords in Devanagari — प्रोजेक्ट, वीडियो, लेक्चर,
इंटरव्यू, सिस्टम, सेगमेंट, प्रोसेस, फ़िट. Whisper transcribing Hindi writes
them in either script, and it does not decide once: in the same run it wrote
`लेक्शर` for the speaker's own voice and `lecture` for the synthesis of the
same word. The two spellings share no characters at all, so a clip that said
the right word is scored as having substituted every character of it.

Measured 2026-09-19, the hi -> hi probe's first run:

  - `इकतीस बिल्कुल फ़िट हुए।` came back as `31 बिलकुल fit हुए` and scored
    CER 0.227. The only real error is बिलकुल for बिल्कुल, worth about 0.045.
  - The script gate excluded five clips for being 60–76% Devanagari. All five
    had said the right words; what was in Latin was `fit`, `lecture` and
    `interview`. The exclusions were not random — they fell on the two
    loanword-heavy sentences — so dropping them flattered the arm.

This is the same kind of object as the number table in src/text/numbers.py,
and it exists for the same reason: an ASR and a fixture disagree about
notation systematically rather than occasionally, and a character comparison
reads that disagreement as speech.

**It is not a transliterator and must not grow into one.** A transliterator
decides how a word is written from how it sounds, which is the open question
this whole project is about. This table only makes two written forms of a word
that is *already in the fixtures* collide, and every Devanagari value is
checked at load to occur in the Hindi corpus. That check is what stops it from
quietly becoming a lexicon that rewrites words the speaker never said.

Applied to Hindi comparisons only. English text has no second spelling to
collide with.
"""

import json
import unicodedata
from functools import lru_cache
from pathlib import Path

TABLE = Path(__file__).resolve().parent.parent.parent / "fixtures" / "loanwords_hi.json"


@lru_cache(maxsize=None)
def table(language="hi"):
    """{latin: devanagari}, empty for any language without a file."""
    if language != "hi" or not TABLE.exists():
        return {}
    data = json.loads(TABLE.read_text(encoding="utf-8"))
    return {k.lower(): unicodedata.normalize("NFC", v)
            for k, v in data["latin_to_devanagari"].items()}


def fold_loanwords(text, language="hi"):
    """
    Whole Latin words replaced by the Devanagari the fixtures spell them with.

    Whole words only, matched after lowercasing, so a Latin run that is not in
    the table is left exactly as it was and still shows up in the script
    fraction. Anything this does not recognise stays visible as a problem
    rather than being quietly absorbed.
    """
    mapping = table(language)
    if not mapping:
        return text
    return " ".join(mapping.get(word.lower(), word) for word in (text or "").split())


def latin_words(text):
    """The Latin-script words in a transcript, for reporting what moved."""
    return [word for word in (text or "").split()
            if any(c.isascii() and c.isalpha() for c in word)]
