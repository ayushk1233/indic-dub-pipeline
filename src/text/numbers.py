"""
Two written forms of the same number, made to agree.

An ASR and a script disagree about numbers systematically rather than
occasionally. Whisper writes `31`, `47`, `20%`, `6`; the scripted text writes
"Thirty-one", "forty-seven", "twenty percent", "six". On a character
comparison those share almost nothing, so the difference is scored as the
model having said the wrong thing.

It has now cost two separate measurements in this project:

  - The slicer matched "Thirty-one fit perfectly." to `fit perfectly,` alone
    and cut a 0.92 s clip — 27 characters per second, faster than anything
    this project has measured from a human or a model.
  - The transliteration probe scored `31 feet perfectly.` at CER 0.458 against
    "Thirty-one fit perfectly.", where the only real error is `feet` for
    `fit`. Most of that 0.458 is the numeral.

Hindi has the same disagreement and no shared root to fall back on. The
scripted Hindi says "सैंतालीस सेगमेंट" and "इकतीस बिल्कुल फ़िट
हुए"; Whisper writes `47` and `31`. English and Hindi number words share not
one character, so the table has to be exact rather than approximate — a near
miss scores as a substitution instead of a match, which is the failure it
exists to prevent. `tests/test_number_normalization.py` pins every number word
that actually appears in the fixtures against this table, so a wrong entry
fails locally rather than as an inflated CER on a GPU.

Hindi cardinals under a hundred are irregular one by one, so they are a table
rather than a rule. Past a hundred the digits are left alone, in both
languages.

This is alignment machinery, not the pipeline's number normaliser. That one
has to decide how an Indian speaker *says* a number, which is a question about
speech. This one only has to make two spellings collide, so "one hundred and
one" versus "one hundred one" does not matter and no attempt is made to be
idiomatic.
"""

import re

# Hindi cardinals are irregular below a hundred, so this is a table and not a
# rule. Only the forms that appear in fixtures/metadata.json have been checked
# against real text; the rest are the standard spellings and are pinned by a
# test that walks the whole table rather than by a measurement.
_HI = (
    "शून्य एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह "
    "सोलह सत्रह अठारह उन्नीस बीस इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस "
    "सत्ताईस अट्ठाईस उनतीस तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस "
    "सैंतीस अड़तीस उनतालीस चालीस इकतालीस बयालीस तैंतालीस चवालीस पैंतालीस "
    "छियालीस सैंतालीस अड़तालीस उनचास पचास इक्यावन बावन तिरपन चौवन पचपन "
    "छप्पन सत्तावन अट्ठावन उनसठ साठ इकसठ बासठ तिरसठ चौंसठ पैंसठ छियासठ "
    "सड़सठ अड़सठ उनहत्तर सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर "
    "छिहत्तर सतहत्तर अठहत्तर उन्यासी अस्सी इक्यासी बयासी तिरासी चौरासी "
    "पचासी छियासी सत्तासी अट्ठासी नवासी नब्बे इक्यानवे बानवे तिरानवे "
    "चौरानवे पचानवे छियानवे सत्तानवे अट्ठानवे निन्यानवे सौ"
).split()

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve "
         "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety")

PERCENT = {"en": " percent ", "hi": " प्रतिशत "}

_DIGITS = re.compile(r"\d+")
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")


def script_language(text):
    """
    Which table to use when the caller did not say.

    `score_text` compares two strings that must normalise the same way, and it
    is handed no language. Reading it off the *intended* text and applying it
    to both is the only reading that cannot disagree with itself: inferring
    per string would spell the reference in English and a Devanagari
    transcript of it in Hindi, and score every number as a substitution.
    """
    return "hi" if _DEVANAGARI.search(text or "") else "en"


def number_words(value, language="en"):
    """47 -> 'forty seven' / 'सैंतालीस'. Past the table the digits stay."""
    if language == "hi":
        return _HI[value] if value < len(_HI) else str(value)
    if value < 20:
        return _ONES[value]
    if value < 100:
        tens = _TENS[value // 10]
        return f"{tens} {_ONES[value % 10]}" if value % 10 else tens
    if value < 1000:
        rest = value % 100
        hundreds = f"{_ONES[value // 100]} hundred"
        return f"{hundreds} {number_words(rest)}" if rest else hundreds
    return str(value)


def spell_numbers(text, language=None):
    """
    Digit runs and `%` replaced by the words a transcript uses for them.

    `language` defaults to whatever script the text is in. Devanagari digits
    are matched too — `\d` is Unicode-aware and `int()` reads ०-९ — because
    Whisper writes a Hindi number either way depending on the clip.

    Spacing is left loose — a caller that cares collapses whitespace or strips
    non-alphanumerics afterwards, and both callers here do.
    """
    text = text or ""
    language = language or script_language(text)
    text = text.replace("%", PERCENT.get(language, PERCENT["en"]))
    return _DIGITS.sub(
        lambda m: f" {number_words(int(m.group()), language)} ", text)
