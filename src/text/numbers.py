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

This is alignment machinery, not the pipeline's number normaliser. That one
has to decide how an Indian speaker *says* a number, which is a question about
speech. This one only has to make two spellings collide, so "one hundred and
one" versus "one hundred one" does not matter and no attempt is made to be
idiomatic.
"""

import re

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve "
         "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety")

_DIGITS = re.compile(r"\d+")


def number_words(value):
    """47 -> 'forty seven'. Past 999 the digits are left alone."""
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


def spell_numbers(text):
    """
    Digit runs and `%` replaced by the words a transcript uses for them.

    Spacing is left loose — a caller that cares collapses whitespace or strips
    non-alphanumerics afterwards, and both callers here do.
    """
    text = (text or "").replace("%", " percent ")
    return _DIGITS.sub(lambda m: f" {number_words(int(m.group()))} ", text)
