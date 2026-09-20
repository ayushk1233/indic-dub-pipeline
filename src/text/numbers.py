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
rather than a rule. Above a hundred they are regular, so that part is scales:
Hindi groups in the Indian system and English in the Western one.

Three readings here are about context rather than value, so they live in
`spell_numbers` and leave `number_words` a pure function of its argument:

  - A bare four-digit run in 1100-1999 is a year. `1947` is उन्नीस सौ
    सैंतालीस, not एक हज़ार नौ सौ सैंतालीस. From 2000 the plain cardinal is
    already the year form, so the rule stops there. A grouped `1,947` was
    written as a quantity and is read as one.
  - A run of IDENTIFIER_DIGITS or more is an identifier, not a quantity.
    Nobody reads a phone number as नौ सौ सतासी करोड़.
  - A leading zero is never a quantity either: `007` is not seven.

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

# The separator in a decimal, spoken. Digits after it are read one at a time,
# which is what both languages do: 3.14 is तीन दशमलव एक चार.
POINT = {"en": " point ", "hi": " दशमलव "}

_HUNDRED = {"en": "hundred", "hi": "सौ"}

# At this many digits a run stops being a quantity and becomes an identifier —
# a phone number, an order id, a pincode. Six is still a quantity worth
# reading as one (400052 is चार लाख बावन); ten is a phone number and reading
# it as a cardinal would be wrong in a way no listener would forgive.
IDENTIFIER_DIGITS = 7

# Length alone is not enough, because a round number is a quantity however
# long it is: `10000000` is एक करोड़, not eight digits read out. Trailing
# zeros are what separate the two — a crore has seven and a phone number has
# at most one or two by chance.
QUANTITY_TRAILING_ZEROS = 3

# Scales, largest first.
#
# Hindi groups in the Indian system, and that is not a stylistic preference:
# an Indian speaker reading 250000 says दो लाख पचास हज़ार. There is no
# ceiling, because the largest scale recurses on its own multiplier.
_HI_SCALE = (
    (10_000_000, "करोड़"),
    (100_000, "लाख"),
    (1_000, "हज़ार"),
    (100, "सौ"),
)

# English groups in the Western system. Note what this path is for: IndicF5
# cannot generate English (§4), so no English number here is ever
# spoken. It exists so that a script's "four thousand" and Whisper's "4000"
# collide when a CER is scored against them.
_EN_SCALE = (
    (10**12, "trillion"),
    (10**9, "billion"),
    (10**6, "million"),
    (1_000, "thousand"),
    (100, "hundred"),
)

# A number as it appears in written text: Indian (1,50,000) or Western
# (1,500,000) digit grouping, with an optional decimal part. The grouped
# alternative comes first so it wins over the bare one.
#
# `,\d{2,3}` and not `,\d+` so that "item 1,2" stays two numbers rather than
# being glued into twelve.
_NUMBER = re.compile(r"\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?")

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


def _hi_below_hundred(value):
    return _HI[value]


def _en_below_hundred(value):
    if value < 20:
        return _ONES[value]

    tens = _TENS[value // 10]

    return f"{tens} {_ONES[value % 10]}" if value % 10 else tens


def _grouped(value, scale, below_hundred):
    """
    Walk `scale` largest-first, recursing on both the multiplier and the
    remainder.

    Recursing on the multiplier is what removes the ceiling: 12 करोड़ and
    2_50_00_00_000 both resolve without the table knowing anything above
    करोड़.
    """
    if value < 100:
        return below_hundred(value)

    for base, name in scale:
        if value < base:
            continue

        count, rest = divmod(value, base)
        head = f"{_grouped(count, scale, below_hundred)} {name}"

        if rest:
            return f"{head} {_grouped(rest, scale, below_hundred)}"

        return head

    # Unreachable: the smallest scale in both tables is 100 and value >= 100.
    return str(value)


def number_words(value, language="en"):
    """
    47 -> 'forty seven' / 'सैंतालीस'. 250000 -> 'दो लाख पचास हज़ार'.

    A pure function of `value`. Everything that depends on how the number was
    *written* — years, identifiers, leading zeros — is in `spell_numbers`,
    because the same integer is read differently depending on what it is.

    Note 100 is एक सौ rather than the bare सौ the table holds at that index,
    so that it composes: the remainder of 1100 has to be एक सौ for एक हज़ार एक
    सौ to come out right.
    """
    if language == "hi":
        return _grouped(value, _HI_SCALE, _hi_below_hundred)

    return _grouped(value, _EN_SCALE, _en_below_hundred)


def _digit_by_digit(digits, language):
    return " ".join(number_words(int(digit), language) for digit in digits)


def _year_words(value, language):
    """
    1947 -> 'उन्नीस सौ सैंतालीस' / 'nineteen forty seven'.

    Hindi keeps सौ and English drops "hundred", because that is what each
    language's speakers say and therefore what each language's scripts write.
    A round year keeps it in both: 1900 is nineteen hundred.
    """
    hundreds, rest = divmod(value, 100)
    high = number_words(hundreds, language)

    if not rest:
        return f"{high} {_HUNDRED[language]}"

    low = number_words(rest, language)

    if language == "hi":
        return f"{high} {_HUNDRED['hi']} {low}"

    return f"{high} {low}"


def _is_identifier(token, digits):
    """
    Whether a long run is an id to read out rather than a quantity to say.

    Digit grouping settles it on its own — somebody who wrote `1,00,00,000`
    meant a number — and past that it is trailing zeros, because that is the
    thing a crore has and a phone number does not.
    """
    if len(digits) < IDENTIFIER_DIGITS or "," in token:
        return False

    return len(digits) - len(digits.rstrip("0०")) < QUANTITY_TRAILING_ZEROS


def _spoken_integer(token, language):
    digits = token.replace(",", "")

    # A leading zero is never a quantity, at any length.
    if len(digits) > 1 and digits[0] in "0०":
        return _digit_by_digit(digits, language)

    if _is_identifier(token, digits):
        return _digit_by_digit(digits, language)

    value = int(digits)

    # Only a bare run is a candidate year. `1,947` carries its own evidence
    # that somebody meant a quantity.
    if len(digits) == 4 and 1100 <= value <= 1999 and "," not in token:
        return _year_words(value, language)

    return number_words(value, language)


def _spoken(token, language):
    whole, point, fraction = token.partition(".")

    if not point:
        return _spoken_integer(token, language)

    return (
        f"{_spoken_integer(whole, language)}"
        f"{POINT[language]}"
        f"{_digit_by_digit(fraction, language)}"
    )


def spell_numbers(text, language=None):
    r"""
    Numbers and `%` replaced by the words a transcript uses for them.

    `language` defaults to whatever script the text is in. Devanagari digits
    are matched too — `\d` is Unicode-aware and `int()` reads ०-९ — because
    Whisper writes a Hindi number either way depending on the clip.

    Spacing is left loose — a caller that cares collapses whitespace or strips
    non-alphanumerics afterwards, and both callers here do.
    """
    text = text or ""
    language = language or script_language(text)
    text = text.replace("%", PERCENT.get(language, PERCENT["en"]))

    return _NUMBER.sub(lambda m: f" {_spoken(m.group(), language)} ", text)
