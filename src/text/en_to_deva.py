"""
Accent dials on English already spelled in Devanagari.

The speaker placed his own reference recording at about 6 on an informal
1-to-10 Indian-accent scale and IndicF5's clone of it at about 9 — "the hard t,
d especially" (FINDINGS §4b). FINDINGS §4d ruled out the first of the two
candidate causes: transliterating the reference transcript fixed intelligibility
outright, reaching the Whisper floor, and moved the accent not at all. That
leaves the orthography, which is what this module turns into something that can
be varied.

These are **transforms over text that has already been hand-transliterated and
reviewed**, not a transliterator. That is deliberate:

  - the drafting judgement stays with the speaker, in fixtures/xlit/*.json
  - each dial is a diff against reviewed text, scannable in seconds, rather
    than seven fresh sentences needing a fresh review
  - the dial is code with tests, so an arm is reproducible from the fixture
    plus a function name instead of from another frozen file

A blanket character substitution is only safe because of what this corpus is.
Every ट and ड in fixtures/xlit/ was written to spell an English /t/ or /d/ —
the text is transliterated English throughout, with no Hindi words in it whose
retroflexes mean something. Do not reuse `dental` on real Hindi.

**Predicted cost, written down before the run rather than after it.** deva_hand
deliberately spells the eight loanwords the way Hindi text spells them —
प्रोजेक्ट, वीडियो, सिस्टम — because that is the distribution IndicF5's training
text contains. `dental` destroys exactly that: सिस्तम, वीदियो and प्रोजेक्त are
not Hindi words and the model has never seen them. So these arms are expected
to trade content for accent, and the content gate and the `floor (him)` row are
what will say how much. An arm that moves the accent to 6 and takes CER from
0.048 to 0.3 has not solved the problem; §14 exists because this project has
written down a favourable half of a trade before.

What is deliberately not here: rhoticity. Post-vocalic र in लॉन्गर and वर्ड्स
does force a rhotic reading, and it is named in §4b as a third dial, but there
is no mechanical rule for it that does not make some words worse — English's
reduced /ə/ has no clean Devanagari vowel, so `longer` becomes either लॉन्गा
("longaa", too long) or लॉन्ग ("long", a different word). It needs hand
drafting, and it should wait for evidence that the stops were not the whole
complaint.
"""

import re

# English /t/ and /d/ are ALVEOLAR. Devanagari offers only dental and
# retroflex, so a transliterator has to pick a side and the convention picks
# retroflex — ट for "tell", ड for "does". That convention is what makes the
# stops sound hard: retroflex is further back than English ever goes, while
# dental is further forward. Neither is correct; dental is the nearer miss,
# and it is the one Indian speakers closer to 6 on the scale use.
DENTAL = {
    "ट": "त", "ठ": "थ", "ड": "द", "ढ": "ध",
    # ण is retroflex /ɳ/; English /n/ is alveolar and न is the nearer miss.
    # It does not occur in the current fixtures, and is here so the map is
    # complete rather than because it fires.
    "ण": "न",
}

# Voiceless stops that Devanagari can aspirate UNAMBIGUOUSLY. प is missing on
# purpose and the omission is the finding: फ is read as /f/ in modern Hindi,
# not as /pʰ/, so "project" aspirated through फ would come back as "froject".
# There is no way to write an aspirated /p/ in this script that IndicF5 will
# read as one, so word-initial /p/ keeps its unaspirated spelling and that arm
# is under-applied rather than wrong.
ASPIRATE = {"त": "थ", "ट": "ठ", "क": "ख", "च": "छ"}

# English aspirates a word-initial voiceless stop only in a STRESSED onset.
# Unstressed monosyllabic function words keep a plain stop: "to" in "take to
# speak" is [tə], never [tʰuː]. Position alone cannot see stress, so the
# exceptions are listed. Both spellings are here so the dials compose in
# either order.
UNSTRESSED = {"टू", "तू"}

VIRAMA = "्"

_WORD = re.compile(r"[^\s—,।]+")


def dental(text):
    """
    Retroflex to dental, everywhere.

    The single dial the complaint actually named. 41 of the characters in the
    current fixtures are ट and 18 are ड, so this is not a subtle edit — it is
    most of the consonant inventory of the generated text.
    """
    return "".join(DENTAL.get(character, character) for character in text)


def aspirate_initial(text):
    """
    Aspirate a word-initial voiceless stop, where English aspirates one.

    English aspirates /p t k/ at the start of a word; Hindi's plain stops are
    unaspirated there, so a Devanagari spelling of "tell" or "keep" asks for a
    stop English speakers do not make. This restores it where the script can
    express it.

    Two positions are skipped, both because English does not aspirate there
    either:

      - cluster-initial (`क्लियर`, `ट्वेंटी`) — the stop is followed by a
        virama. English reduces aspiration in /kl/ and /tw/ onsets to devoicing
        of the following sonorant, which Devanagari cannot write at all, so
        the unaspirated spelling is the closer of the two available.
      - anywhere but word-initially, which covers `स्पीच` and `सिस्टम`.
      - the unstressed function words in `UNSTRESSED`, because English
        aspirates a stressed onset and not a reduced one.

    Apply after `dental` if both are wanted: `त` is in the map so that a
    dentalised word-initial /t/ still gets its aspiration.
    """
    def convert(match):
        word = match.group()
        head = word[0]
        if head not in ASPIRATE or word in UNSTRESSED:
            return word
        if len(word) > 1 and word[1] == VIRAMA:
            return word
        return ASPIRATE[head] + word[1:]

    return _WORD.sub(convert, text)


def soft(text):
    """Both stop dials: dental, then word-initial aspiration."""
    return aspirate_initial(dental(text))


# Named so an arm is a fixture plus a key, and the probe never holds a lambda.
DIALS = {
    "deva_dental": dental,
    "deva_soft": soft,
}


def apply_dial(name, texts):
    """One dial over an arm's {id: devanagari} mapping."""
    transform = DIALS[name]
    return {index: transform(text) for index, text in texts.items()}
