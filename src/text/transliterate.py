"""
English text, written in Devanagari. Same words, different letters.

This exists for exactly one field: the **reference transcript**. IndicF5
clones by conditioning on a reference clip and a transcript of that clip as one
sequence, and on an `en -> hi` job the clip is the speaker's own English. The
transcript therefore has to say English words — but in the script the model is
about to generate, because a script boundary between the reference text and the
generated text puts invented speech at the head of a clip.

So the operation is neither of the two the pipeline already does:

    translation      "project"  ->  परियोजना     different word, wrong here
    transcription    "project"  ->  "project"     right word, wrong script
    transliteration  "project"  ->  प्राजेक्ट      right word, right script

**Why phonemes and not spelling.** English orthography is not phonetic —
`though`, `through` and `tough` share four letters and no ending. Mapping
letters to Devanagari produces nonsense on exactly the common words. CMUdict
gives the pronunciation instead, and Devanagari is close to phonemic, so the
mapping is between two things that are both about sound.

**What this is not.** The pronunciations are American, because CMUdict is. An
Indian speaker says `project` closer to प्रोजेक्ट than the प्राजेक्ट this
produces, and the difference is a vowel quality this module does not try to
model. That is deliberate: deciding how an Indian speaker writes an English
word is a drafting judgement, and the repo reserves those for a human.

What this module is for is removing a blocker. A phonetic approximation with
the right consonant skeleton and the right syllable count aligns to the audio,
which is what the reference transcript is for. When the spelling matters, pass
a reviewed transcript with `--reference-text` — that always wins over this.
"""

import re


class TransliterationUnavailable(RuntimeError):
    """g2p_en is not installed, so there is nothing to fall back to."""


# English /t/ and /d/ are alveolar and map to Hindi's retroflex series by
# convention — ट and ड, not त and द. That convention is why cloned English
# reads as accented, and it is still the right choice here: the reference
# transcript has to match how the *model* reads Devanagari, not how a
# phonetician would prefer to write it.
CONSONANTS = {
    "B": "ब", "CH": "च", "D": "ड", "DH": "द", "F": "फ़", "G": "ग",
    "HH": "ह", "JH": "ज", "K": "क", "L": "ल", "M": "म", "N": "न",
    "P": "प", "R": "र", "S": "स", "SH": "श", "T": "ट", "TH": "थ",
    "V": "व", "W": "व", "Y": "य", "Z": "ज़", "ZH": "ज़",
}

# Each vowel twice: the independent letter for a syllable that starts with it,
# and the matra that hangs off a preceding consonant. AH is the schwa and its
# matra is empty, because a bare Devanagari consonant already carries it —
# that is what makes `but` बट rather than बअट.
VOWELS = {
    "AA": ("आ", "ा"), "AE": ("ऐ", "ै"), "AH": ("अ", ""),
    "AO": ("ऑ", "ॉ"), "AW": ("आउ", "ाउ"), "AY": ("आइ", "ाइ"),
    "EH": ("ए", "े"), "ER": ("अर", "र"), "EY": ("ए", "े"),
    "IH": ("इ", "ि"), "IY": ("ई", "ी"), "OW": ("ओ", "ो"),
    "OY": ("ऑइ", "ॉइ"), "UH": ("उ", "ु"), "UW": ("ऊ", "ू"),
}

VIRAMA = "्"
ANUSVARA = "ं"

# Nasals that can be written as an anusvara on the syllable before them
# rather than spelled out with their own letter.
NASALS = frozenset({"N", "M"})

# What has to follow for that to happen. Hindi takes the anusvara before a
# stop, an affricate or a fricative — नंबर, कंटेंट, आंसर — but keeps the full
# nasal letter before another nasal (`unknown` is अननोन), before the
# semivowels य र ल व (`only` is ओनली), and before ह. NG is excluded here
# because it has its own branch below.
ANUSVARA_BEFORE = frozenset(CONSONANTS) - NASALS - {"NG", "Y", "W", "V", "R", "L", "HH"}

_STRESS = re.compile(r"\d")
_WORD = re.compile(r"[A-Za-z']+")


def _bare(phone: str) -> str:
    """`AA1` -> `AA`. Stress is not written in Devanagari."""
    return _STRESS.sub("", phone)


def phonemes_to_devanagari(phones: list[str]) -> str:
    """
    One word's ARPAbet phonemes, assembled into Devanagari.

    Devanagari is an abugida, so this is not a per-symbol substitution: a
    consonant carries an inherent vowel unless something says otherwise, and
    what follows it decides which of three things is written.
    """
    # Filtered up front rather than skipped inside the loop, because the loop
    # decides what to write from the phoneme that FOLLOWS. An unrecognised
    # token left in the list reads as "not a vowel", so the consonant before
    # it takes a virama and the vowel after it starts a new syllable —
    # `B ?? AH T` came out as ब्अट instead of बट.
    known = set(VOWELS) | set(CONSONANTS) | {"NG"}
    phones = [p for p in (_bare(x) for x in phones) if p in known]

    out: list[str] = []
    index = 0

    while index < len(phones):
        phone = phones[index]
        following = phones[index + 1] if index + 1 < len(phones) else None

        if phone in VOWELS:
            # A vowel here starts its own syllable — either the word does, or
            # the last phone was also a vowel. Independent form.
            out.append(VOWELS[phone][0])
            index += 1
            continue

        if phone == "NG":
            # `meaning` is मीनिंग and `think` is थिंक: the velar nasal is an
            # anusvara, and it only keeps its own ग when no other velar
            # follows to carry the place of articulation.
            out.append(ANUSVARA if following in {"G", "K"} else ANUSVARA + "ग")
            index += 1
            continue

        if (
            phone in NASALS
            and following in ANUSVARA_BEFORE
            and out
            and out[-1] != VIRAMA
        ):
            # `and` is एंड and `number` is नंबर, not अन्ड and नम्बर. The
            # anusvara rides the syllable already written, so it needs one to
            # ride: a word-initial nasal, or one right after a virama, falls
            # through and keeps its own letter.
            out.append(ANUSVARA)
            index += 1
            continue

        out.append(CONSONANTS[phone])

        if following in VOWELS:
            out.append(VOWELS[following][1])
            index += 2
        else:
            # Another consonant, or the end of the word. A virama kills the
            # inherent vowel so the cluster reads as a cluster.
            out.append(VIRAMA)
            index += 1

    word = "".join(out)

    # Hindi writes a word-final consonant bare. प्रोजेक्ट, not प्रोजेक्ट्.
    return word[:-1] if word.endswith(VIRAMA) else word


def _g2p():
    """
    Load CMUdict once, lazily.

    Lazy because the whole pipeline imports this module's package, and g2p_en
    is an optional dependency that costs an NLTK download the first time it
    runs. Nothing should pay for it unless a cross-script job actually needs
    a transliteration.
    """
    if not hasattr(_g2p, "_engine"):
        try:
            from g2p_en import G2p
        except ImportError as exc:
            raise TransliterationUnavailable(
                "g2p_en is not installed, so an English reference transcript "
                "cannot be transliterated automatically. Either install it "
                "(pip install g2p_en) or pass a reviewed transcript with "
                "--reference-text."
            ) from exc

        _g2p._engine = G2p()

    return _g2p._engine


def transliterate_to_devanagari(text: str | None) -> str:
    """
    `so let me tell you` -> `सो लेट मी टेल यू`.

    Non-alphabetic runs are passed through untouched, so spacing survives and
    anything already in Devanagari is left exactly as it is — which matters,
    because a transcript can be code-mixed before it gets here.
    """
    if not text or not text.strip():
        return ""

    engine = _g2p()
    out = []
    cursor = 0

    for match in _WORD.finditer(text):
        out.append(text[cursor:match.start()])

        word = match.group()
        phones = [p for p in engine(word) if p.strip()]
        out.append(phonemes_to_devanagari(phones) or word)

        cursor = match.end()

    out.append(text[cursor:])

    return " ".join("".join(out).split())
