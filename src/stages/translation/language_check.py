"""
Check that a translation is in the language that was asked for.

Two existing checks each miss this, for different reasons.

`script_ratio` in the evaluation layer verifies the writing system, which
catches Latin text where Devanagari was expected. It cannot separate two
languages that share a script.

`FidelityScorer` verifies meaning with LaBSE, which is trained to be
language-agnostic. Measured on a real case, it scored a Maithili translation
of an English sentence at 0.888 while correct Hindi translations of the same
sentence scored around 0.86 — it ranked the wrong-language output *best*.

So this module does the remaining job: given text and an intended language,
how much does the text look like that language rather than a neighbour sharing
its script. The method is closed-class function words, which is crude but
well-suited here, because the languages that collide are close enough that
content words are shared and grammatical particles are exactly what differ.

The scope is deliberately narrow. Devanagari is the only script in this
project's target set where more than one candidate language competes, so only
Devanagari languages carry marker sets; everything else defers to the script
check, which is sufficient and more reliable.

A production system should replace this with a proper language identifier such
as the NLLB LID model, which covers Maithili, Bhojpuri and Hindi as distinct
labels. This heuristic exists because it needs no extra model download and
catches the specific failure observed.
"""

import re
from dataclasses import dataclass


# Grammatical particles and auxiliaries that are frequent, closed-class, and
# distinctive between Devanagari languages.
LANGUAGE_MARKERS: dict[str, set[str]] = {
    "hi": {
        "है", "हैं", "था", "थी", "थे", "का", "की", "के", "को", "में",
        "और", "नहीं", "होता", "होती", "करते", "करता", "गया", "रहा", "हुआ",
    },
    "mai": {
        "छैक", "छथि", "अछि", "छल", "करैत", "सभक", "सभ", "केँ", "नहि",
        "भेल", "गेल", "रहल", "अपन", "एकर",
    },
    "bho": {
        "बा", "बाड़", "बानी", "करेला", "रहे", "होला", "गइल", "भइल", "ह",
    },
    "mr": {
        "आहे", "आहेत", "होता", "होते", "चा", "ची", "चे", "ला", "मध्ये",
        "आणि", "नाही", "केले", "करतो", "करते", "झाले",
    },
    "ne": {
        "छ", "छन्", "थियो", "को", "मा", "र", "छैन", "गर्छ", "भयो", "हुन्",
    },
}

# Which languages could plausibly be confused with each other, because they
# share a script. Only these get the marker comparison.
CONFUSABLE_GROUPS: dict[str, tuple[str, ...]] = {
    "hi": ("hi", "mai", "bho", "mr", "ne"),
    "mr": ("mr", "hi", "mai", "ne"),
}

# Below this share of marker matches belonging to the requested language, the
# text is more likely a neighbour language than the one asked for.
MARKER_FLOOR = 0.5

_TOKEN_RE = re.compile(r"[^\s।॥.,!?;:\"'()\[\]]+")


@dataclass
class LanguageVerdict:
    language: str
    score: float
    matched: int
    competing: int
    checked: bool

    @property
    def suspect(self) -> bool:
        """
        True when the text looks more like a neighbouring language.

        Text that produced no marker evidence at all is never suspect: a short
        phrase may legitimately contain no function words, and guessing from
        no evidence would reject good translations.
        """
        return self.checked and self.score < MARKER_FLOOR


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def language_score(text: str, language: str) -> LanguageVerdict:
    """
    How strongly `text` looks like `language` rather than a script neighbour.

    The score is the share of all marker hits that belong to the requested
    language. 1.0 means every function word found was characteristic of it;
    0.0 means none were and the hits all belonged to neighbours.
    """
    group = CONFUSABLE_GROUPS.get(language)

    if group is None:
        # No neighbour shares this script in our target set; the script check
        # already settles it and a marker verdict would only add noise.
        return LanguageVerdict(language, 1.0, 0, 0, checked=False)

    tokens = set(tokenize(text))

    matched = len(tokens & LANGUAGE_MARKERS.get(language, set()))
    competing = sum(
        len(tokens & LANGUAGE_MARKERS.get(other, set()))
        for other in group
        if other != language
    )

    total = matched + competing

    if total == 0:
        return LanguageVerdict(language, 1.0, 0, 0, checked=False)

    return LanguageVerdict(
        language=language,
        score=matched / total,
        matched=matched,
        competing=competing,
        checked=True,
    )


def is_off_language(text: str, language: str) -> bool:
    return language_score(text, language).suspect
