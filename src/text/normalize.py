"""
The text a TTS model is actually asked to say.

IndicF5 cannot say a digit. A clean GPU run — 30 of 30 segments returned,
every clip landing on its forced duration to within one hop, assembly at
tempo 1.00 — produced two dubbed videos with audible gibberish in them.
Transcribing the synthesized audio back placed it exactly:

    asked  लगभग 20% अधिक समय      heard  लगभग लतक अधिक समय
    asked  उसी 3 सेकंड में         heard  उसी इदस सेकंड में
    asked  ने 47 खंडों को          heard  ने आत्तखंडो को
    asked  31 फीट 9 पूरी तरह से    heard  एंड फिट पूरी तरह से

Nine of the ten segments carrying a digit were corrupted. Of the segments
carrying none, all but one unexplained row were clean. The same run contains
two controls nobody set up: `तीन सिकंड` and `छह अलग-अलग` reached the model
spelled as words — one from the speaker, one because IndicTrans2 happened to
translate "six" rather than emit `6` — and both came back perfect. Same model,
same reference, same session. The fault is in the text.

The damage is not confined to the number. `यू`, `ते`, `एंड` appeared before the
first real word on exactly the digit-bearing rows, because a token the model
cannot align to its conditioned frames desynchronises the unanchored
`ref_audio_len` slice and the remainder is spoken at the head (§5).
One mechanism, two symptoms, one fix.

**Why this is not in the worker.** The loanword fold lives GPU-side in
`IndicF5Worker.prepare_text()` because it is model policy, and number
expansion looks like the same kind of thing. It runs locally anyway, before
length control scores its candidates, because `score_candidates` predicts
duration from the text: `20%` is three characters and `बीस प्रतिशत` is eleven.
Expanding after selection would mean every digit-bearing segment had been
chosen against an estimate for text the model never speaks, and `fix_duration`
would then force that wrong span. Running it first fixes the pronunciation and
the timing together, and leaves a bundle whose request says what will be said.

**Register.** Numbers follow the register of the translation they sit in, so
this function does not take a register argument and must not grow one. The
configured `indictrans2-en-indic-1B` produces literary Hindi, so `20%` becomes
बीस प्रतिशत. A Hinglish output register (ट्वेंटी परसेंट) is a planned,
user-selectable feature that will select a different *translation* path rather
than a different normaliser.

The cardinal table itself is `src/text/numbers.py`, which is alignment
machinery for scoring — its job is to make two spellings of the same number
collide. This module reuses that table and not its contract.
"""

import re

from src.text.numbers import spell_numbers


# A digit run that survived spelling out. `numbers.py` now has no ceiling, so
# in a normalised string this matches nothing — which is the point. It is a
# backstop against text that reached the export without being normalised at
# all, and three separate code paths produce a translation.
_DIGIT_RUN = re.compile(r"\d+")

# A dash standing on its own, which is how IndicTrans2 punctuates a clause
# break — `सरल सही -`, `छह अलग - अलग`. A hyphen inside a word is left alone:
# fixtures/xlit/deva_hand.json mirrors the English hyphen in number compounds
# (फोर्टी-सेवन) and the speaker reviewed that spelling.
_LONE_DASH = re.compile(r"(?:(?<=\s)|(?<=^))[-–—]+(?=\s|$)")


def normalize_for_speech(text: str | None, language: str = "hi") -> str:
    """
    Rewrite `text` into the form the model can pronounce.

    Idempotent, so it is safe to apply at more than one point in the pipeline
    — which it is, because three separate code paths produce a translation.
    """
    if not text:
        return ""

    spelled = spell_numbers(text, language)
    without_dashes = _LONE_DASH.sub(" ", spelled)

    return " ".join(without_dashes.split())


def unspeakable_digits(text: str | None) -> list[str]:
    """
    Digit runs still present after normalisation, in order.

    Empty means the text is safe to synthesize. `spell_numbers` has no
    ceiling, so on normalised text this is always empty and the guard it feeds
    never fires. That is deliberate, and it is not a reason to delete either:
    what it catches now is text that reached the export on a path where
    normalisation did not run, which is a live risk because three separate
    code paths produce a translation and each applies it separately.

    A non-empty result is a hard stop rather than a warning. IndicF5 renders a
    digit as noise, shipping one is exactly the failure this module exists
    for, and the export guard in `BundleExporter` is where it is enforced.
    """
    if not text:
        return []

    return _DIGIT_RUN.findall(text)
