"""
The decode parameters must be names the ASR actually accepts.

This pins a failure that reached a GPU. `ASR_DECODE` was written by copying the
pinned parameters out of `scripts/transcribe_fixtures.py`, which drives
**faster-whisper**, into `colab/indicf5_check.py`, which drives **transformers**.
The two libraries disagree on one name:

    faster-whisper   condition_on_previous_text
    transformers     condition_on_prev_tokens

Nothing rejects the wrong name where it is written. It is forwarded through
`generate_kwargs` to a `generate()` that does not declare it, on to the model's
forward, and raises there — once per clip, inside a per-row `except`. The
28-clip transliteration probe therefore ran to completion, wrote its audio,
confirmed `fix_duration` had been applied, and printed a full report in which
every content number was nan. Because nan compares False against every
threshold, each arm showed zero bad clips and the `latin` negative control —
which FINDINGS §4 measured producing speech that is not English — came back
"clean".

So the assertion is not that the dictionary has particular keys. It is that
every key is one the installed `generate()` declares, which is the thing that
was never checked.
"""

import inspect

import pytest

check = pytest.importorskip("colab.indicf5_check")
whisper_generation = pytest.importorskip(
    "transformers.models.whisper.generation_whisper")

FASTER_WHISPER_ONLY = "condition_on_previous_text"


@pytest.fixture
def accepted():
    parameters = inspect.signature(
        whisper_generation.WhisperGenerationMixin.generate).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        # generate() takes **kwargs, so a bad name is not rejected at the bind.
        # The declared names are still the ones it acts on; everything else is
        # forwarded to the model and raises there, which is the bug.
        pass
    return set(parameters)


def test_every_decode_parameter_is_one_generate_declares(accepted):
    unknown = sorted(set(check.ASR_DECODE) - accepted)
    assert not unknown, (
        f"{unknown} would be forwarded to the model and raise per clip, "
        "leaving every content metric nan on a run that looks complete"
    )


def test_the_faster_whisper_spelling_is_not_used(accepted):
    """
    Named explicitly because it is the one that was wrong, and because it is
    still correct in scripts/transcribe_fixtures.py — so a future reader
    comparing the two files will see the same concept spelled two ways and
    needs to know that is deliberate.
    """
    assert FASTER_WHISPER_ONLY not in check.ASR_DECODE
    assert FASTER_WHISPER_ONLY not in accepted


def test_the_repetition_guard_is_actually_set(accepted):
    """
    Renaming the key must not quietly drop what it was for. Whisper conditions
    on its own previous output and continues a repetition once it starts;
    FINDINGS §13 records the hallucinated repeated tail this produces.
    """
    assert check.ASR_DECODE["condition_on_prev_tokens"] is False


def test_greedy_decoding_is_pinned():
    """
    The other half: Whisper's default retries a failed decode at rising
    temperatures, and those retries sample. Without this, two runs of the same
    clip disagree and part of any arm difference is decode noise.
    """
    assert check.ASR_DECODE["temperature"] == 0.0


def test_the_guard_reports_nothing_for_the_shipping_parameters():
    """The names on disk must pass the check that runs before every batch."""
    assert check.unsupported_decode_params() == []


def test_the_guard_catches_a_name_generate_does_not_declare(monkeypatch):
    """
    The faster-whisper spelling, put back. This is what should have raised
    once at the top of a run instead of 28 times inside a per-row except.
    """
    monkeypatch.setitem(check.ASR_DECODE, FASTER_WHISPER_ONLY, False)

    assert check.unsupported_decode_params() == [FASTER_WHISPER_ONLY]
