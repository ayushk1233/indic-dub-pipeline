"""
IndicF5 was asked for English, which it does not claim to support, and Whisper
looped on the result: a 3.97 second clip transcribed to 2074 characters of
"the process of making" repeated. Scored as content that reads as a 48 second
prefix on a four second clip, and the report printed it as a measurement.

The guard is a physical one, not a quality one.
"""

import numpy as np
import pytest
import soundfile as sf

from colab.indicf5_check import LOOP_CPS, transcript_impossible


@pytest.fixture
def clip(tmp_path):
    def build(seconds, rate=24000):
        path = tmp_path / f"{seconds}.wav"
        sf.write(str(path), np.zeros(int(seconds * rate), dtype="float32"), rate)
        return path
    return build


def test_the_real_loop(clip):
    """3.97s of audio, 2074 characters of transcript: 522 cps."""
    heard = "the process of making " * 95
    assert len(heard) > 2000
    assert transcript_impossible(heard, clip(3.97))


def test_ordinary_hindi_is_not_flagged(clip):
    # 134 characters over 12.4 seconds, the longest real sentence in the set.
    assert not transcript_impossible("क" * 134, clip(12.4))


def test_the_fastest_thing_this_project_has_synthesized_is_not_flagged(clip):
    """The broken hi_ref arm ran at 22.9 cps and was still real speech."""
    assert not transcript_impossible("क" * 229, clip(10.0))


def test_an_empty_transcript_is_not_a_loop(clip):
    assert not transcript_impossible("", clip(4.0))


def test_a_missing_file_does_not_raise(tmp_path):
    assert not transcript_impossible("anything", tmp_path / "absent.wav")


def test_the_boundary_is_the_declared_one(clip):
    assert not transcript_impossible("x" * int(LOOP_CPS * 10), clip(10.0))
    assert transcript_impossible("x" * int(LOOP_CPS * 10 + 20), clip(10.0))
