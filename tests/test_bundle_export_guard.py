"""
The export refuses text the model cannot pronounce.

This is the check that would have caught the run described in
`src/text/normalize.py`: two dubbed videos shipped with nine corrupted
segments, every timing metric green, the fault found only by listening. A
digit costs a GPU round trip to discover by ear and nothing at all to catch
here.

The guard is deliberately at export rather than at synthesis. The bundle is
the artifact that leaves this machine, so it is the last point where a local
test can still fail cheaply.
"""

import wave

import pytest

from src.stages.tts.bundle.exporter import BundleExporter, UnspeakableText
from src.stages.tts.models import SynthesisRequest, SynthesisSegment


def write_wav(path, seconds=1.0, rate=24000):
    path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))

    return path


def request_saying(*texts):
    return SynthesisRequest(
        job_id="guard",
        language="hi",
        reference_text="सो लेट मी टेल यू",
        segments=[
            SynthesisSegment(
                segment_id=index,
                chunk_id=0,
                start_ts=float(index),
                end_ts=float(index) + 1.0,
                text=text,
                reference_audio="request/reference.wav",
            )
            for index, text in enumerate(texts)
        ],
    )


NATURAL_CPS = 12.19  # measured Hindi, §5a


def export(tmp_path, request):
    """
    Export with a reference clip long enough for its own transcript.

    Sized rather than fixed because the length guard measures characters per
    second: a 1s stand-in would fail every test here for the wrong reason,
    and each test should fail only on what it is about.
    """
    text = (request.reference_text or "").strip()
    seconds = max(len(text) / NATURAL_CPS, 1.0)
    reference = write_wav(tmp_path / "reference.wav", seconds=seconds)

    return BundleExporter().export(request, reference, tmp_path / "tts_bundle")


def test_spelled_out_text_exports_normally(tmp_path):
    out = export(tmp_path, request_saying("सात असंभव थे ।", "बीस प्रतिशत अधिक"))

    assert (out / "request" / "synthesis_request.json").exists()


def test_a_digit_stops_the_export(tmp_path):
    with pytest.raises(UnspeakableText):
        export(tmp_path, request_saying("150 सेगमेंट प्रोसेस हुए"))


def test_the_error_names_the_segment_and_the_number(tmp_path):
    """
    A bundle can hold dozens of segments. An error that only says "a digit"
    sends someone reading JSON by hand.
    """
    request = request_saying("सात असंभव थे", "इकतीस फीट", "150 सेगमेंट")

    with pytest.raises(UnspeakableText) as caught:
        export(tmp_path, request)

    assert "segment 2" in str(caught.value)
    assert "150" in str(caught.value)


def test_nothing_is_left_behind_when_the_guard_fires(tmp_path):
    """
    The guard runs before the directory is built, so a refused export does not
    leave a half-written bundle for the importer to find later.
    """
    with pytest.raises(UnspeakableText):
        export(tmp_path, request_saying("200 सेगमेंट"))

    assert not (tmp_path / "tts_bundle" / "request").exists()


def test_every_offending_segment_is_reported_at_once(tmp_path):
    """Fixing them one GPU run at a time is the failure mode to avoid."""
    request = request_saying("150 सेगमेंट", "ठीक है", "300 और")

    with pytest.raises(UnspeakableText) as caught:
        export(tmp_path, request)

    message = str(caught.value)

    assert "2 segment(s)" in message
    assert "150" in message and "300" in message


# -- the reference transcript must be in the generated script -----------------


from src.stages.tts.bundle.exporter import ReferenceScriptMismatch


def request_with_reference(reference_text):
    request = request_saying("सात असंभव थे", "बीस प्रतिशत अधिक")
    request.reference_text = reference_text

    return request


def test_a_devanagari_reference_transcript_exports(tmp_path):
    out = export(tmp_path, request_with_reference("सो लेट मी टेल यू व्हाट दिस"))

    assert (out / "request" / "synthesis_request.json").exists()


def test_a_latin_reference_transcript_is_refused(tmp_path):
    """
    This shipped once. The pipeline transcribes the reference clip to build
    this text, so on an en->hi job it comes back in Latin — English audio
    described to a model about to generate Devanagari. §16e measured the cost
    at 4 leading prefixes in 24 clips, and it is inaudible in every timing
    metric.
    """
    with pytest.raises(ReferenceScriptMismatch):
        export(tmp_path, request_with_reference(
            "So let me tell you what this project actually does."))


def test_the_refusal_says_what_to_do_about_it(tmp_path):
    with pytest.raises(ReferenceScriptMismatch) as caught:
        export(tmp_path, request_with_reference("So let me tell you"))

    message = str(caught.value)

    assert "--reference-text" in message
    assert "0%" in message


# hi_dub's own reference transcript, verbatim. Hindi ASR wrote two English
# loanwords in Latin — `project` and `fit` — which is what a real transcript
# looks like. It scores 0.97.
REAL_HI_REFERENCE = (
    "मैं आपको बताता हूँ कि ये project असल में करता गया है आप इसे एक वीडियो "
    "देते हैं कोई लेक्चर कोई इंटरव्यू कुछ भी जिसमें साफ आवाज हो और यह वही "
    "वीडियो हिंदी में बोलता हुआ वापस देता है आसान लगता है है ना लेकिन नहीं "
    "और वही रखते हैं जो समय में fit हो"
)

# The same text with a third Latin word, standing in for slightly worse ASR.
# It must still export: the guard is for a transcript in the wrong script, not
# for loanwords.
NOISIER_HI_REFERENCE = REAL_HI_REFERENCE + " और system भी"


def test_a_real_transcript_with_latin_loanwords_still_exports(tmp_path):
    """
    Hindi ASR writes loanwords in Latin, so a real reference transcript
    carries islands of it. The threshold has to pass those while refusing a
    transcript that is in the wrong script outright — the gap is 0.97 against
    0.00, so it is not a close call.
    """
    for text in (REAL_HI_REFERENCE, NOISIER_HI_REFERENCE):
        out = export(tmp_path, request_with_reference(text))

        assert (out / "request" / "synthesis_request.json").exists()


def test_the_threshold_sits_between_the_two_real_cases(tmp_path):
    """
    Pins the margin rather than trusting the constant: a measured-good
    transcript must clear it and a measured-bad one must not, or the guard is
    tuned to nothing.
    """
    from src.eval.translation_metrics import script_ratio
    from src.stages.tts.bundle.exporter import MIN_REFERENCE_SCRIPT_RATIO

    good = script_ratio(REAL_HI_REFERENCE, "hi")
    noisier = script_ratio(NOISIER_HI_REFERENCE, "hi")
    bad = script_ratio("So let me tell you what this project actually does.", "hi")

    assert bad < MIN_REFERENCE_SCRIPT_RATIO <= noisier <= good
    # Real ASR output clears the bar by a wide margin, so a guard firing means
    # the script is genuinely wrong rather than the transcript being untidy.
    assert noisier - MIN_REFERENCE_SCRIPT_RATIO > 0.05


# -- the reference transcript must fit its audio ------------------------------


from src.stages.tts.bundle.exporter import (
    MAX_REFERENCE_CPS,
    ReferenceLengthMismatch,
)


def test_a_transcript_that_fits_its_clip_exports(tmp_path):
    """Roughly natural Hindi: ~12 characters per second."""
    reference = write_wav(tmp_path / "reference.wav", seconds=10.0)
    request = request_with_reference("क" * 120)

    out = BundleExporter().export(request, reference, tmp_path / "tts_bundle")

    assert (out / "request" / "synthesis_request.json").exists()


def test_a_transcript_describing_the_whole_video_is_refused(tmp_path):
    """
    The exact shipped case: a 13.9s clip carrying the 634-character
    transcript of a 67s video. `_build_reference` passed None to
    `_transcript_for`, which means "all of it", on any re-export where the
    reference clip already existed.
    """
    reference = write_wav(tmp_path / "reference.wav", seconds=13.9)
    request = request_with_reference("क" * 634)

    with pytest.raises(ReferenceLengthMismatch):
        BundleExporter().export(request, reference, tmp_path / "tts_bundle")


def test_the_refusal_quantifies_the_mismatch(tmp_path):
    reference = write_wav(tmp_path / "reference.wav", seconds=13.9)

    with pytest.raises(ReferenceLengthMismatch) as caught:
        BundleExporter().export(
            request_with_reference("क" * 634), reference, tmp_path / "tts_bundle"
        )

    message = str(caught.value)

    assert "634 characters" in message
    assert "45.6" in message


def test_the_limit_clears_real_reference_geometry(tmp_path):
    """
    Pins the margin against the shipping configuration rather than trusting
    the constant: a 10s IndicF5 reference at natural pace must sit well
    under the limit, or the guard would refuse correct bundles.
    """
    natural_cps = 12.19

    assert natural_cps * 1.5 < MAX_REFERENCE_CPS
