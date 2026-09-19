"""
The IndicF5 worker is the first thing in this project that will run the model
on real pipeline output rather than on a hand-made fixture, and its three
model-specific obligations all fail silently if they are not met:

  - a missing reference transcript clones somebody else's voice (§4),
  - an unnormalised one puts invented speech in front of the sentence (§5c),
  - an uninstrumented run ignores `fix_duration` and returns whatever the
    byte formula asked for, which is wrong by 2.46x (§5a).

None of those raise on a GPU. They come back as audio. So they are asserted
here, against a fake model and a fake `_calls`, with no GPU and no weights.
"""

import json
import sys
import types
from pathlib import Path

import pytest

from src.stages.tts.models import (
    SynthesisRequest,
    SynthesisSegment,
)

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

from colab.indicf5_worker import (  # noqa: E402
    MAX_FIXED_SPAN_S,
    SUPPORTED_BUNDLE_VERSIONS,
    IndicF5Worker,
    InstrumentationError,
)

REFERENCE_SECONDS = 10.0
SAMPLE_RATE = 24000


def build_bundle(tmp_path, *, version="1.2", reference_text="So let me tell you.",
                 slots=(2.0, 3.0), reference_seconds=REFERENCE_SECONDS,
                 budgets=None):
    """A bundle on disk, exactly as BundleExporter writes one."""
    bundle = tmp_path / "tts_bundle"
    (bundle / "request").mkdir(parents=True)

    reference = bundle / "request" / "reference.wav"
    samples = int(reference_seconds * SAMPLE_RATE)
    sf.write(str(reference),
             np.zeros(samples, dtype=np.float32), SAMPLE_RATE, subtype="PCM_16")

    segments = []
    start = 0.0
    for index, slot in enumerate(slots):
        segments.append(SynthesisSegment(
            segment_id=index, chunk_id=0,
            start_ts=start, end_ts=start + slot,
            text="यह आवाज़ मेरी है।",
            reference_audio="request/reference.wav",
            budget_s=None if budgets is None else budgets[index],
        ))
        start += slot

    request = SynthesisRequest(
        job_id="test", language="hi",
        output_sample_rate=SAMPLE_RATE,
        reference_text=reference_text,
        segments=segments,
    )
    (bundle / "request" / "synthesis_request.json").write_text(
        json.dumps(request.model_dump(), ensure_ascii=False), encoding="utf-8")

    (bundle / "manifest.json").write_text(json.dumps({
        "metadata": {"bundle_version": version, "job_id": "test"},
        "paths": {"request_json": "request/synthesis_request.json",
                  "reference_audio": "request/reference.wav"},
    }), encoding="utf-8")

    return bundle


class FakeModel:
    """Records what it was asked for and returns audio of the asked length."""

    def __init__(self, mode, calls, peak=1.0):
        self.mode = mode
        self.calls = calls
        self.peak = peak
        self.seen = []

    def __call__(self, text, ref_audio_path=None, ref_text=None):
        self.seen.append({"text": text, "ref_audio_path": ref_audio_path,
                          "ref_text": ref_text,
                          "fix_duration": self.mode["fix_duration"],
                          "one_chunk": self.mode["one_chunk"]})

        fixed = self.mode["fix_duration"]
        span = (fixed - REFERENCE_SECONDS) if fixed else 1.0
        self.calls.append({"fix_duration": fixed, "requested_s": span,
                           "ref_s": REFERENCE_SECONDS})
        return np.full(int(span * SAMPLE_RATE), self.peak, dtype=np.float32)


@pytest.fixture
def worker(tmp_path, monkeypatch):
    """A worker whose model and instrumentation are fakes, not f5_tts."""
    mode = {"speed": None, "one_chunk": False, "fix_duration": None}
    calls = []

    fake = types.ModuleType("colab.indicf5_diagnose")
    fake._mode = mode
    fake._calls = calls
    fake.install_patches = lambda: None
    fake.reset_mode = lambda: mode.update(
        {"speed": None, "one_chunk": False, "fix_duration": None})
    monkeypatch.setitem(sys.modules, "colab.indicf5_diagnose", fake)

    def make(bundle, peak=1.0):
        w = IndicF5Worker(bundle)
        w.load_bundle()
        w.prepare_reference()
        w.model = FakeModel(mode, calls, peak=peak)
        return w

    return make


# -- the reference transcript ------------------------------------------------


def test_a_bundle_without_a_reference_transcript_is_refused(tmp_path, worker):
    """IndicF5 cannot clone from audio alone; running anyway is a wrong voice."""
    bundle = build_bundle(tmp_path, reference_text="")

    w = IndicF5Worker(bundle)
    w.load_bundle()

    with pytest.raises(ValueError, match="reference_text"):
        w.prepare_reference()


def test_bundle_1_0_is_refused_because_it_cannot_carry_a_transcript(tmp_path):
    assert "1.0" not in SUPPORTED_BUNDLE_VERSIONS

    bundle = build_bundle(tmp_path, version="1.0")

    with pytest.raises(ValueError, match="Unsupported bundle version"):
        IndicF5Worker(bundle).load_bundle()


def test_the_reference_transcript_is_normalised_before_the_model_sees_it(
        tmp_path, worker):
    """
    §5c: punctuation in ref_text is text the model must place somewhere, and
    it places it at the start of the kept region. Measured 1-of-7 to 0-of-7.
    """
    bundle = build_bundle(
        tmp_path, reference_text="So let me tell you what this PROJECT does.")
    w = worker(bundle)

    w.synthesize_segment(w.request.segments[0])

    sent = w.model.seen[0]["ref_text"]
    assert sent == "so let me tell you what this project does"
    assert "." not in sent


# -- duration ----------------------------------------------------------------


def test_fix_duration_is_the_slot_plus_the_reference_not_the_slot(
        tmp_path, worker):
    """
    §5a: fix_duration sets TOTAL frames, reference included. Passing the slot
    alone asks for a clip shorter than the reference and is the single most
    expensive misreading in this project's history.
    """
    bundle = build_bundle(tmp_path, slots=(2.0, 3.0))
    w = worker(bundle)

    for segment in w.request.segments:
        w.synthesize_segment(segment)

    asked = [call["fix_duration"] for call in w.model.seen]
    assert asked == [REFERENCE_SECONDS + 2.0, REFERENCE_SECONDS + 3.0]


def test_a_fixed_duration_run_is_always_a_single_chunk(tmp_path, worker):
    """A chunked generation prepends the reference to every chunk (§5a)."""
    bundle = build_bundle(tmp_path, slots=(2.0,))
    w = worker(bundle)

    w.synthesize_segment(w.request.segments[0])

    assert w.model.seen[0]["one_chunk"] is True


def test_a_slot_past_the_single_chunk_cap_drops_fix_duration(tmp_path, worker):
    """
    §16a: one chunk is clean to 19.7s and breaks by 21.2s. Past the cap the
    honest thing is to stop forcing the span, not to force it and ship the
    repetition.
    """
    bundle = build_bundle(tmp_path, slots=(MAX_FIXED_SPAN_S + 5.0,))
    w = worker(bundle)

    w.synthesize_segment(w.request.segments[0])

    assert w.model.seen[0]["fix_duration"] is None
    assert w.model.seen[0]["one_chunk"] is False


# -- the instrumentation guard -----------------------------------------------


def test_an_uninstrumented_segment_raises_rather_than_returning_audio(
        tmp_path, worker):
    """
    An empty `_calls` means the wrappers are not bound and fix_duration was
    ignored, so the clip's length came from the byte formula. §13a: this cost
    two whole GPU runs, and both times the audio looked fine.
    """
    bundle = build_bundle(tmp_path, slots=(2.0,))
    w = worker(bundle)

    class Unpatched:
        """Real audio back, and nothing recorded — an unbound wrapper."""

        def __call__(self, text, ref_audio_path=None, ref_text=None):
            return np.zeros(SAMPLE_RATE, dtype=np.float32)

    w.model = Unpatched()

    import sys as _sys
    assert not _sys.modules["colab.indicf5_diagnose"]._calls

    with pytest.raises(InstrumentationError, match="RESTART THE RUNTIME"):
        w.synthesize_segment(w.request.segments[0])

    # and no file was left behind for the importer to accept as a result
    assert not (bundle / "output" / "seg_00000.wav").exists()


def test_an_instrumentation_failure_aborts_the_run_it_does_not_fail_segments(
        tmp_path, worker, monkeypatch):
    """
    The per-segment handler exists so one bad segment does not lose the job.
    An unbound lever is not one bad segment — every clip after it is wrong
    the same way, so it must stop.
    """
    bundle = build_bundle(tmp_path, slots=(2.0, 3.0, 4.0))
    w = worker(bundle)

    monkeypatch.setattr(w, "load_bundle", lambda: None)
    monkeypatch.setattr(w, "prepare_reference", lambda: None)
    monkeypatch.setattr(w, "load_model", lambda: w.model)
    monkeypatch.setattr(w, "synthesize_segment", lambda segment: (_ for _ in ()).throw(
        InstrumentationError("RESTART THE RUNTIME")))

    with pytest.raises(InstrumentationError):
        w.run()

    # and it did not mark three segments failed on the way out
    written = json.loads((bundle / "output" / "synthesis_result.json").read_text())
    assert written["segments"] == []


# -- audio -------------------------------------------------------------------


def test_int16_scaled_output_is_rescaled_rather_than_written_as_is(
        tmp_path, worker):
    """
    IndicF5 returns [-1, 1] on some paths and int16-scaled floats on others.
    Writing the latter as PCM_16 clips every sample to full scale.
    """
    bundle = build_bundle(tmp_path, slots=(2.0,))
    w = worker(bundle, peak=20000.0)

    result = w.synthesize_segment(w.request.segments[0])

    written, rate = sf.read(str(bundle / result.audio_path))
    assert rate == SAMPLE_RATE
    assert float(np.max(np.abs(written))) < 1.0


def test_audio_is_written_as_integer_pcm_for_the_stdlib_wave_reader(
        tmp_path, worker):
    """The local importer reads these with `wave`, which is integer PCM only."""
    bundle = build_bundle(tmp_path, slots=(2.0,))
    w = worker(bundle)

    result = w.synthesize_segment(w.request.segments[0])

    assert sf.info(str(bundle / result.audio_path)).subtype == "PCM_16"


def test_the_result_names_the_model_and_the_levers_that_produced_it(
        tmp_path, worker):
    """
    §5e's rule: the settings that produced a given audio file belong in the
    artifact beside it, because a decoder parameter changing silently is this
    project's central failure mode.
    """
    bundle = build_bundle(tmp_path, slots=(2.0,))
    w = worker(bundle)
    w.synthesized = [w.synthesize_segment(w.request.segments[0])]

    result = w.build_result()

    assert result.model_id == "ai4bharat/IndicF5"
    assert result.params["reference_seconds"] == REFERENCE_SECONDS
    assert result.params["reference_text_normalization"] == "plain"


def test_identity_is_reported_as_unmeasured_rather_than_as_a_number(
        tmp_path, worker):
    """
    IndicF5 exposes no speaker encoder and XTTS is out of the project, so
    identity is genuinely unmeasured in the pipeline (§17). A fabricated
    similarity would read as a measurement.
    """
    bundle = build_bundle(tmp_path, slots=(2.0,))
    w = worker(bundle)

    result = w.synthesize_segment(w.request.segments[0])

    assert result.speaker_similarity is None
    assert result.gpt_tokens is None


# -- the reference length, which is where XTTS's default is wrong ------------


def test_a_reference_past_the_clipping_threshold_warns(tmp_path, worker, capsys):
    """
    §5d: IndicF5 clips reference audio at 15s and never shortens ref_text, so
    a 25s reference — which is XTTS's default and was this pipeline's — hands
    the model a transcript describing audio it cannot hear.
    """
    bundle = build_bundle(tmp_path, reference_seconds=25.0)

    w = IndicF5Worker(bundle)
    w.load_bundle()
    w.prepare_reference()

    assert "does NOT" in capsys.readouterr().out


# -- the budget, which is not the slot ---------------------------------------


def test_the_span_asked_for_is_the_budget_the_text_was_chosen_against(
        tmp_path, worker):
    """
    Length control picks a candidate that fits `budget_s`, which pools the
    pause after the segment; the assembly cascade absorbs the overhang. On
    english.mov ten of fourteen segments had no candidate that fit even the
    budget, so compressing further into the bare slot is compression this
    pipeline never asked for.
    """
    bundle = build_bundle(tmp_path, slots=(2.0,), budgets=(3.4,))
    w = worker(bundle)

    w.synthesize_segment(w.request.segments[0])

    assert w.model.seen[0]["fix_duration"] == REFERENCE_SECONDS + 3.4


def test_a_bundle_without_a_budget_falls_back_to_the_slot(tmp_path, worker):
    """Bundle 1.1 carries no budget. Falling back must not raise."""
    bundle = build_bundle(tmp_path, version="1.1", slots=(2.0,), budgets=None)
    w = worker(bundle)

    w.synthesize_segment(w.request.segments[0])

    assert w.model.seen[0]["fix_duration"] == REFERENCE_SECONDS + 2.0


def test_the_cap_is_applied_to_the_budget_not_the_slot(tmp_path, worker):
    """
    A short slot with a long budget still generates for the whole budget, so
    it is the budget that can cross the single-chunk cap (§16a).
    """
    bundle = build_bundle(tmp_path, slots=(2.0,),
                          budgets=(MAX_FIXED_SPAN_S + 5.0,))
    w = worker(bundle)

    w.synthesize_segment(w.request.segments[0])

    assert w.model.seen[0]["fix_duration"] is None


def test_the_exported_bundle_carries_a_budget_for_every_segment():
    """
    The worker's fallback is silent by design, so nothing downstream would
    notice the exporter quietly dropping the field.
    """
    from src.stages.tts.processor import TTSProcessor
    from src.stages.translation.models import (
        TranslatedSegment,
        TranslationResult,
    )

    translation = TranslationResult(
        job_id="t", source_language="en", target_language="hi",
        segments=[
            TranslatedSegment(
                segment_id=i, chunk_id=0,
                start_ts=float(i * 5), end_ts=float(i * 5) + 2.0,
                source_text="x", translated_text="y",
                source_language="en", target_language="hi",
            )
            for i in range(3)
        ],
    )

    request = TTSProcessor().build_request(translation, "request/reference.wav")

    assert all(s.budget_s is not None for s in request.segments)
    # The pause after each of the first two is pooled, so the budget exceeds
    # the 2s slot; the last has no follower and keeps its slot.
    assert request.segments[0].budget_s > 2.0
    assert request.segments[-1].budget_s == pytest.approx(2.0)


# -- Latin loanwords in the generated text -----------------------------------


def test_a_latin_loanword_is_folded_before_the_model_sees_it(tmp_path, worker):
    """
    Hindi ASR writes English loanwords in Latin, and IndicF5 cannot generate
    English (§4). Two of sixteen segments from hindi.mov carried one —
    `project` and `fit` — and both are in the table.
    """
    bundle = build_bundle(tmp_path, slots=(2.0,))
    w = worker(bundle)
    w.request.segments[0].text = "ये project असल में क्या करता है"

    w.synthesize_segment(w.request.segments[0])

    sent = w.model.seen[0]["text"]
    assert "project" not in sent
    assert "प्रोजेक्ट" in sent


def test_an_unmapped_latin_word_is_reported_rather_than_guessed(
        tmp_path, worker, capsys):
    """
    Guessing a Devanagari spelling is the drafting judgement §4f reserves for
    a human. Passing it through silently is the failure the listener hears.
    """
    bundle = build_bundle(tmp_path, slots=(2.0,))
    w = worker(bundle)
    w.request.segments[0].text = "ये kubernetes असल में क्या करता है"

    w.synthesize_segment(w.request.segments[0])

    out = capsys.readouterr().out
    assert "kubernetes" in out
    assert "loanword table" in out


def test_folding_leaves_devanagari_alone(tmp_path, worker):
    bundle = build_bundle(tmp_path, slots=(2.0,))
    w = worker(bundle)
    original = "यह आवाज़ मेरी है।"
    w.request.segments[0].text = original

    w.synthesize_segment(w.request.segments[0])

    assert w.model.seen[0]["text"] == original
