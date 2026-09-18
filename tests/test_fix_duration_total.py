"""
`fix_duration` sets the **total** length, reference audio included.

FINETUNE_PLAN.md §0 records this read from f5_tts/infer/utils_infer.py (~line
447): `duration = int(fix_duration * sr / hop)`, replacing the byte-ratio
allocation rather than adding to it. The consequence for the probe is the whole
point — to get a `slot_seconds` clip you pass `ref_seconds + slot_seconds`, and
passing `slot_seconds` alone asks for a clip shorter than the reference, which
F5-TTS being an in-filling model will dutifully produce by cutting the sentence
off.

The failure this pins is silent in both directions. Pass the wrong total and
the model still returns clean audio of the wrong length; and if a future
IndicF5 stops forwarding the argument, nothing raises — the duration simply
reverts to the byte formula, which for an English reference generating
Devanagari over-allocates by about 2.15x (FINDINGS §5a) and fills the surplus
with invented speech. Both look like a model problem in a report.

The fake below implements the fix_duration branch, which the fake in
test_indicf5_diagnose.py deliberately does not — that one exists to prove the
patch intercepts a pre-bound import, and accepts `fix_duration` only to have
the real signature. No GPU and no model here either.
"""

import sys
import types

import pytest

torch = pytest.importorskip("torch")


SAMPLE_RATE = 24000
HOP_LENGTH = 256
FRAME_RATE = SAMPLE_RATE / HOP_LENGTH          # 93.75 frames per second
ONE_HOP_S = 1.0 / FRAME_RATE

# Same shape as the real utils_infer: module-level names resolved at call time,
# so patching them works. The one difference from the diagnose fake is the
# fix_duration branch, which is the behaviour under test.
FAKE_SOURCE = """
import torch

SAMPLE_RATE = 24000
HOP_LENGTH = 256


def chunk_text(text, max_chars=135):
    out, current = [], ""
    for word in text.split():
        if len((current + " " + word).encode("utf-8")) > max_chars and current:
            out.append(current)
            current = word
        else:
            current = (current + " " + word).strip()
    if current:
        out.append(current)
    return out


class Model:
    def __init__(self):
        self.durations = []

    def sample(self, cond, text, duration, steps=32, cfg_strength=2.0,
               sway_sampling_coef=-1):
        self.durations.append(duration)
        return duration, None


def infer_batch_process(ref_audio, ref_text, gen_text_batches, model_obj,
                        vocoder, mel_spec_type="vocos", progress=None,
                        target_rms=0.1, cross_fade_duration=0.15,
                        nfe_step=32, cfg_strength=2.0,
                        sway_sampling_coef=-1, speed=1, fix_duration=None,
                        device=None):
    audio, sr = ref_audio
    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "
    ref_audio_len = audio.shape[-1] // HOP_LENGTH
    for gen_text in gen_text_batches:
        if fix_duration is not None:
            # The whole point: a total, not an addition to the reference.
            duration = int(fix_duration * SAMPLE_RATE / HOP_LENGTH)
        else:
            ref_text_len = len(ref_text.encode("utf-8"))
            gen_text_len = len(gen_text.encode("utf-8"))
            duration = ref_audio_len + int(ref_audio_len / ref_text_len
                                           * gen_text_len / speed)
        model_obj.sample(cond=audio, text=[ref_text + gen_text],
                         duration=duration)
    return "wave", None


def infer_process(ref_audio, ref_text, gen_text, model_obj, vocoder,
                  speed=1, fix_duration=None):
    audio, sr = ref_audio
    max_chars = int(len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr)
                    * (25 - audio.shape[-1] / sr))
    batches = chunk_text(gen_text, max_chars=max_chars)
    return infer_batch_process((audio, sr), ref_text, batches, model_obj,
                               vocoder, speed=speed, fix_duration=fix_duration)
"""

REF_SECONDS = 10.0
SLOT_SECONDS = 3.68            # fixtures/en_speaker slot for sentence 6
ENGLISH_REF_TEXT = "a" * 135   # a Latin reference, as the probe uses
DEVANAGARI = "यह एक हिंदी वाक्य है जिसे मॉडल को बोलना है और यह काफी लंबा है"


@pytest.fixture
def diag(monkeypatch):
    utils = types.ModuleType("f5_tts.infer.utils_infer")
    exec(compile(FAKE_SOURCE, "fake_utils_infer", "exec"), utils.__dict__)

    package = types.ModuleType("f5_tts")
    infer = types.ModuleType("f5_tts.infer")
    infer.utils_infer = utils
    package.infer = infer

    monkeypatch.setitem(sys.modules, "f5_tts", package)
    monkeypatch.setitem(sys.modules, "f5_tts.infer", infer)
    monkeypatch.setitem(sys.modules, "f5_tts.infer.utils_infer", utils)

    module = pytest.importorskip("colab.indicf5_diagnose")

    # Bound up front, the way IndicF5's remote code binds infer_process.
    remote = utils.infer_process

    monkeypatch.setattr(utils, "_diagnose_installed", False, raising=False)
    module._calls.clear()
    module._pending.clear()
    module._mode.update({"speed": None, "one_chunk": True, "fix_duration": None})
    module.install_patches()

    return module, utils, remote


def run(utils, remote, text, fix_duration=None, ref_seconds=REF_SECONDS,
        ref_text=ENGLISH_REF_TEXT):
    audio = torch.zeros(1, int(ref_seconds * SAMPLE_RATE))
    model = utils.Model()
    remote((audio, SAMPLE_RATE), ref_text, text, model, None,
           fix_duration=fix_duration)
    return model


def test_the_generated_span_is_the_slot_when_the_total_includes_the_reference(diag):
    """The assertion the plan asks for, within one hop."""
    module, utils, remote = diag
    run(utils, remote, DEVANAGARI, fix_duration=REF_SECONDS + SLOT_SECONDS)

    call = module._calls[-1]
    assert call["requested_s"] == pytest.approx(SLOT_SECONDS, abs=ONE_HOP_S)


def test_passing_the_slot_alone_asks_for_less_than_nothing(diag):
    """
    The mistake the argument invites: `fix_duration=slot` is a *total* of 3.68s
    against a 10s reference, so the generated span comes back negative. Silent
    in the library, obvious here.
    """
    module, utils, remote = diag
    run(utils, remote, DEVANAGARI, fix_duration=SLOT_SECONDS)

    assert module._calls[-1]["requested_s"] < 0


def test_without_it_an_english_reference_over_allocates_as_before(diag):
    """
    The control. If fix_duration silently stopped being forwarded, this is the
    behaviour that returns — and FINDINGS §5a measured what it does to content.
    """
    module, utils, remote = diag
    run(utils, remote, DEVANAGARI, fix_duration=None)

    call = module._calls[-1]
    assert call["requested_s"] / call["natural_s"] > 1.8


def test_the_slot_is_honoured_whatever_the_script_costs_in_bytes(diag):
    """
    The reason to use fix_duration rather than `speed`: the byte ratio is what
    breaks across scripts, and a fixed total does not consult it. Devanagari at
    three bytes a character and Latin at one must land on the same duration.
    """
    module, utils, remote = diag
    total = REF_SECONDS + SLOT_SECONDS

    run(utils, remote, DEVANAGARI, fix_duration=total)
    devanagari = module._calls[-1]["requested_s"]

    run(utils, remote, "this is a latin sentence of some length", fix_duration=total)
    latin = module._calls[-1]["requested_s"]

    assert devanagari == pytest.approx(latin, abs=ONE_HOP_S)


def test_the_patch_injects_it_when_the_caller_cannot(diag):
    """
    The probe does not pass fix_duration to the model — IndicF5's remote
    __call__ chooses which of infer_process's arguments it forwards, and a
    dropped keyword reverts to the byte formula without raising. Injecting it
    inside the wrapper puts it in the call that actually happens.
    """
    module, utils, remote = diag
    module._mode["fix_duration"] = REF_SECONDS + SLOT_SECONDS

    run(utils, remote, DEVANAGARI)          # nothing passed by the caller

    call = module._calls[-1]
    assert call["fix_duration"] == pytest.approx(REF_SECONDS + SLOT_SECONDS)
    assert call["requested_s"] == pytest.approx(SLOT_SECONDS, abs=ONE_HOP_S)


def test_injection_refuses_an_install_that_would_ignore_it(diag):
    """
    The silent-failure mode this repo has been bitten by: a parameter accepted
    somewhere harmless and never applied. If the signature has no fix_duration,
    say so rather than inject nothing and report the byte formula's durations.
    """
    module, utils, remote = diag

    def without_fix_duration(ref_audio, ref_text, gen_text_batches, model_obj,
                             vocoder, speed=1, device=None):
        raise AssertionError("should not be reached")

    # Re-install against a signature that lacks the argument.
    utils.infer_batch_process = without_fix_duration
    utils._diagnose_installed = False
    module.install_patches()
    module._mode["fix_duration"] = REF_SECONDS + SLOT_SECONDS

    # Called the way a caller that knows nothing about fix_duration would call
    # it. Going through infer_process instead would fail one step earlier, at
    # the signature bind, because that fake passes the keyword along — also a
    # loud failure, but not the one this guard is for.
    audio = torch.zeros(1, int(REF_SECONDS * SAMPLE_RATE))
    with pytest.raises(TypeError, match="no fix_duration"):
        utils.infer_batch_process((audio, SAMPLE_RATE), ENGLISH_REF_TEXT,
                                  [DEVANAGARI], utils.Model(), None)


def test_a_longer_slot_produces_a_proportionally_longer_span(diag):
    module, utils, remote = diag

    run(utils, remote, DEVANAGARI, fix_duration=REF_SECONDS + 2.0)
    short = module._calls[-1]["requested_s"]

    run(utils, remote, DEVANAGARI, fix_duration=REF_SECONDS + 8.0)
    long = module._calls[-1]["requested_s"]

    assert long - short == pytest.approx(6.0, abs=ONE_HOP_S)
