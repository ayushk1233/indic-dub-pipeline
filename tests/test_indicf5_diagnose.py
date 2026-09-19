"""
The diagnostic's claim is that it measures IndicF5 rather than modelling it.
That rests on one assumption: patching a name in f5_tts.infer.utils_infer
intercepts calls made through infer_process even when IndicF5's remote code
imported infer_process directly. If that assumption is wrong the report would
print this file's arithmetic and call it a measurement.

These tests stand up a fake utils_infer with the real signatures and check the
interception, the recording, and the two overrides. No GPU and no model.
"""

import sys
import types

import pytest

torch = pytest.importorskip("torch")


# The real utils_infer resolves chunk_text and infer_batch_process as module
# globals at call time, which is exactly what makes them patchable. Defining
# them as nested functions here would make them closure cells instead and the
# test would prove nothing, so the fake is exec'd into a module dict the same
# way an import would build one.
FAKE_SOURCE = """
import torch


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
    ref_audio_len = audio.shape[-1] // 256
    for gen_text in gen_text_batches:
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


def build_fake_f5(monkeypatch):
    """A stand-in for f5_tts.infer.utils_infer with the real call shapes."""
    utils = types.ModuleType("f5_tts.infer.utils_infer")
    exec(compile(FAKE_SOURCE, "fake_utils_infer", "exec"), utils.__dict__)

    package = types.ModuleType("f5_tts")
    infer = types.ModuleType("f5_tts.infer")
    infer.utils_infer = utils
    package.infer = infer

    monkeypatch.setitem(sys.modules, "f5_tts", package)
    monkeypatch.setitem(sys.modules, "f5_tts.infer", infer)
    monkeypatch.setitem(sys.modules, "f5_tts.infer.utils_infer", utils)
    return utils


@pytest.fixture
def diag(monkeypatch):
    utils = build_fake_f5(monkeypatch)
    module = pytest.importorskip("colab.indicf5_diagnose")

    # Emulate IndicF5's remote code: bind infer_process once, up front, the
    # way `from ... import infer_process` does, before any patch exists.
    remote_infer_process = utils.infer_process

    monkeypatch.setattr(utils, "_diagnose_installed", False, raising=False)
    module._calls.clear()
    module._pending.clear()
    module._mode.update({"speed": None, "one_chunk": False})
    module.install_patches()

    return module, utils, remote_infer_process


def run(utils, remote, text, ref_seconds=10.0, ref_text="a" * 135):
    audio = torch.zeros(1, int(ref_seconds * 24000))
    model = utils.Model()
    remote((audio, 24000), ref_text, text, model, None)
    return model


HINDI = "यह एक हिंदी वाक्य है जिसे मॉडल को बोलना है और यह काफी लंबा है"

# A 10.5s English reference sets max_chars near 200 bytes, which is only
# about 67 Devanagari characters. The real sentences run past 130, so the
# split path is the normal case on that arm rather than an edge case.
LONG_HINDI = " ".join([HINDI] * 3)


def test_patch_intercepts_a_pre_bound_infer_process(diag):
    module, utils, remote = diag
    run(utils, remote, HINDI)
    assert module._calls, "the patch never fired through a pre-bound import"


def test_records_the_duration_the_sampler_was_actually_given(diag):
    module, utils, remote = diag
    model = run(utils, remote, HINDI)

    call = module._calls[-1]
    ref_frames = int(10.0 * 24000) // 256
    expected = (sum(model.durations) - len(model.durations) * ref_frames) / (24000 / 256)
    assert call["requested_s"] == pytest.approx(expected, rel=1e-6)


def test_english_reference_over_allocates_for_devanagari(diag):
    """The bug itself: a 1-byte reference pricing a 3-byte script."""
    module, utils, remote = diag
    run(utils, remote, HINDI)

    call = module._calls[-1]
    assert call["requested_s"] / call["natural_s"] > 1.8


def test_hindi_reference_does_not(diag):
    module, utils, remote = diag
    # 104 Devanagari characters, as fixtures/hindi_reference_short.wav has.
    run(utils, remote, HINDI, ref_text="क" * 104)

    call = module._calls[-1]
    assert 0.8 < call["requested_s"] / call["natural_s"] < 1.3


def test_auto_speed_brings_allocation_back_to_natural(diag):
    module, utils, remote = diag
    module._mode["speed"] = "auto"
    run(utils, remote, HINDI)

    call = module._calls[-1]
    assert call["speed"] > 1.5
    assert call["requested_s"] / call["natural_s"] == pytest.approx(1.0, abs=0.05)


def test_auto_speed_leaves_chunking_alone(diag):
    """The two arms must differ in exactly one thing."""
    module, utils, remote = diag

    run(utils, remote, HINDI)
    base_chunks = module._calls[-1]["chunks"]

    module._mode["speed"] = "auto"
    run(utils, remote, HINDI)
    assert module._calls[-1]["chunks"] == base_chunks


def test_one_chunk_forces_a_single_batch_without_touching_duration(diag):
    module, utils, remote = diag

    run(utils, remote, LONG_HINDI)
    base = module._calls[-1]
    assert base["chunks"] > 1, "fixture no longer exercises the split path"

    module._mode["one_chunk"] = True
    run(utils, remote, LONG_HINDI)
    forced = module._calls[-1]

    assert forced["chunks"] == 1
    assert forced["speed"] == base["speed"]

    # Not exactly equal: chunk_text strips the whitespace at each split, so
    # rejoining costs one byte per boundary. That is well under a percent of
    # the allocation and is the only thing the one_chunk arm changes besides
    # the chunk count itself.
    assert forced["gen_bytes"] == pytest.approx(base["gen_bytes"], rel=0.01)


def test_installing_twice_does_not_stack_wrappers(diag):
    module, utils, remote = diag
    module.install_patches()
    run(utils, remote, HINDI)
    assert len(module._calls) == 1


def test_sample_is_restored_after_the_call(diag):
    module, utils, remote = diag
    model = run(utils, remote, HINDI)
    assert "sample" not in vars(model)


def test_english_reference_splits_a_real_length_sentence(diag):
    """
    The second fault, stated as a fact about the two references rather than a
    claim about the model: the same sentence takes a different code path
    depending only on which language the reference clip is in.
    """
    module, utils, remote = diag

    run(utils, remote, LONG_HINDI)
    english_chunks = module._calls[-1]["chunks"]

    run(utils, remote, LONG_HINDI, ref_text="क" * 104)
    hindi_chunks = module._calls[-1]["chunks"]

    assert english_chunks > hindi_chunks


# ------------------------------------ surviving colab/reimport.py's fresh()


def test_a_reimported_module_rebinds_the_patch(monkeypatch):
    """
    The failure that cost a run on 2026-09-19.

    The wrappers close over this module's `_mode` and `_calls`. `f5_tts` is not
    ours, so `fresh()` does not purge it: after a re-import the wrappers still
    point at the PREVIOUS module's dictionaries while every caller writes to
    the new ones. The old guard was a boolean on utils_infer, so
    install_patches() returned early and never rebound them.

    What came out was 24 clips at exactly 3.82s whatever the text — 3.82s being
    the target of the last clip of the previous run, still sitting in the old
    `_mode["fix_duration"]`. Nothing raised.
    """
    utils = build_fake_f5(monkeypatch)
    module = pytest.importorskip("colab.indicf5_diagnose")

    monkeypatch.setattr(utils, "_diagnose_installed", False, raising=False)
    monkeypatch.setattr(utils, "_diagnose_token", None, raising=False)
    module.reset_mode()
    module.install_patches()

    # Stand in for what `fresh()` leaves behind: a different module object with
    # its own _mode, against a utils_infer still holding the old wrappers.
    stale_mode, stale_calls = module._mode, module._calls
    monkeypatch.setattr(module, "_mode", dict(stale_mode))
    monkeypatch.setattr(module, "_calls", [])
    assert module._mode is not stale_mode

    module.install_patches()
    assert utils._diagnose_token is module._mode, "patch still reads the old dict"

    module._mode["one_chunk"] = True
    module._mode["fix_duration"] = 14.32
    run(utils, utils.infer_process, LONG_HINDI)

    assert module._calls, "the new module's _calls never filled"
    assert stale_calls == [], "the stale module's _calls filled instead"
    assert module._calls[-1]["fix_duration"] == 14.32
    assert module._calls[-1]["chunks"] == 1


def test_rebinding_does_not_stack_wrappers(monkeypatch):
    """
    Each rebind must replace the wrapper, not wrap it. A stacked wrapper would
    record every call twice and halve every rate derived from the count.
    """
    utils = build_fake_f5(monkeypatch)
    module = pytest.importorskip("colab.indicf5_diagnose")

    monkeypatch.setattr(utils, "_diagnose_installed", False, raising=False)
    monkeypatch.setattr(utils, "_diagnose_token", None, raising=False)
    module.reset_mode()
    module.install_patches()
    real = {name: getattr(getattr(utils, name), module.WRAPPER_MARK)
            for name in module.PATCHED}

    for _ in range(3):
        monkeypatch.setattr(module, "_mode", dict(module._mode))
        monkeypatch.setattr(module, "_calls", [])
        module.install_patches()
        assert {name: getattr(getattr(utils, name), module.WRAPPER_MARK)
                for name in module.PATCHED} == real

    module._mode["one_chunk"] = True
    run(utils, utils.infer_process, HINDI)
    assert len(module._calls) == 1, "one call recorded more than once"


def test_reset_mode_clears_a_duration_left_by_the_previous_run(monkeypatch):
    """
    `_mode` is module-level and persists between calls. A probe that sets
    fix_duration and returns leaves it set, and the next caller that does not
    set it inherits the previous run's LAST duration rather than the byte
    formula — which is not a state anything downstream can detect.
    """
    utils = build_fake_f5(monkeypatch)
    module = pytest.importorskip("colab.indicf5_diagnose")

    monkeypatch.setattr(utils, "_diagnose_installed", False, raising=False)
    monkeypatch.setattr(utils, "_diagnose_token", None, raising=False)
    module.install_patches()

    module._mode.update({"fix_duration": 14.32, "one_chunk": True})
    module._calls.append({"stale": True})

    module.reset_mode()
    assert module._mode == {"speed": None, "one_chunk": False,
                            "fix_duration": None}
    assert module._calls == []

    run(utils, utils.infer_process, HINDI)
    assert module._calls[-1]["fix_duration"] is None


def test_both_probes_reset_before_they_load_the_model():
    """
    Reset has to happen before the GPU is spent, not after the run, or a stale
    duration is only discovered once the clips exist.
    """
    for name in ("colab/indicf5_tts_probe.py", "colab/indicf5_hindi_probe.py"):
        source = __import__("pathlib").Path(name).read_text(encoding="utf-8")
        assert "reset_mode()" in source, name
        assert source.index("reset_mode()") < source.index("load_indicf5()"), name


def test_an_unmarked_wrapper_is_refused_rather_than_wrapped(monkeypatch):
    """
    The migration case, which cost a second run on 2026-09-19.

    The fix for the first failure recorded the originals in a table on
    utils_infer. A kernel that still held wrappers from the version BEFORE that
    fix had no such table, so the new install read the old wrapper as if it
    were the genuine function and wrapped it. `inspect.signature` of the old
    wrapper is `(*args, **kwargs)`, so `bound.arguments` had no `ref_audio` and
    every one of 24 clips raised `KeyError: 'ref_audio'`.

    A side table belongs to the module version that wrote it. The mark has to
    live on the wrapper, and a wrapper without one cannot be unwrapped at all —
    the function it replaced is in a closure cell nothing recorded — so the
    only correct answer is to refuse and say what to do.
    """
    utils = build_fake_f5(monkeypatch)
    module = pytest.importorskip("colab.indicf5_diagnose")

    genuine = utils.infer_batch_process

    def install_patches():
        """Stand in for the old install_patches, whose wrappers are nested in it."""
        def infer_batch_process(*args, **kwargs):
            return genuine(*args, **kwargs)
        return infer_batch_process

    unmarked = install_patches()
    # exactly what an old wrapper looks like: our module, nested in that name
    unmarked.__module__ = module.__name__
    unmarked.__qualname__ = "install_patches.<locals>.infer_batch_process"
    utils.infer_batch_process = unmarked

    monkeypatch.setattr(utils, "_diagnose_token", None, raising=False)
    with pytest.raises(RuntimeError, match="RESTART THE RUNTIME"):
        module.install_patches()

    # and it must not have half-installed on the way out
    assert utils.infer_batch_process is unmarked


def test_a_marked_wrapper_is_unwrapped_to_the_real_function(monkeypatch):
    """The same situation, once the mark exists: rebind, do not refuse."""
    utils = build_fake_f5(monkeypatch)
    module = pytest.importorskip("colab.indicf5_diagnose")

    genuine = {name: getattr(utils, name) for name in module.PATCHED}
    monkeypatch.setattr(utils, "_diagnose_token", None, raising=False)
    module.reset_mode()
    module.install_patches()

    for name in module.PATCHED:
        assert getattr(getattr(utils, name), module.WRAPPER_MARK) is genuine[name]

    # a second module instance rebinds straight onto the real functions
    monkeypatch.setattr(module, "_mode", dict(module._mode))
    monkeypatch.setattr(module, "_calls", [])
    module.install_patches()
    for name in module.PATCHED:
        assert getattr(getattr(utils, name), module.WRAPPER_MARK) is genuine[name]


def test_the_signature_bound_is_the_real_one(monkeypatch):
    """
    The actual symptom, pinned. A wrapper wrapping a wrapper binds
    `(*args, **kwargs)` and every named argument disappears.
    """
    utils = build_fake_f5(monkeypatch)
    module = pytest.importorskip("colab.indicf5_diagnose")

    monkeypatch.setattr(utils, "_diagnose_token", None, raising=False)
    module.reset_mode()
    module.install_patches()

    for _ in range(3):
        monkeypatch.setattr(module, "_mode", dict(module._mode))
        monkeypatch.setattr(module, "_calls", [])
        module.install_patches()

    module._mode["one_chunk"] = True
    module._mode["fix_duration"] = 14.32
    run(utils, utils.infer_process, HINDI)       # KeyError: 'ref_audio' if stacked
    assert module._calls[-1]["fix_duration"] == 14.32
