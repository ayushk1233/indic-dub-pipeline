"""
Everything the GPU worker needs, and nothing else.

`colab/indicf5_worker.py` used to import from four research modules, one of
which reached the the earlier baseline baseline worker. That made the smallest thing anybody
has to run — synthesize a bundle — depend on the largest thing in the tree.
This module is those pieces, lifted verbatim so the behaviour is identical:

    plain()            the reference-transcript normaliser
    load_indicf5()     the model loader, with its meta-device workaround
    install_patches()  the fix_duration / one_chunk instrumentation
    reset_mode()       clearing state a previous run left behind

The research modules still exist and still work; they simply are not on the
path a dubbing run takes.
"""

import inspect
import re

import torch


_lines = []


def p(text=""):
    print(text)
    _lines.append(text)


# -- reference transcript normalisation ---------------------------------------

_PUNCT = re.compile(r"[.,!?;:\u0964\u0965\"'()\[\]{}\u2014\u2013-]")


def plain(text):
    """The same words with the punctuation and capitalisation taken off."""
    return " ".join(_PUNCT.sub(" ", (text or "").lower()).split())


# -- loading the model --------------------------------------------------------

INDICF5_REPO = "ai4bharat/IndicF5"

def _assert_materialized(model):
    """
    Refuse a model whose weights were never actually allocated.

    A meta-weighted model does not raise on its own: it synthesizes noise, and
    noise scored against a speaker anchor looks exactly like a model that
    clones badly. That failure would be written down as a result rather than
    as a bug, so it is made loud here.
    """
    meta = [name for name, tensor in model.named_parameters() if tensor.is_meta]
    meta += [name for name, tensor in model.named_buffers() if tensor.is_meta]

    if meta:
        raise RuntimeError(
            f"{len(meta)} parameters are still on the meta device "
            f"(first: {meta[0]}); the weights were never materialized"
        )

    return model


def _load_direct():
    """
    Build IndicF5's remote class directly, outside transformers' meta context.

    `AutoModel.from_pretrained` runs the remote `__init__` under an
    empty-weights context, so every parameter it creates lands on the meta
    device. IndicF5 builds its Vocos vocoder inside that `__init__` and calls
    `.to(device)` on it there, which raises:

        NotImplementedError: Cannot copy out of meta tensor; no data!

    `low_cpu_mem_usage=False` does not help, because the exception comes from
    inside `__init__` rather than from the weight-loading that flag controls.

    Instantiating the class ourselves skips that context entirely: the module
    tree and the vocoder are built with real storage, and the checkpoint is
    then loaded on top. Key mismatches are reported rather than swallowed,
    because `strict=False` silently tolerating a renamed prefix would leave a
    randomly-initialized model that runs perfectly well and sounds wrong.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = AutoConfig.from_pretrained(INDICF5_REPO, trust_remote_code=True)

    auto_map = getattr(config, "auto_map", None) or {}
    reference = auto_map.get("AutoModel")

    if not reference:
        raise RuntimeError(f"config has no auto_map['AutoModel']: {auto_map}")

    model_class = get_class_from_dynamic_module(reference, INDICF5_REPO)
    model = model_class(config)

    state = load_file(hf_hub_download(INDICF5_REPO, "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)

    total = sum(1 for _ in model.named_parameters())

    if missing or unexpected:
        p(f"  checkpoint: {len(missing)} missing of {total} parameters, "
          f"{len(unexpected)} unexpected")
        if missing:
            p(f"    first missing: {missing[0]}")
        if unexpected:
            p(f"    first unexpected: {unexpected[0]}")

    # Vocos is fetched by the remote __init__ from its own repo, so some of
    # this model's parameters are legitimately absent from this checkpoint and
    # a few missing keys are expected. What is not survivable is the checkpoint
    # belonging to a different model: unexpected keys mean the names do not
    # line up, and a majority of parameters missing means most of the network
    # is still at its random initialization. Either way it would run fine and
    # sound wrong, which is the failure worth refusing.
    if unexpected or (total and len(missing) > total / 2):
        raise RuntimeError(
            f"checkpoint does not match the model: {len(missing)}/{total} "
            f"missing, {len(unexpected)} unexpected"
        )

    return model


def load_indicf5():
    """
    Load IndicF5 with real weights, by whichever route works.

    The direct route is tried first because it is the one that survives
    transformers' meta-device initialization. `from_pretrained` stays as a
    fallback for the case where a future version stops needing the workaround,
    and both are checked for meta tensors before being returned.
    """
    attempts = []

    for name, build in (("direct", _load_direct),
                        ("from_pretrained", lambda: __import__(
                            "transformers", fromlist=["AutoModel"]
                        ).AutoModel.from_pretrained(
                            INDICF5_REPO, trust_remote_code=True))):
        try:
            model = _assert_materialized(build())
            p(f"  loaded via {name}")
            break
        except Exception as exc:
            attempts.append(f"{name}: {type(exc).__name__}: {exc}")
            model = None

    if model is None:
        raise RuntimeError(" | ".join(attempts))

    if torch.cuda.is_available():
        model = model.to("cuda")

    return _assert_materialized(model)




# IndicF5 generates at 24 kHz through Vocos, and its mel hop is 256 samples —
# so one mel frame is 1/93.75 of a second. `fix_duration` is counted in frames,
# which is why every duration in this module converts through FRAME_RATE.
SAMPLE_RATE = 24000
HOP_LENGTH = 256
FRAME_RATE = SAMPLE_RATE / HOP_LENGTH

# The measured Hindi speaking rate. Taken from the pipeline's own table rather
# than kept as a second copy — the research module had its own literal and the
# two agreed at 10.81, which is exactly the kind of agreement that stops being
# true silently.
from src.eval.translation_metrics import NATURAL_CPS as _NATURAL_CPS

NATURAL_CPS_HI = _NATURAL_CPS["hi"]


# -- fix_duration instrumentation --------------------------------------------

_calls = []
_pending = {}

# fix_duration is injected here rather than passed to the model, for the same
# reason speed is: IndicF5's remote __call__ decides which of infer_process's
# arguments it forwards, and a keyword it drops on the floor fails silently —
# the duration simply reverts to the byte formula and the run looks like a
# model result. Setting it inside the wrapper puts it in the arguments that
# infer_batch_process is actually called with, and _calls records what the
# sampler was then given, so the two can be checked against each other.
_mode = {"speed": None, "one_chunk": False, "fix_duration": None}

_lines = []


PATCHED = ("chunk_text", "infer_batch_process")

# Stamped on every wrapper this module installs, holding the function it
# replaced. Read from the function object rather than from a table on
# utils_infer, because a table only exists if the module version that wrote it
# is the one that installed — and the version before this one wrote none.
WRAPPER_MARK = "_diagnose_wrapper_of"


def _wrapped_by_us(function):
    """True when `function` is a wrapper this module installed and marked."""
    return function is not None and hasattr(function, WRAPPER_MARK)


def _looks_like_an_unmarked_wrapper(function):
    """
    A wrapper from a version of this module that predates the mark.

    It cannot be unwrapped: the function it replaced is in a closure cell and
    nothing recorded it. Wrapping it again is what produced `KeyError:
    'ref_audio'` on 24 clips — the new wrapper took `inspect.signature` of the
    old wrapper, whose signature is `(*args, **kwargs)`, so `bound.arguments`
    had no `ref_audio` in it.
    """
    return (getattr(function, "__module__", None) == __name__
            and "install_patches.<locals>" in getattr(function, "__qualname__", ""))


def _unwrap(utils_infer, name):
    """The genuine utils_infer function behind `name`, whatever is there now.

    Per name rather than all-or-nothing, because the two can disagree: a caller
    may have replaced one of them while our wrapper is still on the other, and
    taking a wrapper for an original records every call twice.
    """
    current = getattr(utils_infer, name)
    if _wrapped_by_us(current):
        return getattr(current, WRAPPER_MARK)
    if _looks_like_an_unmarked_wrapper(current):
        raise RuntimeError(
            f"utils_infer.{name} is an instrumentation wrapper from an older "
            "version of colab/indicf5_diagnose, and the function it replaced "
            "was never recorded, so it cannot be unwrapped. Wrapping it again "
            "breaks every call. RESTART THE RUNTIME and run again — a git "
            "pull plus colab.reimport.fresh() cannot repair this, because "
            "f5_tts is not a package fresh() purges. Costs a model reload."
        )
    return current


def install_patches():
    """
    Instrument utils_infer in place.

    Idempotent, so a second import or a re-run inside the same kernel does not
    stack wrappers — but idempotent **per module instance**, which is the part
    that was wrong and cost a run.

    The wrappers close over this module's `_mode` and `_calls`. `f5_tts` is not
    ours and is not purged by colab/reimport.py's `fresh()`, so after a
    re-import the wrappers on utils_infer still point at the PREVIOUS
    `colab.indicf5_diagnose`'s dictionaries while every caller writes to the
    new ones. The old guard was a plain boolean on utils_infer, so
    install_patches() returned early and never rebound them.

    Measured 2026-09-19. A run after `fresh()` in a live kernel produced 24
    clips at exactly 3.82s each, whatever the text: 3.82s was the target of the
    LAST clip of the previous run, still sitting in the previous module's
    `_mode["fix_duration"]`. `one_chunk` was stale the same way, `_calls` never
    filled, and every `requested_s` came back nan. The report's instrumentation
    guard would have caught it — that is what it is for — but nothing stopped
    the GPU being spent first.

    The token is this module's `_mode` object itself, so the check is exactly
    "are the installed wrappers reading the dictionaries this module is
    writing", which is the real question. Originals are cached on utils_infer
    so rebinding replaces the wrappers rather than wrapping them again.
    """
    from f5_tts.infer import utils_infer

    bound_to_us = all(_wrapped_by_us(getattr(utils_infer, name, None))
                      for name in PATCHED)
    if bound_to_us and getattr(utils_infer, "_diagnose_token", None) is _mode:
        return

    original_chunk = _unwrap(utils_infer, "chunk_text")
    original_batch = _unwrap(utils_infer, "infer_batch_process")

    batch_signature = inspect.signature(original_batch)

    def chunk_text(text, max_chars=135):
        _pending["max_chars"] = max_chars
        if _mode["one_chunk"]:
            return [text]
        return original_chunk(text, max_chars=max_chars)

    def infer_batch_process(*args, **kwargs):
        bound = batch_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments

        audio, sr = arguments["ref_audio"]
        ref_seconds = audio.shape[-1] / sr

        # Replicating the one-space append infer_batch_process is about to do,
        # so the byte count recorded here is the byte count it will divide by.
        ref_text = arguments["ref_text"]
        if ref_text and len(ref_text[-1].encode("utf-8")) == 1:
            ref_text = ref_text + " "
        ref_bytes = len(ref_text.encode("utf-8"))

        batches = list(arguments["gen_text_batches"])
        gen_bytes = sum(len(b.encode("utf-8")) for b in batches)
        gen_chars = sum(len(b) for b in batches)

        allocated = ref_seconds / ref_bytes * gen_bytes if ref_bytes else float("nan")
        wanted = gen_chars / NATURAL_CPS_HI if gen_chars else float("nan")

        # A total, reference included — read from
        # utils_infer: `duration = int(fix_duration * sr / hop)`. It replaces
        # the byte-ratio allocation rather than adjusting it, so speed stops
        # mattering on any call that sets it.
        if _mode["fix_duration"] is not None:
            if "fix_duration" not in arguments:
                raise TypeError(
                    "infer_batch_process takes no fix_duration on this install; "
                    "injecting it would do nothing and the byte formula would "
                    "silently decide every duration instead"
                )
            arguments["fix_duration"] = float(_mode["fix_duration"])

        if _mode["speed"] == "auto":
            # Divide out exactly the over-allocation measured on this call.
            # Nothing here assumes a script or a bytes-per-character constant:
            # it is the time the formula asked for over the time the text needs
            # at the Hindi rate measured from FLEURS.
            arguments["speed"] = allocated / wanted if wanted else 1.0
        elif _mode["speed"] is not None:
            arguments["speed"] = float(_mode["speed"])

        model = arguments["model_obj"]
        original_sample = model.sample
        sample_signature = inspect.signature(original_sample)
        frames = []

        def sample(*sample_args, **sample_kwargs):
            inner = sample_signature.bind(*sample_args, **sample_kwargs)
            inner.apply_defaults()
            frames.append(int(inner.arguments["duration"]))
            return original_sample(*sample_args, **sample_kwargs)

        model.sample = sample
        try:
            result = original_batch(**arguments)
        finally:
            try:
                del model.sample
            except AttributeError:
                pass

        ref_frames = int(ref_seconds * SAMPLE_RATE) // HOP_LENGTH
        requested = (sum(frames) - len(frames) * ref_frames) / FRAME_RATE

        _calls.append({
            "ref_s": ref_seconds,
            "ref_bytes": ref_bytes,
            "s_per_byte": ref_seconds / ref_bytes if ref_bytes else float("nan"),
            "max_chars": _pending.get("max_chars"),
            "chunks": len(batches),
            "gen_bytes": gen_bytes,
            "gen_chars": gen_chars,
            "speed": arguments.get("speed"),
            "fix_duration": arguments.get("fix_duration"),
            "formula_s": allocated,
            "requested_s": requested,
            "natural_s": wanted,
        })
        return result

    # The mark goes ON the wrapper and carries the function it replaced, so a
    # later install can unwrap it without consulting anything this module
    # instance wrote down. That is the whole point: side tables belong to the
    # module version that wrote them, and the version before this one wrote
    # none.
    setattr(chunk_text, WRAPPER_MARK, original_chunk)
    setattr(infer_batch_process, WRAPPER_MARK, original_batch)

    utils_infer.chunk_text = chunk_text
    utils_infer.infer_batch_process = infer_batch_process
    # The token is the dictionary the wrappers read, not a boolean. See above.
    utils_infer._diagnose_token = _mode
    utils_infer._diagnose_installed = True


def reset_mode():
    """
    Put the injection state back to "inject nothing".

    `_mode` is module-level and persists between calls, so a probe that sets
    fix_duration and then returns leaves it set. The next caller that forgets
    to set it inherits the previous run's last duration — which is exactly the
    number that came back 24 times on 2026-09-19. Probes call this before they
    start rather than relying on every path setting every key.
    """
    _mode.update({"speed": None, "one_chunk": False, "fix_duration": None})
    _calls.clear()
    _pending.clear()

