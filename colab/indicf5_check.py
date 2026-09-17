"""
XTTS-v2 against IndicF5, on the same scale, for the case that ships.

The four-arm run settled the diagnosis: identity survives cross-lingual
conditioning intact, and what costs XTTS its similarity is the language it is
asked to *speak*. Hindi came out at 78% of the achievable scale and sounded
Indian and unrobotic; English came out at 47% and heavily American.

That leaves one question worth spending GPU time on. XTTS-v2's weights are
CPML, Coqui is gone, and nobody can now grant commercial terms — so whatever
XTTS scores, it cannot be the thing that ships. IndicF5 is MIT, 0.4B
parameters, trained on 1417 hours of Indian speech, and clones from a
reference clip plus its transcript. The only number that matters is where it
lands on the scale the four-arm run calibrated.

Everything is held identical between the two models: the same reference clips,
the same seven Hindi sentences, the same speaker encoder doing the scoring,
the same anchors and the same floor. XTTS's own encoder scores both, which is
not a handicap for IndicF5 — the encoder never saw either model's training
data and has no stake in which one wins.

Seven sentences per arm rather than three. The four-arm run showed a
within-arm spread of 0.02 to 0.08 across three sentences, and a model
difference worth acting on could be 0.05, so three samples could not have
resolved it. The standard error is reported so a difference can be read
against its own noise instead of eyeballed.

    import colab.indicf5_check as check
    rows = check.main()
    check.listen(rows)
"""

import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from colab.english_report import cosine


REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "fixtures"
OUT = Path("/content/indicf5_check")
REPORT = Path("/content/indicf5_report.txt")

INDICF5_REPO = "ai4bharat/IndicF5"
SAMPLE_RATE = 24000

# Identical to colab/four_arm.py, so the XTTS arm here is directly comparable
# to the numbers already recorded rather than a fresh measurement of its own.
CONDITIONING = {
    "gpt_cond_len": 30,
    "gpt_cond_chunk_len": 4,
    "max_ref_length": 30,
    "sound_norm_refs": False,
}

ANCHOR = {
    "gpt_cond_len": 60,
    "gpt_cond_chunk_len": 30,
    "max_ref_length": 60,
    "sound_norm_refs": False,
}

# Greedy. It beat sampling on three of the four arms and tied on the fourth,
# and it is what makes XTTS's output duration reproducible at all.
XTTS_DECODE = {
    "do_sample": False,
    "repetition_penalty": 5.0,
    "enable_text_splitting": False,
}

# IndicF5 is a flow-matching model and samples fresh noise every call, so a run
# is not reproducible without this. Fixed rather than averaged over seeds:
# the question is where the model lands, not how much it wobbles.
SEED = 0

NATURAL_CPS_HI = 10.81

# Transcribing the output back is the only check that sees content. Speaker
# similarity reads timbre, so gibberish in the right voice scores *higher*
# than clean speech in a slightly wrong one: the en_ref_25s arm placed at 97%
# of scale, the best of any arm, while audibly speaking nonsense between the
# intended words. Nothing in an identity metric can catch that.
#
# Through transformers rather than faster-whisper on purpose. faster-whisper
# needs ctranslate2, and this environment already required pinning numpy into
# a one-minor-version window to keep transformers, numba and f5-tts from
# breaking each other. Whisper via transformers adds no new dependency at all.
ASR_MODEL = "openai/whisper-large-v3-turbo"

# What counts as broken rather than imperfect, as a fraction of the intended
# length. A single error rate cannot do this job: gibberish appended to an
# otherwise correct sentence scores about 0.31, which sits under any threshold
# loose enough to tolerate Whisper's own Hindi error. Insertions and deletions
# are therefore judged separately, which is also what distinguishes the two
# failures — extra speech the model invented, against speech it never finished.
MAX_EXTRA = 0.15
MAX_MISSING = 0.25
MAX_CER = 0.35

# Above this many characters per second the transcript cannot describe the clip
# it came from. Natural speech runs 10.8 cps in Hindi and 13.1 in English, and
# the fastest thing this project has ever synthesized was 22.9. Forty is not a
# quality threshold, it is an impossibility threshold: Whisper loops on audio it
# cannot parse, and a 3.97s clip that transcribes to 2074 characters is 522 cps.
# Scoring that as content would report a 48 second prefix on a four second clip.
LOOP_CPS = 40.0

# How many pieces to cut each real take into when measuring the ceiling.
# A 50 to 60 second take gives roughly 18s, 10s, 6s and 4s pieces, which
# brackets both the test sentences and real dubbing segments.
SPLITS = (3, 6, 10, 14)

_SENTENCE_RE = re.compile(r"(?<=[.!?।])\s+")

LINES: list[str] = []


def p(*args):
    line = " ".join(str(a) for a in args)
    LINES.append(line)
    print(line)


def section(title):
    p("\n" + "=" * 76)
    p(title)
    p("=" * 76)


def sentences(text, minimum=25):
    return [s.strip() for s in _SENTENCE_RE.split(text or "") if len(s.strip()) >= minimum]


_PUNCT = re.compile(r"[.,!?;:\u0964\u0965\"'()\[\]{}—–-]")


def normalize(text):
    """
    Strip what a transcript comparison should not be graded on.

    Punctuation, case and whitespace differences are not the model saying the
    wrong thing. Devanagari danda counts as punctuation.
    """
    return " ".join(_PUNCT.sub(" ", (text or "").lower()).split())


def align(reference, hypothesis):
    """
    Levenshtein alignment returning substitutions, insertions and deletions.

    The counts matter more than the distance here. Gibberish spoken between
    the intended words is an insertion; a sentence cut off early is a
    deletion; a mispronounced word is a substitution. A single error rate adds
    them together and hides which one happened, and they have different
    causes: insertions point at reference text the model had no audio for,
    deletions at a duration estimate that ran out of room.
    """
    n, m = len(reference), len(hypothesis)

    # cost, then (substitutions, insertions, deletions) carried alongside it.
    row = [(j, (0, j, 0)) for j in range(m + 1)]

    for i in range(1, n + 1):
        previous, row = row, [(i, (0, 0, i))]

        for j in range(1, m + 1):
            same = reference[i - 1] == hypothesis[j - 1]

            sub_cost, sub_counts = previous[j - 1]
            if not same:
                sub_cost += 1
                s, ins, dels = sub_counts
                sub_counts = (s + 1, ins, dels)

            del_cost, del_counts = previous[j]
            del_cost += 1
            s, ins, dels = del_counts
            del_counts = (s, ins, dels + 1)

            ins_cost, ins_counts = row[j - 1]
            ins_cost += 1
            s, ins, dels = ins_counts
            ins_counts = (s, ins + 1, dels)

            row.append(min((sub_cost, sub_counts),
                           (del_cost, del_counts),
                           (ins_cost, ins_counts),
                           key=lambda pair: pair[0]))

    return row[m][1]


def score_text(intended, heard):
    """
    How far the spoken content is from the text that was asked for.

    Returns character error rate together with the insertion and deletion
    rates it is made of, each relative to the intended length.
    """
    reference, hypothesis = normalize(intended), normalize(heard)

    if not reference:
        return {"cer": float("nan"), "extra": float("nan"),
                "missing": float("nan")}

    subs, insertions, deletions = align(reference, hypothesis)
    length = len(reference)

    return {
        "cer": (subs + insertions + deletions) / length,
        "extra": insertions / length,
        "missing": deletions / length,
        "lead": leading_extra(reference, hypothesis) / length,
        "lead_chars": leading_extra(reference, hypothesis),
    }


def leading_extra(reference, hypothesis):
    """
    How many characters the model speaks before the sentence it was asked for
    begins.

    A total insertion rate cannot see this. Three seconds of invented speech at
    the front of a twelve second clip scores 0.12 — under any threshold loose
    enough to tolerate Whisper's own Hindi error — while thirteen characters
    scattered through the same sentence would be genuinely harmless. The
    failure is contiguous and positional, so it has to be measured that way.

    The alignment is the usual Levenshtein one with a free start: beginning
    anywhere in the transcript costs nothing, so the cheapest reading of a clip
    that opens with junk is to skip the junk, and the length of that skip is
    the answer. The end is not free, because speech that runs on past the
    sentence is a real insertion and is already counted as one.
    """
    n, m = len(reference), len(hypothesis)
    if not n:
        return 0

    row = [(0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        previous, row = row, [(i, 0)]
        for j in range(1, m + 1):
            sub_cost, sub_start = previous[j - 1]
            if reference[i - 1] != hypothesis[j - 1]:
                sub_cost += 1
            del_cost, del_start = previous[j]
            ins_cost, ins_start = row[j - 1]
            row.append(min((sub_cost, sub_start),
                           (del_cost + 1, del_start),
                           (ins_cost + 1, ins_start)))

    return row[m][1]


def transcript_impossible(heard, path):
    """
    True when the transcript is longer than the clip could physically contain.

    Whisper repeats itself on audio it cannot parse, and the repetition is
    fluent, so nothing downstream notices. This is the guard: no clip holds
    more characters than LOOP_CPS times its own duration.
    """
    if not heard:
        return False
    try:
        info = sf.info(str(path))
        duration = info.frames / info.samplerate
    except Exception:
        return False
    return bool(duration > 0 and len(heard) / duration > LOOP_CPS)


def transcribe_outputs(rows):
    """
    Read every generated clip back and score it against the text asked for.

    Loaded after synthesis so it never competes with the two TTS models for
    GPU memory.
    """
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    asr = pipeline(
        "automatic-speech-recognition",
        model=ASR_MODEL,
        device=device,
        torch_dtype=torch.float16 if device == 0 else torch.float32,
    )

    for row in rows:
        if not row.get("text"):
            continue
        try:
            out = asr(str(row["path"]),
                      generate_kwargs={"language": row.get("language", "hi"),
                                       "task": "transcribe"})
            row["heard"] = (out or {}).get("text", "").strip()
            row["asr_looped"] = transcript_impossible(row["heard"], row["path"])
            if row["asr_looped"]:
                # Deliberately not scored. A degenerate transcript says the
                # model produced something Whisper could not parse, which is a
                # finding — but every number derived from it would describe
                # Whisper's loop rather than the clip.
                row.update({"cer": float("nan"), "extra": float("nan"),
                            "missing": float("nan"), "lead": float("nan"),
                            "lead_chars": 0})
            else:
                row.update(score_text(row["text"], row["heard"]))
        except Exception as exc:
            row["heard"] = f"<{type(exc).__name__}: {exc}>"
            row.update({"cer": float("nan"), "extra": float("nan"),
                        "missing": float("nan"), "lead": float("nan"),
                        "lead_chars": 0})

    return rows


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


def as_float_wave(audio):
    """
    IndicF5 returns int16 on some paths and float on others. Writing int16
    samples into a float32 wav yields silence-shaped noise at full scale.
    """
    audio = np.asarray(audio).reshape(-1)

    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0

    return audio.astype(np.float32)


def main():
    needed = ["english_speech.wav", "hindi_speech.wav", "english_reference.wav",
              "hindi_reference.wav", "english_reference_short.wav",
              "hindi_reference_short.wav", "scripted_text.json",
              "reference_text.json"]
    missing = [n for n in needed if not (FIXTURES / n).exists()]

    if missing:
        p(f"!! missing fixtures: {missing}")
        p("   Run `python -m scripts.build_fixtures` and "
          "`python -m scripts.transcribe_fixtures` locally, commit, push, "
          "then `git pull` here.")
        return []

    OUT.mkdir(parents=True, exist_ok=True)

    scripted = json.loads((FIXTURES / "scripted_text.json").read_text(encoding="utf-8"))
    ref_text = json.loads((FIXTURES / "reference_text.json").read_text(encoding="utf-8"))
    hindi = sentences(scripted["hi"])

    section("SETUP")
    for module in ("torch", "TTS", "transformers", "librosa", "soundfile"):
        try:
            p(f"{module:14s} {getattr(__import__(module), '__version__', '?')}")
        except Exception as exc:
            p(f"{module:14s} MISSING ({type(exc).__name__})")
    p(f"cuda           {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
    p(f"\n{len(hindi)} Hindi sentences per arm, 2 references, 2 models")

    # ------------------------------------------------------- the scoring rig
    from colab.xtts_worker import XTTSWorker

    worker = XTTSWorker(Path("/content/unused_bundle"))
    worker.load_model()
    xtts = worker.xtts

    def latents(path, **params):
        with torch.no_grad():
            return xtts.get_conditioning_latents(audio_path=[str(path)], **params)

    def embed(path):
        try:
            _, embedding = latents(path)
            return embedding
        except Exception:
            return None

    _, anchor_en = latents(FIXTURES / "english_speech.wav", **ANCHOR)
    _, anchor_hi = latents(FIXTURES / "hindi_speech.wav", **ANCHOR)

    section("CALIBRATION — a ceiling per clip length, not one ceiling")

    def pieces(path, count, label, anchor):
        """
        Cut a real take into `count` equal pieces and score each against the
        anchor. No synthesis: this is the speaker against himself.
        """
        audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        edges = np.linspace(0, audio.size, count + 1).astype(int)
        scores, durations = [], []

        for index in range(count):
            piece = OUT / f"{label}_{count}_{index}.wav"
            sf.write(str(piece), audio[edges[index]:edges[index + 1]], rate,
                     subtype="PCM_16")
            embedding = embed(piece)
            if embedding is not None:
                scores.append(cosine(anchor, embedding))
                durations.append((edges[index + 1] - edges[index]) / rate)

        if not scores:
            return None

        return float(np.mean(durations)), float(np.mean(scores))

    def curve(path, label, anchor):
        """
        The ceiling as a function of how much audio is being judged.

        A speaker embedding taken from five seconds is a noisier estimate than
        one taken from twenty, and it regresses toward the population mean, so
        a short clip scores lower against the same anchor even when it is the
        same person on the same tape. Measured across all 14 XTTS clips in the
        previous run, clip duration correlated with similarity at r = +0.82,
        and the short clips scored 0.11 below the long ones.

        That matters because real dubbing segments are short. The four-arm run
        used the three longest sentences and scored them against a ceiling
        built from 17-second thirds, which flattered the result: the same model
        on production-length segments sits well below it. One ceiling per
        length bucket fixes the mismatch.
        """
        points = [p for p in (pieces(path, n, label, anchor) for n in SPLITS)
                  if p is not None]
        return sorted(points)

    same_curve = curve(FIXTURES / "english_speech.wav", "en_piece", anchor_en)
    cross_curve = curve(FIXTURES / "hindi_speech.wav", "hi_piece", anchor_en)

    floor_scores = []
    try:
        for _, entry in xtts.speaker_manager.speakers.items():
            floor_scores.append(cosine(anchor_en, entry["speaker_embedding"]))
    except Exception as exc:
        p(f"  floor unavailable: {type(exc).__name__}: {exc}")

    floor = float(np.median(floor_scores)) if floor_scores else 0.0

    p(f"  floor  {floor:.3f}   ({len(floor_scores)} real strangers)")
    p("")
    p(f"  {'clip length':>13}{'same-language':>16}{'cross-language':>16}")
    for (d_same, c_same), (d_cross, c_cross) in zip(same_curve, cross_curve):
        p(f"  {d_same:>11.1f}s{c_same:>16.3f}{c_cross:>16.3f}")
    p("")
    p("  You against yourself, no synthesis. Shorter clips score lower because")
    p("  the embedding is a noisier estimate, not because the voice changed.")

    def ceiling_for(duration, points):
        """Ceiling measured at whichever clip length is closest to this one."""
        if not points:
            return float("nan")
        return min(points, key=lambda point: abs(point[0] - duration))[1]

    def position(score, duration, points):
        ceiling = ceiling_for(duration, points)
        span = ceiling - floor
        if not np.isfinite(score) or not np.isfinite(span) or span <= 0:
            return float("nan")
        return 100.0 * (score - floor) / span

    # --------------------------------------------------------------- the arms
    # The English reference speaking Hindi is scored against the cross-language
    # curve; the Hindi reference against the same-language one. Each clip is
    # then placed against the point on that curve nearest its own duration.
    #
    # The two models get different reference lengths on purpose. XTTS clones
    # timbre from whatever it is handed and its output duration does not depend
    # on the reference at all. IndicF5 scales generated length by the ratio of
    # generated to reference transcript bytes, times the reference audio's
    # duration — and it clips the reference internally. Handed the 25s clips it
    # used only 12 to 14 seconds of them while still using the whole transcript,
    # so it believed the speaker talks twice as fast as he does and returned
    # every Hindi sentence at 0.48x the duration natural Hindi needs, with a
    # standard deviation of 0.0 across seven sentences. The 10s references sit
    # under that clipping threshold. The 25s arm is kept as a control so the
    # bug and its fix are visible in the same run rather than asserted.
    xtts_arms = [
        ("en_ref", "english_reference.wav", "english", anchor_en, cross_curve),
        ("hi_ref", "hindi_reference.wav", "hindi", anchor_hi, same_curve),
    ]

    indicf5_arms = [
        ("en_ref_10s", "english_reference_short.wav", "english_short",
         anchor_en, cross_curve),
        ("hi_ref_10s", "hindi_reference_short.wav", "hindi_short",
         anchor_hi, same_curve),
        ("en_ref_25s", "english_reference.wav", "english",
         anchor_en, cross_curve),
    ]


    rows = []

    def record(model, arm, index, sentence, wav, anchor, points, elapsed):
        directory = OUT / model / arm
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{index:02d}.wav"
        sf.write(str(path), wav, SAMPLE_RATE, subtype="PCM_16")

        embedding = embed(path)
        score = cosine(anchor, embedding) if embedding is not None else float("nan")
        duration = wav.size / SAMPLE_RATE

        rows.append({
            "model": model, "arm": arm, "index": index, "text": sentence,
            "path": path, "dur": duration, "sim": score,
            "ceiling": ceiling_for(duration, points),
            "position": position(score, duration, points),
            "peak": float(np.abs(wav).max()) if wav.size else 0.0,
            "cps": len(sentence) / duration if duration else 0.0,
            "gen_s": elapsed,
        })
        p(f"    [{index}] {duration:6.2f}s  {rows[-1]['cps']:5.1f} cps  "
          f"sim {score:.3f}  vs {rows[-1]['ceiling']:.3f}  "
          f"{rows[-1]['position']:4.0f}% of scale")

    section("XTTS-v2   (CPML, non-commercial — the baseline, not a candidate)")

    for arm, filename, _, anchor, points in xtts_arms:
        latent, embedding = latents(FIXTURES / filename, **CONDITIONING)
        p(f"\n--- xtts {arm} -> hi")

        for index, sentence in enumerate(hindi):
            started = time.perf_counter()
            try:
                with torch.no_grad():
                    result = xtts.inference(
                        text=sentence, language="hi",
                        gpt_cond_latent=latent, speaker_embedding=embedding,
                        **XTTS_DECODE,
                    )
            except Exception as exc:
                p(f"    [{index}] FAILED {type(exc).__name__}: {exc}")
                continue

            record("xtts", arm, index, sentence,
                   np.asarray(result["wav"], dtype=np.float32),
                   anchor, points, time.perf_counter() - started)

    section("IndicF5   (MIT — the one that could actually ship)")

    try:
        indicf5 = load_indicf5()
    except Exception as exc:
        p(f"!! IndicF5 did not load: {type(exc).__name__}: {exc}")
        p("   pip install git+https://github.com/ai4bharat/IndicF5.git")
        indicf5 = None

    if indicf5 is not None:
        for arm, filename, key, anchor, points in indicf5_arms:
            transcript = ref_text[key]["text"]
            p(f"\n--- indicf5 {arm} -> hi")
            p(f"    conditioned on {len(transcript)} chars of {key} transcript")

            for index, sentence in enumerate(hindi):
                torch.manual_seed(SEED)
                started = time.perf_counter()
                try:
                    audio = indicf5(
                        sentence,
                        ref_audio_path=str(FIXTURES / filename),
                        ref_text=transcript,
                    )
                except Exception as exc:
                    p(f"    [{index}] FAILED {type(exc).__name__}: {exc}")
                    continue

                record("indicf5", arm, index, sentence, as_float_wave(audio),
                       anchor, points, time.perf_counter() - started)

    # ------------------------------------------------------------------ result
    section("RESULT")
    p(f"{'model':<10}{'arm':<9}{'n':>3}{'sim':>8}{'se':>7}{'position':>10}"
      f"{'se':>6}{'cps':>7}{'natural':>9}{'gen':>7}")
    p("  Position is per clip against the ceiling for that clip's own length,")
    p("  then averaged — not the mean score placed on one ceiling.")
    p("")

    table = {}
    for model, model_arms in (("xtts", xtts_arms), ("indicf5", indicf5_arms)):
        for arm, _, _, _, _ in model_arms:
            items = [r for r in rows if r["model"] == model and r["arm"] == arm]
            scores = [r["sim"] for r in items if np.isfinite(r["sim"])]
            places = [r["position"] for r in items if np.isfinite(r["position"])]
            if not scores:
                continue

            def mean_se(values):
                mean = float(np.mean(values))
                if len(values) < 2:
                    return mean, float("nan")
                return mean, float(np.std(values, ddof=1) / np.sqrt(len(values)))

            mean, se = mean_se(scores)
            place, place_se = mean_se(places) if places else (float("nan"),) * 2
            cps = float(np.mean([r["cps"] for r in items]))
            gen = float(np.mean([r["gen_s"] for r in items]))
            table[(model, arm)] = (mean, se, place, place_se)
            p(f"{model:<10}{arm:<9}{len(scores):>3}{mean:>8.3f}{se:>7.3f}"
              f"{place:>9.0f}%{place_se:>6.0f}"
              f"{cps:>7.1f}{cps / NATURAL_CPS_HI:>8.2f}x{gen:>7.1f}s")

    # --------------------------------------------------------------- content
    # Free IndicF5 first: the ASR model is the third one this process loads.
    indicf5 = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    section("CONTENT — does it say what it was asked to say?")
    p("  Every clip transcribed back and compared to the sentence it was given.")
    p("  This is the only check here that reads content. Speaker similarity")
    p("  reads timbre, and gibberish in the right voice scores higher than")
    p("  clean speech in a slightly wrong one.")
    p("")
    p("    extra    invented speech, as a fraction of the intended length")
    p("    missing  speech never produced — a sentence that stopped early")
    p("")

    try:
        transcribe_outputs(rows)
    except Exception as exc:
        p(f"  !! could not transcribe: {type(exc).__name__}: {exc}")

    p(f"  {'model':<10}{'arm':<12}{'n':>3}{'cer':>7}{'extra':>8}{'missing':>9}"
      f"{'bad':>5}")

    content_table = {}
    for model, model_arms in (("xtts", xtts_arms), ("indicf5", indicf5_arms)):
        for arm, _, _, _, _ in model_arms:
            items = [r for r in rows
                     if r["model"] == model and r["arm"] == arm
                     and np.isfinite(r.get("cer", float("nan")))]
            if not items:
                continue

            mean = {k: float(np.mean([r[k] for r in items]))
                    for k in ("cer", "extra", "missing")}
            bad = [r for r in items
                   if r["extra"] > MAX_EXTRA or r["missing"] > MAX_MISSING
                   or r["cer"] > MAX_CER]
            content_table[(model, arm)] = (mean, len(bad), len(items))

            p(f"  {model:<10}{arm:<12}{len(items):>3}{mean['cer']:>7.3f}"
              f"{mean['extra']:>8.3f}{mean['missing']:>9.3f}"
              f"{len(bad):>4}/{len(items)}")

    worst = sorted((r for r in rows if np.isfinite(r.get("extra", float("nan")))),
                   key=lambda r: -r["extra"])[:3]
    if worst and worst[0]["extra"] > MAX_EXTRA:
        p("")
        p("  Worst inserted speech — what was asked for, then what came out:")
        for row in worst:
            p(f"    {row['model']} {row['arm']} [{row['index']}] "
              f"extra {row['extra']:.2f}")
            p(f"      asked: {row['text'][:70]}")
            p(f"      heard: {row.get('heard', '')[:70]}")

    section("PACE — is the generated audio the right length?")
    p("  IndicF5 sets generated duration from a byte ratio:")
    p("      gen_seconds  =  ref_seconds x gen_bytes / ref_bytes")
    p("  Inverting it recovers how much reference audio the model actually used.")
    p("  If that comes back near the clip's real length, the timing is sound; if")
    p("  it comes back short, the reference was clipped and every duration is")
    p("  scaled by the same wrong factor.")
    p("")
    p(f"  {'model':<10}{'arm':<12}{'ref':>6}{'cps':>7}{'natural':>9}"
      f"{'implied ref':>13}{'sd':>7}{'predicted':>11}")
    p("")
    p("  'predicted' assumes the reference is used whole and the byte ratio holds.")
    p("  Where it matches, the duration behaviour is understood. Where the implied")
    p("  reference comes back shorter than the clip, the model clipped it.")

    ref_seconds = {}
    for name in ("english_reference.wav", "hindi_reference.wav",
                 "english_reference_short.wav", "hindi_reference_short.wav"):
        try:
            ref_seconds[name] = sf.info(str(FIXTURES / name)).duration
        except Exception:
            pass

    for model, model_arms in (("xtts", xtts_arms), ("indicf5", indicf5_arms)):
        for arm, filename, key, _, _ in model_arms:
            items = [r for r in rows if r["model"] == model and r["arm"] == arm]
            if not items:
                continue

            actual = ref_seconds.get(filename, float("nan"))
            ref_bytes = len(ref_text[key]["text"].encode()) if key in ref_text else 0
            implied = [r["dur"] * ref_bytes / len(r["text"].encode())
                       for r in items if ref_bytes and r["text"]]
            cps = float(np.mean([r["cps"] for r in items]))

            if implied and model == "indicf5":
                shown = f"{np.mean(implied):>12.1f}s{np.std(implied):>7.1f}"
                # Whole reference, byte ratio held: what pace should come out.
                wanted = [actual * len(r["text"].encode()) / ref_bytes for r in items]
                natural = [len(r["text"]) / NATURAL_CPS_HI for r in items]
                predicted = f"{np.mean(wanted) / np.mean(natural):>10.2f}x"
            else:
                shown = f"{'-':>13}{'-':>7}"
                predicted = f"{'-':>11}"

            p(f"  {model:<10}{arm:<12}{actual:>5.1f}s{cps:>7.1f}"
              f"{cps / NATURAL_CPS_HI:>8.2f}x{shown}{predicted}")

    p("")
    p("  XTTS is listed for pace only. Its output duration does not depend on")
    p("  the reference, so the implied-length arithmetic does not apply to it.")
    p("")
    p("  The byte ratio assumes the reference and the generated text share a")
    p("  script. They do not: Devanagari is three bytes per character and Latin")
    p("  is one, so an English reference implies 0.075 s/byte where generating")
    p("  Hindi needs 0.035. An English reference therefore overstates duration by")
    p("  about 2.15x, and last run's en_ref only looked correct because clipping")
    p("  25s down to 13.7s divided by 1.83 and very nearly cancelled it.")

    section("LENGTH — does either model hold up on short segments?")
    p("  Real dubbing segments are short. The 14-segment bundle from english.mov")
    p("  averages under 5 seconds, which is the bucket where the embedding is")
    p("  least reliable and, on the previous run, where XTTS lost the most.")
    p("")
    p(f"  {'model':<10}{'bucket':<10}{'n':>3}{'sim':>8}{'ceiling':>9}{'position':>10}")

    for model in ("xtts", "indicf5"):
        for label, keep in (("< 8s", lambda d: d < 8.0), (">= 8s", lambda d: d >= 8.0)):
            items = [r for r in rows
                     if r["model"] == model and keep(r["dur"])
                     and np.isfinite(r["sim"])]
            if not items:
                continue
            p(f"  {model:<10}{label:<10}{len(items):>3}"
              f"{np.mean([r['sim'] for r in items]):>8.3f}"
              f"{np.mean([r['ceiling'] for r in items]):>9.3f}"
              f"{np.mean([r['position'] for r in items]):>9.0f}%")

    quiet = [r for r in rows if r["peak"] < 0.01]
    if quiet:
        p("")
        p(f"  !! {len(quiet)} clips peaked below 0.01 — near silence. Check the")
        p(f"     audio before trusting any score from that model.")

    section("VERDICT")

    production = [table.get(("xtts", "en_ref")),
                  table.get(("indicf5", "en_ref_10s"))]

    if all(production):
        (x_mean, x_se, x_pos, _), (i_mean, i_se, i_pos, _) = production
        gap = i_mean - x_mean
        noise = float(np.hypot(x_se, i_se))

        p(f"  Production case, English reference speaking Hindi:")
        p(f"    XTTS-v2   {x_mean:.3f}  ({x_pos:.0f}% of scale)")
        p(f"    IndicF5   {i_mean:.3f}  ({i_pos:.0f}% of scale)")
        p(f"    difference {gap:+.3f}, combined standard error {noise:.3f}")
        p("")

        clean = {}
        for key, (mean, bad, total) in content_table.items():
            clean[key] = bad == 0

        for key in (("xtts", "en_ref"), ("indicf5", "en_ref_10s"),
                    ("indicf5", "hi_ref_10s"), ("indicf5", "en_ref_25s")):
            if key in content_table:
                mean, bad, total = content_table[key]
                state = "clean" if bad == 0 else f"{bad}/{total} clips broken"
                p(f"    content, {key[0]} {key[1]:<12} {state}")
        p("")

        if not clean.get(("indicf5", "en_ref_10s"), True):
            p("  Note: the winning arm does not say what it was asked to say.")
            p("  Identity is not the binding constraint while that is true.")
            p("")

        if abs(gap) < noise:
            p("  The two models are indistinguishable on identity at this sample")
            p("  size. That settles it in IndicF5's favour anyway: it is MIT and")
            p("  XTTS-v2 cannot be licensed for a product at any quality.")
        elif gap > 0:
            p("  IndicF5 clones this speaker better than XTTS-v2 does, and it is")
            p("  the licensable one. Make it the default and judge naturalness")
            p("  by ear before committing.")
        else:
            p("  XTTS-v2 still clones better. It cannot ship, so the question")
            p("  becomes whether fine-tuning IndicF5 closes a gap of")
            p(f"  {abs(gap):.3f} — which is what 20 to 30 minutes of this speaker buys.")
    else:
        p("  One of the two models produced nothing. Read the log above.")

    p("")
    p("  Identity is one axis. Listen for naturalness and for whether the")
    p("  emphasis lands where you would put it — the cosine cannot see either.")

    REPORT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\nreport written to {REPORT}")

    return rows


def listen(rows=None, arms=None, indexes=None):
    """
    Play the real recording, then each arm on the same sentence.

    `arms` narrows the comparison — with five arms and seven sentences the
    full set is 35 clips, which is more than anyone listens to carefully.
    `indexes` narrows to particular sentences.

        listen(rows, arms=("en_ref", "hi_ref_10s", "en_ref_25s"))
        listen(rows, indexes=(5, 6))          # the short ones
    """
    from IPython.display import Audio, HTML, display

    if not rows:
        rows = []
        for path in sorted(OUT.glob("*/*/[0-9][0-9].wav")):
            rows.append({"model": path.parent.parent.name, "arm": path.parent.name,
                         "index": int(path.stem), "path": path,
                         "dur": sf.info(str(path)).duration, "sim": float("nan"),
                         "text": "", "position": float("nan")})

    if arms:
        rows = [r for r in rows if r["arm"] in arms]
    if indexes is not None:
        rows = [r for r in rows if r["index"] in indexes]

    if not rows:
        display(HTML("<p>Nothing matched that filter.</p>"))
        return

    display(HTML(
        "<h3>You, speaking Hindi &mdash; the target</h3>"
        "<p style='color:#666;margin:2px 0 8px'>Everything below is judged "
        "against this, on three axes the similarity score cannot see: is it "
        "you, does it say the whole sentence, and does the stress land where "
        "you would put it?</p>"
    ))
    display(Audio(filename=str(FIXTURES / "hindi_speech.wav")))

    for index in sorted({r["index"] for r in rows}):
        items = [r for r in rows if r["index"] == index]
        text = next((r["text"] for r in items if r.get("text")), "")
        natural = len(text) / NATURAL_CPS_HI if text else None

        header = f"<h3 style='margin:20px 0 4px'>Sentence {index}</h3>"
        if text:
            header += f"<div style='font-size:15px'>{text}</div>"
        if natural:
            header += (f"<div style='color:#666;font-size:13px;margin-top:3px'>"
                       f"natural Hindi would take about {natural:.1f}s</div>")
        display(HTML(header))

        for row in sorted(items, key=lambda r: (r["model"], r["arm"])):
            bits = [f"{row['dur']:.2f}s"]

            if natural:
                ratio = row["dur"] / natural
                flag = "" if 0.8 <= ratio <= 1.25 else "  <-- pace"
                bits.append(f"{ratio:.2f}x natural{flag}")

            sim, place = row.get("sim"), row.get("position")
            if isinstance(sim, float) and np.isfinite(sim):
                bits.append(f"sim {sim:.3f}")
            if isinstance(place, float) and np.isfinite(place):
                bits.append(f"{place:.0f}% of scale")

            extra, missing = row.get("extra"), row.get("missing")
            if isinstance(extra, float) and np.isfinite(extra):
                bits.append(f"extra {extra:.2f}")
                if extra > MAX_EXTRA:
                    bits.append("<b>GIBBERISH</b>")
            if isinstance(missing, float) and np.isfinite(missing):
                if missing > MAX_MISSING:
                    bits.append(f"<b>CUT SHORT {missing:.2f}</b>")

            display(HTML(
                f"<div style='margin-top:10px'><b>{row['model']} &middot; "
                f"{row['arm']}</b> <span style='color:#666'>&mdash; "
                f"{', '.join(bits)}</span></div>"
            ))
            display(Audio(filename=str(row["path"])))

            # What the transcriber heard, so a bad clip can be read as well as
            # listened to. Shown only when it differs enough to be worth it.
            heard = row.get("heard")
            if heard and isinstance(row.get("cer"), float) and row["cer"] > 0.05:
                display(HTML(
                    f"<div style='color:#a33;font-size:13px;margin:2px 0 0 12px'>"
                    f"heard: {heard}</div>"
                ))


if __name__ == "__main__":
    main()
