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


def load_indicf5():
    """
    Load IndicF5 through transformers' remote-code path.

    Kept in its own function so the import failure, which is the likely one on
    a runtime that has not installed it, is reported as itself rather than as
    a mysterious failure halfway through a synthesis loop.
    """
    from transformers import AutoModel

    model = AutoModel.from_pretrained(INDICF5_REPO, trust_remote_code=True)

    # The model card never moves it to a device, so the remote code may place
    # its own submodules and expose nothing to move. Try, and carry on if the
    # object does not support it rather than failing the whole comparison.
    try:
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    except (AttributeError, NotImplementedError, RuntimeError) as exc:
        p(f"  note: could not move IndicF5 to the GPU ({type(exc).__name__}); "
          f"using it as loaded")

    return model


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
              "hindi_reference.wav", "scripted_text.json", "reference_text.json"]
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

    section("CALIBRATION — same rig as the four-arm run")

    def thirds(path, label, anchor):
        audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        edges = np.linspace(0, audio.size, 4).astype(int)
        scores = []
        for index in range(3):
            piece = OUT / f"{label}_{index}.wav"
            sf.write(str(piece), audio[edges[index]:edges[index + 1]], rate, subtype="PCM_16")
            embedding = embed(piece)
            if embedding is not None:
                scores.append(cosine(anchor, embedding))
        return scores

    same = thirds(FIXTURES / "english_speech.wav", "en_third", anchor_en)
    cross = [cosine(anchor_en, anchor_hi)] + thirds(
        FIXTURES / "hindi_speech.wav", "hi_third", anchor_en
    )

    floor_scores = []
    try:
        for _, entry in xtts.speaker_manager.speakers.items():
            floor_scores.append(cosine(anchor_en, entry["speaker_embedding"]))
    except Exception as exc:
        p(f"  floor unavailable: {type(exc).__name__}: {exc}")

    floor = float(np.median(floor_scores)) if floor_scores else 0.0
    ceiling_same = float(np.mean(same)) if same else float("nan")
    ceiling_cross = float(np.mean(cross)) if cross else float("nan")

    p(f"  floor                   {floor:.3f}   ({len(floor_scores)} real strangers)")
    p(f"  same-language ceiling   {ceiling_same:.3f}")
    p(f"  cross-language ceiling  {ceiling_cross:.3f}   (you in Hindi vs your English anchor)")

    def position(score, ceiling):
        span = ceiling - floor
        if not np.isfinite(score) or not np.isfinite(span) or span <= 0:
            return float("nan")
        return 100.0 * (score - floor) / span

    # --------------------------------------------------------------- the arms
    arms = [
        ("en_ref", "english_reference.wav", "english", anchor_en, ceiling_cross),
        ("hi_ref", "hindi_reference.wav", "hindi", anchor_hi, ceiling_same),
    ]

    rows = []

    def record(model, arm, index, sentence, wav, anchor, ceiling, elapsed):
        directory = OUT / model / arm
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{index:02d}.wav"
        sf.write(str(path), wav, SAMPLE_RATE, subtype="PCM_16")

        embedding = embed(path)
        score = cosine(anchor, embedding) if embedding is not None else float("nan")
        duration = wav.size / SAMPLE_RATE

        rows.append({
            "model": model, "arm": arm, "index": index, "text": sentence,
            "path": path, "dur": duration, "sim": score, "ceiling": ceiling,
            "cps": len(sentence) / duration if duration else 0.0,
            "gen_s": elapsed,
        })
        p(f"    [{index}] {duration:6.2f}s  {rows[-1]['cps']:5.1f} cps  "
          f"sim {score:.3f}  {position(score, ceiling):4.0f}% of scale")

    section("XTTS-v2   (CPML, non-commercial — the baseline, not a candidate)")

    for arm, filename, _, anchor, ceiling in arms:
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
                   anchor, ceiling, time.perf_counter() - started)

    section("IndicF5   (MIT — the one that could actually ship)")

    try:
        indicf5 = load_indicf5()
    except Exception as exc:
        p(f"!! IndicF5 did not load: {type(exc).__name__}: {exc}")
        p("   pip install git+https://github.com/ai4bharat/IndicF5.git")
        indicf5 = None

    if indicf5 is not None:
        for arm, filename, key, anchor, ceiling in arms:
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
                       anchor, ceiling, time.perf_counter() - started)

    # ------------------------------------------------------------------ result
    section("RESULT")
    p(f"{'model':<10}{'arm':<9}{'n':>3}{'sim':>8}{'se':>7}{'ceiling':>9}"
      f"{'position':>10}{'cps':>7}{'natural':>9}{'gen':>7}")

    table = {}
    for model in ("xtts", "indicf5"):
        for arm, _, _, _, ceiling in arms:
            items = [r for r in rows if r["model"] == model and r["arm"] == arm]
            scores = [r["sim"] for r in items if np.isfinite(r["sim"])]
            if not scores:
                continue
            mean = float(np.mean(scores))
            se = float(np.std(scores, ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else float("nan")
            cps = float(np.mean([r["cps"] for r in items]))
            gen = float(np.mean([r["gen_s"] for r in items]))
            table[(model, arm)] = (mean, se, position(mean, ceiling))
            p(f"{model:<10}{arm:<9}{len(scores):>3}{mean:>8.3f}{se:>7.3f}"
              f"{ceiling:>9.3f}{position(mean, ceiling):>9.0f}%"
              f"{cps:>7.1f}{cps / NATURAL_CPS_HI:>8.2f}x{gen:>7.1f}s")

    section("VERDICT")

    production = [table.get(("xtts", "en_ref")), table.get(("indicf5", "en_ref"))]

    if all(production):
        (x_mean, x_se, x_pos), (i_mean, i_se, i_pos) = production
        gap = i_mean - x_mean
        noise = float(np.hypot(x_se, i_se))

        p(f"  Production case, English reference speaking Hindi:")
        p(f"    XTTS-v2   {x_mean:.3f}  ({x_pos:.0f}% of scale)")
        p(f"    IndicF5   {i_mean:.3f}  ({i_pos:.0f}% of scale)")
        p(f"    difference {gap:+.3f}, combined standard error {noise:.3f}")
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


def listen(rows=None):
    """
    Play the real recording, then both models side by side per sentence.
    """
    from IPython.display import Audio, HTML, display

    if not rows:
        rows = []
        for path in sorted(OUT.glob("*/*/[0-9][0-9].wav")):
            rows.append({"model": path.parent.parent.name, "arm": path.parent.name,
                         "index": int(path.stem), "path": path,
                         "dur": sf.info(str(path)).duration, "sim": float("nan"),
                         "text": ""})

    display(HTML("<h3>You, speaking Hindi — the target</h3>"))
    display(Audio(filename=str(FIXTURES / "hindi_speech.wav")))

    for index in sorted({r["index"] for r in rows}):
        items = [r for r in rows if r["index"] == index]
        display(HTML(f"<h3 style='margin:18px 0 4px'>Sentence {index}</h3>"
                     f"<div style='font-size:15px'>{items[0].get('text', '')}</div>"))
        for row in sorted(items, key=lambda r: (r["model"], r["arm"])):
            sim = row.get("sim")
            label = f"{row['model']} · {row['arm']}"
            detail = f"{row['dur']:.2f}s"
            if isinstance(sim, float) and np.isfinite(sim):
                detail += f", sim {sim:.3f}"
            display(HTML(f"<div style='margin-top:8px'><b>{label}</b> "
                         f"<span style='color:#666'>— {detail}</span></div>"))
            display(Audio(filename=str(row["path"])))


if __name__ == "__main__":
    main()
