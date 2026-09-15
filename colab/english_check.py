"""
Is it Hindi, or is it the whole setup?

XTTS-v2 is strongest in English. This synthesizes the speaker's own English
sentences, from the same reference, at the same conditioning and decoding, and
plays them next to the real recording. Everything except the language is held
fixed, so the comparison is clean.

If the English sounds like the speaker, the pipeline is sound and XTTS's Hindi
is the ceiling, which settles the IndicF5 question before running it. If the
English is thin too, the fault is upstream of language and no amount of model
swapping will help.

The text is the transcript the ASR stage produced for the reference clip, so
the model is asked to say the same words the reference recording contains.
That makes the reference itself the ground truth to judge against.
"""

import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from IPython.display import Audio, HTML, display

from colab.xtts_worker import XTTSWorker


BUNDLE = Path("/content/tts_bundle")
OUT_ROOT = Path("/content/english_check")

# Measured from FLEURS, n=394 English and n=239 Hindi.
NATURAL_CPS = {"en": 13.13, "hi": 10.81}

# The conditioning that won the sweep. Held fixed here; language is the
# variable under test.
CONDITIONING = {
    "gpt_cond_len": 20,
    "gpt_cond_chunk_len": 10,
    "max_ref_length": 30,
    "sound_norm_refs": True,
}

DECODES = {
    "greedy": {
        "do_sample": False,
        "repetition_penalty": 5.0,
        "enable_text_splitting": False,
    },
    "sampled": {
        "do_sample": True,
        "temperature": 0.75,
        "repetition_penalty": 5.0,
        "enable_text_splitting": False,
    },
}

# Used only if the bundle carries no reference transcript.
FALLBACK = [
    "This project extracts audio from a video and dubs it into another language.",
    "The hardest part is making the translation fit the time available.",
    "Hello everyone, my name is Ayush.",
]

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def sentences(text, limit=5):
    parts = [s.strip() for s in _SENTENCE_RE.split(text or "") if len(s.strip()) > 15]
    return parts[:limit]


def cosine(a, b):
    a = a.reshape(1, -1).float()
    b = b.reshape(1, -1).float().to(a.device)
    return float(torch.nn.functional.cosine_similarity(a, b).item())


def main():
    worker = XTTSWorker(BUNDLE)
    worker.load_bundle()
    worker.load_model()

    reference = BUNDLE / "request" / "reference.wav"
    rate = worker.request.output_sample_rate

    reference_text = getattr(worker.request, "reference_text", None)
    texts = sentences(reference_text) or FALLBACK

    if not reference_text:
        print("!! bundle carries no reference_text; using generic sentences. "
              "Re-export with bundle 1.1 to say the speaker's own words.")

    info = sf.info(str(reference))
    print(f"reference {info.duration:.2f}s at {info.samplerate} Hz")
    print(f"{len(texts)} English sentences, {len(DECODES)} decoder settings\n")

    _, anchor = worker.xtts.get_conditioning_latents(
        audio_path=[str(reference)], gpt_cond_len=60,
        gpt_cond_chunk_len=30, max_ref_length=60, sound_norm_refs=True,
    )

    latent, embedding = worker.xtts.get_conditioning_latents(
        audio_path=[str(reference)], **CONDITIONING
    )

    rows = []

    for decode_name, decode in DECODES.items():
        out_dir = OUT_ROOT / decode_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"--- {decode_name}")

        for index, text in enumerate(texts):
            started = time.perf_counter()

            with torch.no_grad():
                out = worker.xtts.inference(
                    text=text,
                    language="en",
                    gpt_cond_latent=latent,
                    speaker_embedding=embedding,
                    **decode,
                )

            wav = np.asarray(out["wav"], dtype=np.float32)
            path = out_dir / f"en_{index:02d}.wav"
            sf.write(str(path), wav, rate, subtype="PCM_16")

            try:
                _, produced = worker.xtts.get_conditioning_latents(
                    audio_path=[str(path)]
                )
                sim = cosine(anchor, produced)
            except Exception:
                sim = None

            duration = wav.size / rate
            rows.append({
                "decode": decode_name,
                "index": index,
                "text": text,
                "path": path,
                "dur": duration,
                "cps": len(text) / duration if duration else 0.0,
                "sim": sim,
                "gen_s": time.perf_counter() - started,
            })
            print(f"    [{index}] {duration:6.2f}s  "
                  f"{rows[-1]['cps']:5.1f} cps  "
                  f"sim {sim:.3f}" if sim is not None else "    sim -")

    # ------------------------------------------------------------- summary
    print("\n" + "=" * 72)
    print(f"ENGLISH  (natural English is {NATURAL_CPS['en']} cps)")
    print("=" * 72)
    print(f"{'decode':<10}{'mean cps':>10}{'mean sim':>10}{'vs natural':>12}")
    for name in DECODES:
        rs = [r for r in rows if r["decode"] == name]
        sims = [r["sim"] for r in rs if r["sim"] is not None]
        cps = np.mean([r["cps"] for r in rs])
        print(f"{name:<10}{cps:>10.1f}"
              f"{(np.mean(sims) if sims else float('nan')):>10.3f}"
              f"{cps / NATURAL_CPS['en']:>11.2f}x")

    print("\nHindi from the last full run, for comparison:")
    print("  mean similarity around 0.50 across every conditioning setting,")
    print(f"  delivered 5 to 10 cps against {NATURAL_CPS['hi']} natural.")

    # ------------------------------------------------------------- listening
    display(HTML("<h3 style='margin:18px 0 6px'>The real recording "
                 "&mdash; ground truth</h3>"))
    display(Audio(filename=str(reference)))

    for index, text in enumerate(texts):
        display(HTML(
            f"<h3 style='margin:18px 0 4px'>Sentence {index}</h3>"
            f"<div style='font-size:15px;margin-bottom:8px'>{text}</div>"
        ))
        for name in DECODES:
            match = next(
                (r for r in rows if r["decode"] == name and r["index"] == index),
                None,
            )
            if match is None:
                continue
            display(HTML(
                f"<div style='margin-top:8px'><b>{name}</b> "
                f"<span style='color:#666'>&mdash; {match['dur']:.2f}s, "
                f"{match['cps']:.1f} cps</span></div>"
            ))
            display(Audio(filename=str(match["path"])))

    print("\nThe question is not whether the English is perfect. It is whether "
          "it sounds like the person in the reference, and whether it sounds "
          "like a person at all.")


if __name__ == "__main__":
    main()
