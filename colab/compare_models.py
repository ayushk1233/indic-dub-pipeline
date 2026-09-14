"""
Does the voice sound thin because of how XTTS is conditioned, or because it is
XTTS?

Two questions, one run. First, whether XTTS improves when it is allowed to use
a long reference: the pipeline shipped gpt_cond_len=8 and max_ref_length=10,
which throw away all but 10 seconds of whatever it is handed. Those numbers
came from a determinism investigation, not from a cloning one, and nobody has
checked what they cost.

Second, whether XTTS-v2 is the wrong model for Hindi. IndicF5 is trained on
Indian languages and is MIT licensed, where XTTS-v2's weights are
non-commercial under a license nobody can now grant. It conditions on
reference audio together with its transcript, which bundle 1.1 now carries.

Writes audio under /content/model_comparison. Changes nothing in the bundle.
"""

import json
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from colab.xtts_worker import XTTSWorker


BUNDLE = Path("/content/tts_bundle")
OUT_ROOT = Path("/content/model_comparison")

NATURAL_CPS_HI = 10.81

# Decoding is held fixed across both XTTS runs so the only variable is how much
# reference the model is allowed to see. Sampling, because the sweep showed it
# scored the best speaker similarity of any decoder setting.
DECODE = {
    "do_sample": True,
    "temperature": 0.75,
    "repetition_penalty": 5.0,
    "enable_text_splitting": False,
}

CONDITIONING = {
    # What the pipeline ships. Caps the reference at 10s however long it is.
    "xtts_short_cond": {
        "gpt_cond_len": 8,
        "gpt_cond_chunk_len": 4,
        "max_ref_length": 10,
        "sound_norm_refs": True,
    },
    # Let it use the whole clip. coqui-tts defaults are 6/6/30.
    "xtts_long_cond": {
        "gpt_cond_len": 30,
        "gpt_cond_chunk_len": 12,
        "max_ref_length": 30,
        "sound_norm_refs": True,
    },
}


def anchor_embedding(worker, reference):
    """
    One fixed yardstick for every row in this comparison.

    Scoring each configuration against the embedding it produced for itself
    measures self-consistency, which rises as conditioning narrows. That would
    have made the 10-second cap look like the winner by construction.
    """
    _, embedding = worker.xtts.get_conditioning_latents(
        audio_path=[str(reference)],
        gpt_cond_len=60,
        gpt_cond_chunk_len=30,
        max_ref_length=60,
        sound_norm_refs=True,
    )
    return embedding


def score(worker, path, anchor):
    try:
        _, produced = worker.xtts.get_conditioning_latents(audio_path=[str(path)])
    except Exception:
        return None

    a = anchor.reshape(1, -1).float()
    b = produced.reshape(1, -1).float().to(a.device)

    return float(torch.nn.functional.cosine_similarity(a, b).item())


def summarize(name, rows):
    sims = [r["sim"] for r in rows if r.get("sim") is not None]
    print(
        f"{name:<18}"
        f"{np.mean([r['ratio'] for r in rows]):>12.2f}"
        f"{np.mean([r['cps'] for r in rows]):>10.1f}"
        f"{(np.mean(sims) if sims else float('nan')):>10.3f}"
        f"{sum(r['gen_s'] for r in rows):>9.1f}"
    )


def run_xtts(worker, segments, rate, results, anchor):
    for name, conditioning in CONDITIONING.items():
        out_dir = OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- {name}: {json.dumps(conditioning)}")

        # Recondition from scratch; the latents are cached on the worker.
        worker.gpt_cond_latent = None
        worker.speaker_embedding = None
        latent, embedding = worker.xtts.get_conditioning_latents(
            audio_path=[str(BUNDLE / "request" / "reference.wav")],
            **conditioning,
        )
        worker.gpt_cond_latent, worker.speaker_embedding = latent, embedding

        rows = []

        for segment in segments:
            slot = segment.end_ts - segment.start_ts
            started = time.perf_counter()

            with torch.no_grad():
                out = worker.xtts.inference(
                    text=segment.text,
                    language=worker.request.language,
                    gpt_cond_latent=latent,
                    speaker_embedding=embedding,
                    **DECODE,
                )

            wav = np.asarray(out["wav"], dtype=np.float32)
            path = out_dir / f"seg_{segment.segment_id:05d}.wav"
            sf.write(str(path), wav, rate, subtype="PCM_16")

            duration = wav.size / rate
            rows.append({
                "id": segment.segment_id,
                "dur": duration,
                "ratio": duration / slot if slot else 0.0,
                "cps": len(segment.text) / duration if duration else 0.0,
                "sim": score(worker, path, anchor),
                "gen_s": time.perf_counter() - started,
            })
            print(f"    seg {segment.segment_id}  {duration:6.2f}s  "
                  f"{rows[-1]['ratio']:5.2f}x slot  sim {rows[-1]['sim']}")

        results[name] = rows


def run_indicf5(worker, segments, rate, results, anchor):
    print("\n--- indicf5")

    request = worker.request
    reference_text = getattr(request, "reference_text", None)

    if not reference_text:
        print("    SKIPPED: the bundle carries no reference_text. "
              "Re-export with bundle 1.1.")
        return

    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)
        model = model.to("cuda")
    except Exception:
        print("    FAILED to load. Full traceback:")
        traceback.print_exc()
        print("    Install it with:  "
              "pip install git+https://github.com/ai4bharat/IndicF5.git")
        return

    out_dir = OUT_ROOT / "indicf5"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    print(f"    reference transcript: {reference_text[:80]}...")

    for segment in segments:
        slot = segment.end_ts - segment.start_ts
        started = time.perf_counter()

        try:
            audio = model(
                segment.text,
                ref_audio_path=str(BUNDLE / "request" / "reference.wav"),
                ref_text=reference_text,
            )
        except Exception:
            print(f"    seg {segment.segment_id} FAILED:")
            traceback.print_exc()
            continue

        wav = np.asarray(audio, dtype=np.float32)
        if wav.dtype == np.int16 or np.abs(wav).max() > 1.5:
            wav = wav.astype(np.float32) / 32768.0

        path = out_dir / f"seg_{segment.segment_id:05d}.wav"
        sf.write(str(path), wav, rate, subtype="PCM_16")

        duration = wav.size / rate
        rows.append({
            "id": segment.segment_id,
            "dur": duration,
            "ratio": duration / slot if slot else 0.0,
            "cps": len(segment.text) / duration if duration else 0.0,
            # XTTS's speaker encoder, used on non-XTTS audio. Comparable
            # between these rows; not an absolute score.
            "sim": score(worker, path, anchor),
            "gen_s": time.perf_counter() - started,
        })
        print(f"    seg {segment.segment_id}  {duration:6.2f}s  "
              f"{rows[-1]['ratio']:5.2f}x slot  sim {rows[-1]['sim']}")

    if rows:
        results["indicf5"] = rows


def main():
    worker = XTTSWorker(BUNDLE)
    worker.load_bundle()
    worker.load_model()

    segments = worker.request.segments
    rate = worker.request.output_sample_rate

    ref = BUNDLE / "request" / "reference.wav"
    info = sf.info(str(ref))
    print(f"\nreference {info.duration:.2f}s at {info.samplerate} Hz, "
          f"{len(segments)} segments")

    anchor = anchor_embedding(worker, ref)

    results = {}
    run_xtts(worker, segments, rate, results, anchor)
    run_indicf5(worker, segments, rate, results, anchor)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'model':<18}{'mean ratio':>12}{'mean cps':>10}{'mean sim':>10}{'gen s':>9}")
    for name, rows in results.items():
        summarize(name, rows)
    print(f"\nnatural Hindi is {NATURAL_CPS_HI} cps; a ratio of 1.00 fits the slot")
    print(f"audio under {OUT_ROOT}")


if __name__ == "__main__":
    main()
