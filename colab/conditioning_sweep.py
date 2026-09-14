"""
What does capping the reference at 10 seconds cost?

The pipeline ships gpt_cond_len=8, gpt_cond_chunk_len=4, max_ref_length=10.
Those came out of an investigation into random output durations, not out of
any cloning measurement, and they discard everything past the first 10 seconds
of a reference that is now 21 seconds long. This sweeps them.

Method note, because it decides whether the numbers mean anything. Speaker
similarity is scored against a FIXED anchor embedding computed once from the
whole reference, never against the embedding each configuration produced for
itself. Scoring each configuration against its own embedding measures how
self-consistent it is, which improves as conditioning gets narrower and would
have pointed at exactly the wrong answer.

Decoding is greedy throughout, deliberately. It sounds worse, but it is
deterministic, so every difference in the table is attributable to
conditioning rather than to a lucky sample. The winning configuration is then
re-run with sampling so there is something worth listening to.
"""

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from colab.xtts_worker import XTTSWorker


BUNDLE = Path("/content/tts_bundle")
OUT_ROOT = Path("/content/conditioning")

NATURAL_CPS_HI = 10.81

# Scoring only. Long enough to take in the whole reference whatever its length.
ANCHOR = {"gpt_cond_len": 60, "gpt_cond_chunk_len": 30, "max_ref_length": 60}

GREEDY = {
    "do_sample": False,
    "repetition_penalty": 5.0,
    "enable_text_splitting": False,
}

SAMPLED = {
    "do_sample": True,
    "temperature": 0.75,
    "repetition_penalty": 5.0,
    "enable_text_splitting": False,
}

# gpt_cond_len is how many seconds of reference feed the GPT conditioning;
# max_ref_length truncates the clip before any of that happens, so raising one
# without the other changes nothing. coqui-tts defaults are 6 / 6 / 30.
CONFIGS = {
    "ship_8_4_10":     {"gpt_cond_len": 8,  "gpt_cond_chunk_len": 4,  "max_ref_length": 10},
    "coqui_6_6_30":    {"gpt_cond_len": 6,  "gpt_cond_chunk_len": 6,  "max_ref_length": 30},
    "mid_12_6_30":     {"gpt_cond_len": 12, "gpt_cond_chunk_len": 6,  "max_ref_length": 30},
    "long_20_10_30":   {"gpt_cond_len": 20, "gpt_cond_chunk_len": 10, "max_ref_length": 30},
    "full_30_12_30":   {"gpt_cond_len": 30, "gpt_cond_chunk_len": 12, "max_ref_length": 30},
    "nochunk_30_30_30": {"gpt_cond_len": 30, "gpt_cond_chunk_len": 30, "max_ref_length": 30},
    # Isolates the level normalization from the length question.
    "full_no_norm":    {"gpt_cond_len": 30, "gpt_cond_chunk_len": 12, "max_ref_length": 30,
                        "sound_norm_refs": False},
}

DEFAULT_NORM = {"sound_norm_refs": True}


def cosine(a, b):
    a = a.reshape(1, -1).float()
    b = b.reshape(1, -1).float().to(a.device)
    return float(torch.nn.functional.cosine_similarity(a, b).item())


def generate(worker, segments, rate, latent, embedding, decode, out_dir, anchor):
    out_dir.mkdir(parents=True, exist_ok=True)
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
                **decode,
            )

        wav = np.asarray(out["wav"], dtype=np.float32)
        path = out_dir / f"seg_{segment.segment_id:05d}.wav"
        sf.write(str(path), wav, rate, subtype="PCM_16")

        try:
            _, produced = worker.xtts.get_conditioning_latents(audio_path=[str(path)])
            sim = cosine(anchor, produced)
        except Exception:
            sim = None

        duration = wav.size / rate
        rows.append({
            "id": segment.segment_id,
            "dur": duration,
            "ratio": duration / slot if slot else 0.0,
            "cps": len(segment.text) / duration if duration else 0.0,
            "sim": sim,
            "gen_s": time.perf_counter() - started,
        })

    return rows


def main():
    worker = XTTSWorker(BUNDLE)
    worker.load_bundle()
    worker.load_model()

    segments = worker.request.segments
    rate = worker.request.output_sample_rate
    reference = BUNDLE / "request" / "reference.wav"

    info = sf.info(str(reference))
    print(f"\nreference {info.duration:.2f}s at {info.samplerate} Hz, "
          f"{len(segments)} segments, {len(CONFIGS)} configurations")

    if info.duration < 15:
        print("!! reference under 15s; this sweep cannot show what long "
              "conditioning buys. Re-export with bundle 1.1 first.")

    # The fixed yardstick. Computed once, never from a swept configuration.
    _, anchor = worker.xtts.get_conditioning_latents(
        audio_path=[str(reference)], sound_norm_refs=True, **ANCHOR
    )
    print(f"anchor embedding from the full reference: {tuple(anchor.shape)}")

    results = {}

    for name, conditioning in CONFIGS.items():
        params = {**DEFAULT_NORM, **conditioning}
        print(f"\n--- {name}: {json.dumps(params)}")

        latent, embedding = worker.xtts.get_conditioning_latents(
            audio_path=[str(reference)], **params
        )
        print(f"    latent {tuple(latent.shape)}  embedding {tuple(embedding.shape)}")

        rows = generate(
            worker, segments, rate, latent, embedding,
            GREEDY, OUT_ROOT / name, anchor,
        )
        results[name] = rows

        sims = [r["sim"] for r in rows if r["sim"] is not None]
        print(f"    mean sim {np.mean(sims):.3f}" if sims else "    sim unavailable")

    # ------------------------------------------------------------- tables
    print("\n" + "=" * 78)
    print("SPEAKER SIMILARITY vs the fixed anchor  (higher is a better clone)")
    print("=" * 78)
    head = f"{'config':<20}" + "".join(f"{'s' + str(s.segment_id):>9}" for s in segments) + f"{'mean':>9}"
    print(head)
    for name, rows in results.items():
        line = f"{name:<20}"
        for r in rows:
            line += f"{r['sim']:>9.3f}" if r["sim"] is not None else f"{'-':>9}"
        sims = [r["sim"] for r in rows if r["sim"] is not None]
        line += f"{np.mean(sims):>9.3f}" if sims else f"{'-':>9}"
        print(line)

    print("\n" + "=" * 78)
    print(f"DURATION AND PACE  (natural Hindi {NATURAL_CPS_HI} cps)")
    print("=" * 78)
    print(f"{'config':<20}{'mean ratio':>12}{'mean cps':>10}{'gen s':>9}")
    for name, rows in results.items():
        print(f"{name:<20}"
              f"{np.mean([r['ratio'] for r in rows]):>12.2f}"
              f"{np.mean([r['cps'] for r in rows]):>10.1f}"
              f"{sum(r['gen_s'] for r in rows):>9.1f}")

    # ------------------------------------------------- the winner, audible
    def mean_sim(name):
        sims = [r["sim"] for r in results[name] if r["sim"] is not None]
        return np.mean(sims) if sims else -1.0

    best = max(results, key=mean_sim)
    ship = mean_sim("ship_8_4_10")

    print("\n" + "=" * 78)
    print(f"BEST BY SIMILARITY: {best} at {mean_sim(best):.3f}, "
          f"against {ship:.3f} for what ships today "
          f"({100 * (mean_sim(best) - ship) / ship:+.1f}%)")
    print("=" * 78)

    latent, embedding = worker.xtts.get_conditioning_latents(
        audio_path=[str(reference)], **{**DEFAULT_NORM, **CONFIGS[best]}
    )
    generate(
        worker, segments, rate, latent, embedding,
        SAMPLED, OUT_ROOT / f"{best}_sampled", anchor,
    )

    print(f"\nSampled audio for {best} written to "
          f"{OUT_ROOT / (best + '_sampled')}.")
    print("Numbers rank the clones; only listening says whether any of them "
          "sounds like a person.")


if __name__ == "__main__":
    main()
