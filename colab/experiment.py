"""
Decoder sweep on a returned bundle.

Everything in this project's synthesis settings traces back to one decision:
do_sample=False, taken because sampling produced random output durations. That
fixed durations and, on the evidence of the first two real runs, cost all the
prosody. This script measures the trade rather than arguing about it.

It loads the model once, runs every configuration over every segment, and
writes the audio out so it can be listened to side by side. It changes nothing
in the repo and does not touch the existing bundle output.
"""

import json
import sys
import time
import unicodedata
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from colab.xtts_worker import CONDITIONING_PARAMS, XTTSWorker


BUNDLE = Path("/content/tts_bundle")
OUT_ROOT = Path("/content/experiments")

# Measured from FLEURS, n=239. See docs/measured_speaking_rates.txt.
NATURAL_CPS_HI = 10.81

# Each entry is what gets passed to Xtts.inference(). Signature defaults in
# coqui-tts 0.27.5: temperature 0.75, repetition_penalty 10.0, top_k 50,
# top_p 0.85, do_sample True, speed 1.0.
CONFIGS = {
    # What the pipeline ships today.
    "greedy": {
        "do_sample": False,
        "repetition_penalty": 5.0,
        "enable_text_splitting": False,
    },
    # XTTS as its authors decode it. The prosody reference point.
    "sampled": {
        "do_sample": True,
        "temperature": 0.75,
        "repetition_penalty": 5.0,
        "enable_text_splitting": False,
    },
    # Sampling plus native rate control, which beats stretching afterward.
    "sampled_fast": {
        "do_sample": True,
        "temperature": 0.75,
        "repetition_penalty": 5.0,
        "speed": 1.2,
        "enable_text_splitting": False,
    },
    # Does splitting help the long segments, or just add seams?
    "sampled_split": {
        "do_sample": True,
        "temperature": 0.75,
        "repetition_penalty": 5.0,
        "enable_text_splitting": True,
    },
    # Isolates rate control from sampling.
    "greedy_fast": {
        "do_sample": False,
        "repetition_penalty": 5.0,
        "speed": 1.2,
        "enable_text_splitting": False,
    },
}

# How many times to repeat one segment under sampling, to measure the duration
# spread that greedy decoding was adopted to eliminate.
VARIANCE_RUNS = 4
VARIANCE_CONFIG = "sampled"


def audio_stats(path):
    with wave.open(str(path), "rb") as h:
        rate, n = h.getframerate(), h.getnframes()
        raw = h.readframes(n)
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    if a.size == 0:
        return {"silent_pct": 0.0, "peak": 0.0}
    peak = float(np.abs(a).max())
    quiet = np.abs(a) <= max(peak * 0.02, 1e-4)
    return {"silent_pct": 100.0 * float(quiet.mean()), "peak": peak}


def main():
    worker = XTTSWorker(BUNDLE)
    worker.load_bundle()
    worker.load_model()
    worker.compute_speaker_embedding()

    segments = worker.request.segments
    rate = worker.request.output_sample_rate

    print(f"\n{len(segments)} segments, {len(CONFIGS)} configurations, "
          f"conditioning {CONDITIONING_PARAMS}\n")

    rows = {}

    for name, params in CONFIGS.items():
        out_dir = OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        rows[name] = []

        print(f"--- {name}: {json.dumps(params)}")

        for segment in segments:
            slot = segment.end_ts - segment.start_ts
            started = time.perf_counter()

            with torch.no_grad():
                out = worker.xtts.inference(
                    text=segment.text,
                    language=worker.request.language,
                    gpt_cond_latent=worker.gpt_cond_latent,
                    speaker_embedding=worker.speaker_embedding,
                    **params,
                )

            wav = np.asarray(out["wav"], dtype=np.float32)
            path = out_dir / f"seg_{segment.segment_id:05d}.wav"
            sf.write(str(path), wav, rate, subtype="PCM_16")

            duration = wav.size / rate
            latents = out.get("gpt_latents")

            rows[name].append({
                "id": segment.segment_id,
                "slot": slot,
                "dur": duration,
                "ratio": duration / slot if slot else 0.0,
                "cps": len(segment.text) / duration if duration else 0.0,
                "sim": worker._speaker_similarity(path),
                "tok": int(latents.shape[1]) if latents is not None else None,
                "sil": audio_stats(path)["silent_pct"],
                "gen_s": time.perf_counter() - started,
            })

            print(f"    seg {segment.segment_id}  {duration:6.2f}s  "
                  f"{duration / slot if slot else 0:5.2f}x slot")

    # ---------------------------------------------------------------- tables
    print("\n" + "=" * 78)
    print("PER SEGMENT  (ratio = generated / slot, 1.00 is a perfect fit)")
    print("=" * 78)

    head = f"{'config':<14}" + "".join(f"{'s' + str(s.segment_id):>10}" for s in segments)
    for metric, label, fmt in (
        ("ratio", "duration ratio", "{:>10.2f}"),
        ("cps", "chars/sec", "{:>10.1f}"),
        ("sim", "speaker sim", "{:>10.3f}"),
        ("sil", "silence %", "{:>10.1f}"),
    ):
        print(f"\n{label}   (natural Hindi is {NATURAL_CPS_HI} cps)"
              if metric == "cps" else f"\n{label}")
        print(head)
        for name in CONFIGS:
            line = f"{name:<14}"
            for row in rows[name]:
                v = row[metric]
                line += fmt.format(v) if isinstance(v, (int, float)) else f"{'-':>10}"
            print(line)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'config':<14}{'mean ratio':>12}{'mean cps':>10}{'mean sim':>10}"
          f"{'mean sil%':>11}{'gen s':>8}")
    for name in CONFIGS:
        rs = rows[name]
        sims = [r["sim"] for r in rs if r["sim"] is not None]
        print(f"{name:<14}"
              f"{np.mean([r['ratio'] for r in rs]):>12.2f}"
              f"{np.mean([r['cps'] for r in rs]):>10.1f}"
              f"{(np.mean(sims) if sims else float('nan')):>10.3f}"
              f"{np.mean([r['sil'] for r in rs]):>11.1f}"
              f"{sum(r['gen_s'] for r in rs):>8.1f}")

    # ------------------------------------------------------ duration spread
    print("\n" + "=" * 78)
    print(f"DURATION SPREAD UNDER SAMPLING  ({VARIANCE_RUNS} runs, identical input)")
    print("This is the number greedy decoding was adopted to eliminate.")
    print("=" * 78)

    target = max(segments, key=lambda s: len(s.text))
    spread_dir = OUT_ROOT / "spread"
    spread_dir.mkdir(parents=True, exist_ok=True)
    durations = []

    for run in range(VARIANCE_RUNS):
        with torch.no_grad():
            out = worker.xtts.inference(
                text=target.text,
                language=worker.request.language,
                gpt_cond_latent=worker.gpt_cond_latent,
                speaker_embedding=worker.speaker_embedding,
                **CONFIGS[VARIANCE_CONFIG],
            )
        wav = np.asarray(out["wav"], dtype=np.float32)
        sf.write(str(spread_dir / f"run_{run}.wav"), wav, rate, subtype="PCM_16")
        durations.append(wav.size / rate)
        print(f"  run {run}: {durations[-1]:.2f}s")

    lo, hi = min(durations), max(durations)
    print(f"\n  segment {target.segment_id}, {len(target.text)} chars")
    print(f"  min {lo:.2f}s  max {hi:.2f}s  spread {hi - lo:.2f}s "
          f"({100 * (hi - lo) / lo:.0f}% of the shortest)")
    print(f"  std {np.std(durations):.3f}s")

    print(f"\nAudio written under {OUT_ROOT}. Listen to the same segment across "
          f"configurations before reading anything into the numbers.")


if __name__ == "__main__":
    main()
