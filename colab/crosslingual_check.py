"""
Split "speak good Hindi" from "sound like this person".

The app that sounded good cloned a Hindi voice saying Hindi. This pipeline
asks an English speaker to say Hindi, which is cross-lingual cloning, and that
is where XTTS-v2 is weak. Every parameter swept so far has moved speaker
similarity by less than the noise between segments, which is what a task-level
limit looks like rather than a tuning problem.

So stop asking one model to do both jobs. Three paths, measured on the same
text:

  A  XTTS conditioned on the English speaker, in Hindi. What ships now.
  B  XTTS conditioned on a native Hindi donor voice. Should sound like good
     Hindi, spoken by the wrong person.
  C  B, then voice conversion onto the English speaker. Good Hindi in the
     right voice, if the conversion holds up.

C is how production dubbing systems are usually built, and it is the reason to
run this before touching IndicF5 or any fine-tune: if the split works, the
synthesis model stops being the bottleneck.

Needs a Hindi reference at HINDI_REF. Upload the sample that sounded good in
the desktop app.
"""

import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from IPython.display import Audio, HTML, display

from colab.xtts_worker import XTTSWorker


BUNDLE = Path("/content/tts_bundle")
HINDI_REF = Path("/content/hindi_ref.wav")
OUT_ROOT = Path("/content/crosslingual")

VC_MODEL = "voice_conversion_models/multilingual/multi-dataset/openvoice_v2"

NATURAL_CPS_HI = 10.81

# The shipped XTTS-v2 config, which is what tts_to_file uses and what the
# desktop app therefore ran. Not the worker's settings.
CONDITIONING = {
    "gpt_cond_len": 30,
    "gpt_cond_chunk_len": 4,
    "max_ref_length": 30,
    "sound_norm_refs": False,
}

DECODE = {
    "temperature": 0.75,
    "length_penalty": 1.0,
    "repetition_penalty": 5.0,
    "top_k": 50,
    "top_p": 0.85,
    "do_sample": True,
}


def cosine(a, b):
    a = a.reshape(1, -1).float()
    b = b.reshape(1, -1).float().to(a.device)
    return float(torch.nn.functional.cosine_similarity(a, b).item())


def synth(worker, segments, rate, latent, embedding, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    produced = {}

    for segment in segments:
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
        produced[segment.segment_id] = (path, wav.size / rate,
                                        time.perf_counter() - started)
        print(f"    seg {segment.segment_id}  {wav.size / rate:6.2f}s")

    return produced


def measure(worker, produced, segments, anchor):
    rows = []
    by_id = {s.segment_id: s for s in segments}

    for sid, (path, duration, gen_s) in produced.items():
        segment = by_id[sid]
        slot = segment.end_ts - segment.start_ts

        try:
            _, embedding = worker.xtts.get_conditioning_latents(audio_path=[str(path)])
            sim = cosine(anchor, embedding)
        except Exception:
            sim = None

        rows.append({
            "id": sid, "path": path, "dur": duration,
            "ratio": duration / slot if slot else 0.0,
            "cps": len(segment.text) / duration if duration else 0.0,
            "sim": sim, "gen_s": gen_s,
        })

    return sorted(rows, key=lambda r: r["id"])


def main():
    worker = XTTSWorker(BUNDLE)
    worker.load_bundle()
    worker.load_model()

    segments = worker.request.segments
    rate = worker.request.output_sample_rate
    english_ref = BUNDLE / "request" / "reference.wav"

    if not HINDI_REF.exists():
        print(f"!! No Hindi reference at {HINDI_REF}.")
        print("   Upload the sample that sounded good in the desktop app:")
        print("   from google.colab import files; files.upload()")
        print("   then rename it to /content/hindi_ref.wav")
        return

    print(f"english reference {sf.info(str(english_ref)).duration:.2f}s")
    print(f"hindi reference   {sf.info(str(HINDI_REF)).duration:.2f}s")

    # Identity is always scored against the English speaker, because that is
    # whose voice the dub is supposed to be in. Path B should score badly by
    # construction; that is the point of including it.
    _, anchor = worker.xtts.get_conditioning_latents(
        audio_path=[str(english_ref)], gpt_cond_len=60,
        gpt_cond_chunk_len=30, max_ref_length=60, sound_norm_refs=True,
    )

    results = {}

    print("\n--- A: XTTS on the English speaker, in Hindi (what ships)")
    latent, embedding = worker.xtts.get_conditioning_latents(
        audio_path=[str(english_ref)], **CONDITIONING
    )
    a = synth(worker, segments, rate, latent, embedding, OUT_ROOT / "a_english_ref")
    results["A english ref"] = measure(worker, a, segments, anchor)

    print("\n--- B: XTTS on the Hindi donor voice")
    latent, embedding = worker.xtts.get_conditioning_latents(
        audio_path=[str(HINDI_REF)], **CONDITIONING
    )
    b = synth(worker, segments, rate, latent, embedding, OUT_ROOT / "b_hindi_donor")
    results["B hindi donor"] = measure(worker, b, segments, anchor)

    print("\n--- C: B converted onto the English speaker")
    try:
        from TTS.api import TTS as TTSApi

        converter = TTSApi(VC_MODEL)
        converter.to("cuda" if torch.cuda.is_available() else "cpu")

        out_dir = OUT_ROOT / "c_converted"
        out_dir.mkdir(parents=True, exist_ok=True)
        converted = {}

        for sid, (path, duration, _) in b.items():
            target = out_dir / f"seg_{sid:05d}.wav"
            started = time.perf_counter()
            converter.voice_conversion_to_file(
                source_wav=str(path),
                target_wav=str(english_ref),
                file_path=str(target),
            )
            info = sf.info(str(target))
            converted[sid] = (target, info.duration, time.perf_counter() - started)
            print(f"    seg {sid}  {info.duration:6.2f}s")

        results["C converted"] = measure(worker, converted, segments, anchor)
    except Exception:
        print("    FAILED:")
        traceback.print_exc()

    # ------------------------------------------------------------- summary
    print("\n" + "=" * 72)
    print("SIMILARITY TO THE ENGLISH SPEAKER  (who the dub should sound like)")
    print("=" * 72)
    head = f"{'path':<16}" + "".join(f"{'s' + str(s.segment_id):>9}" for s in segments) + f"{'mean':>9}"
    print(head)
    for name, rows in results.items():
        line = f"{name:<16}"
        for r in rows:
            line += f"{r['sim']:>9.3f}" if r["sim"] is not None else f"{'-':>9}"
        sims = [r["sim"] for r in rows if r["sim"] is not None]
        line += f"{np.mean(sims):>9.3f}" if sims else f"{'-':>9}"
        print(line)

    print("\n" + "=" * 72)
    print(f"PACE  (natural Hindi {NATURAL_CPS_HI} cps, ratio 1.00 fits the slot)")
    print("=" * 72)
    print(f"{'path':<16}{'mean ratio':>12}{'mean cps':>10}")
    for name, rows in results.items():
        print(f"{name:<16}"
              f"{np.mean([r['ratio'] for r in rows]):>12.2f}"
              f"{np.mean([r['cps'] for r in rows]):>10.1f}")

    # ------------------------------------------------------------ listening
    display(HTML("<h3>English reference — the voice the dub should use</h3>"))
    display(Audio(filename=str(english_ref)))
    display(HTML("<h3>Hindi donor — the voice that speaks Hindi well</h3>"))
    display(Audio(filename=str(HINDI_REF)))

    by_id = {s.segment_id: s for s in segments}

    for segment in segments:
        display(HTML(
            f"<h3 style='margin:18px 0 4px'>Segment {segment.segment_id}</h3>"
            f"<div style='font-size:15px;margin-bottom:8px'>{segment.text}</div>"
        ))
        for name, rows in results.items():
            row = next((r for r in rows if r["id"] == segment.segment_id), None)
            if row is None:
                continue
            display(HTML(
                f"<div style='margin-top:8px'><b>{name}</b> "
                f"<span style='color:#666'>&mdash; {row['dur']:.2f}s, "
                f"{row['cps']:.1f} cps</span></div>"
            ))
            display(Audio(filename=str(row["path"])))

    print("\nThree questions, in order. Does B sound like natural Hindi? Does C "
          "still sound like natural Hindi? Does C sound like the English "
          "speaker? A yes to all three means the synthesis problem is solved "
          "and the remaining work is timing.")


if __name__ == "__main__":
    main()
