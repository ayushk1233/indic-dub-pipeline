"""
Fill a bundle's output with placeholder audio, so the second half of the
pipeline can be tested without spending GPU time.

Synthesis is the only stage that needs a GPU and the only one that needs a
human to judge the result. Everything after it — import, the fitting cascade,
assembly, remux, the report — is deterministic arithmetic about durations, and
all of it can be exercised with tones of a known length.

By default each segment gets exactly as much audio as a native speaker would
need for its text, using the measured rate for the target language. That makes
the run honest rather than flattering: if the timeline still cannot absorb the
result, the problem is the translation length, not the synthesis.

    ./venv/bin/python -m scripts.stub_synthesis --bundle artifacts/demo/tts_bundle

Use --pace to simulate a model that runs slow or fast: 1.35 emits 35% more
audio than natural, which is roughly what an untuned XTTS produced.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.eval.translation_metrics import natural_cps
from src.stages.assemble import write_wav_samples
from src.stages.tts.models import (
    SynthesisRequest,
    SynthesisResult,
    SynthesizedSegment,
)


TONE_HZ = 220.0
AMPLITUDE = 0.3


def stub_bundle(
    bundle_dir: Path,
    pace: float = 1.0,
    trailing_silence_s: float = 0.0,
    skip: set[int] | None = None,
) -> SynthesisResult:
    """
    Write one tone per requested segment plus the result file the GPU worker
    would have produced.

    `skip` leaves those segment ids out entirely, which is how a timed-out
    partial run looks and is worth testing deliberately.
    """
    bundle_dir = Path(bundle_dir)
    request_path = bundle_dir / "request" / "synthesis_request.json"

    if not request_path.exists():
        raise FileNotFoundError(
            f"No synthesis request at {request_path}. Export a bundle first."
        )

    with open(request_path, "r", encoding="utf-8") as f:
        request = SynthesisRequest(**json.load(f))

    output_dir = bundle_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    rate = request.output_sample_rate
    skip = skip or set()
    produced: list[SynthesizedSegment] = []

    for segment in request.segments:
        if segment.segment_id in skip:
            continue

        natural = len(segment.text.strip()) / natural_cps(request.language)
        duration = max(natural * pace, 0.2)

        count = int(duration * rate)
        t = np.arange(count) / rate
        samples = (AMPLITUDE * np.sin(2 * np.pi * TONE_HZ * t)).astype(np.float32)

        if trailing_silence_s > 0:
            silence = np.zeros(int(trailing_silence_s * rate), dtype=np.float32)
            samples = np.concatenate([samples, silence])

        name = f"seg_{segment.segment_id:05d}.wav"
        write_wav_samples(output_dir / name, samples, rate)

        produced.append(
            SynthesizedSegment(
                segment_id=segment.segment_id,
                chunk_id=segment.chunk_id,
                audio_path=f"output/{name}",
                duration=samples.size / rate,
                num_samples=samples.size,
            )
        )

    result = SynthesisResult(
        job_id=request.job_id,
        sample_rate=rate,
        model_id="stub",
        params={"stub": True, "pace": pace, "trailing_silence_s": trailing_silence_s},
        segments=produced,
    )

    with open(output_dir / "synthesis_result.json", "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write placeholder synthesis output into a bundle."
    )
    parser.add_argument("--bundle", required=True, help="Bundle directory.")
    parser.add_argument(
        "--pace",
        type=float,
        default=1.0,
        help=(
            "Multiple of the natural speaking rate to emit. 1.0 is what a "
            "native speaker needs; 1.35 simulates a model that runs long."
        ),
    )
    parser.add_argument(
        "--trailing-silence",
        type=float,
        default=0.0,
        help="Seconds of silence to append, to exercise the trimming tier.",
    )
    parser.add_argument(
        "--skip",
        type=int,
        nargs="*",
        default=None,
        help="Segment ids to leave out, simulating a partial run.",
    )
    args = parser.parse_args()

    result = stub_bundle(
        Path(args.bundle),
        pace=args.pace,
        trailing_silence_s=args.trailing_silence,
        skip=set(args.skip) if args.skip else None,
    )

    print(
        f"Stubbed {len(result.segments)} segments at {result.sample_rate} Hz "
        f"into {args.bundle}/output"
    )


if __name__ == "__main__":
    main()
