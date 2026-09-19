"""
Run a synthesis bundle through IndicF5 on a GPU host.

The XTTS counterpart is `colab/xtts_worker.py` and the two implement the same
contract — read a bundle, write `output/synthesis_result.json` after every
segment — so the local side does not know or care which one ran. What differs
is everything model-specific, and on this model the differences are not
cosmetic:

**The reference transcript is mandatory.** XTTS clones from audio alone;
IndicF5 conditions on `ref_audio + ref_text` as one sequence and cannot clone
without the text. A bundle that carries no `reference_text` is refused here
rather than synthesized badly, which is why bundle 1.0 is not accepted.

**The transcript is normalised before use.** Lowercased, punctuation stripped
— FINDINGS §5c measured this taking the leading prefix from 1 of 7 clips to 0
of 7. Punctuation the speaker did not pause for is text the model must place
somewhere, and what does not fit the conditioned frames is spoken at the start
of the kept region. This is model policy, so it lives behind the backend
boundary and not in `src/`.

**Latin loanwords are folded to their conventional Devanagari spelling.**
Hindi ASR writes English loanwords in Latin — `project`, `fit` — and IndicF5
cannot generate English (FINDINGS §4), so a Latin word inside a Devanagari
sentence is a word it cannot say. `fixtures/loanwords_hi.json` holds the
conventional Hindi spelling of each, which is the form IndicF5's training text
actually contains. Anything Latin that survives the fold is warned about
rather than silently synthesized, because that is the case the table does not
cover and a listener will hear.

**Duration is set per segment, not left to the model.** FINDINGS §5a: IndicF5
allocates duration from the reference's UTF-8 byte ratio, which is wrong by
2.46x when a Latin transcript describes a clip that will generate Devanagari.
`fix_duration` sets *total* frames including the reference, so it is passed as
`reference_seconds + slot_seconds`, and it requires a single chunk because a
chunked generation prepends the reference to every chunk.

Neither lever is exposed by the model's `__call__`, so both are applied by
`colab.indicf5_diagnose.install_patches()`, which wraps `chunk_text` and
`infer_batch_process` as module globals. That module refuses to report if no
call was intercepted, and this worker refuses to synthesize for the same
reason: an uninstrumented run silently ignores `fix_duration` and returns
whatever length the byte ratio asked for.

Run it:

    python -m colab.indicf5_worker --bundle /kaggle/working/tts_bundle
"""

import json
import time
import traceback
from pathlib import Path

from src.stages.tts.models import (
    SynthesisRequest,
    SynthesisResult,
    SynthesisSegment,
    SynthesizedSegment,
)

MODEL_ID = "ai4bharat/IndicF5"

# 1.0 has no `reference_text` field at all. XTTS can run such a bundle;
# IndicF5 cannot clone from audio alone, so accepting one would mean
# synthesizing in some other voice and reporting success.
SUPPORTED_BUNDLE_VERSIONS = {"1.1", "1.2"}

# IndicF5 generates at 24 kHz through Vocos. The bundle asks for a rate and
# the two have always agreed; if they ever stop, resampling belongs here
# rather than silently mislabelling the file's duration.
MODEL_SAMPLE_RATE = 24000

# FINDINGS §16a: a single-chunk generation is clean to 19.7 s / 240 Devanagari
# characters and breaks by 21.2 s — the transcript runs on past the sentence
# and repeats. `fix_duration` forces one chunk, so a slot past this cap cannot
# be served the fast way. Real pipeline segments have a median of 2.76 s and a
# measured maximum of 7.78 s, so this is a guard rather than a code path.
MAX_FIXED_SPAN_S = 20.0

# §5d, and the reason src.stages.reference.REFERENCE_SECONDS exists:
# `preprocess_ref_audio_text` clips reference audio past this and never
# truncates `ref_text` to match. A longer clip is not an error the worker can
# fix — the transcript describing it is already wrong — so it warns loudly.
REFERENCE_CLIP_THRESHOLD_S = 15.0


class InstrumentationError(RuntimeError):
    """
    The `fix_duration` / `one_chunk` levers were not applied.

    Separate from every other failure because it must abort the run rather
    than fail one segment: nothing downstream of it is worth computing, and
    the per-segment handler would otherwise burn the whole bundle producing
    clips whose length came from the byte formula (FINDINGS §5a).
    """


class IndicF5Worker:
    def __init__(self, bundle_dir: Path):
        self.bundle_dir = Path(bundle_dir)
        self.manifest_path = self.bundle_dir / "manifest.json"
        self.request_path = self.bundle_dir / "request" / "synthesis_request.json"
        self.reference_path = self.bundle_dir / "request" / "reference.wav"
        self.output_dir = self.bundle_dir / "output"
        self.logs_dir = self.bundle_dir / "logs"
        self.result_path = self.output_dir / "synthesis_result.json"

        self.request: SynthesisRequest | None = None
        self.manifest = None
        self.model = None

        self.reference_seconds: float | None = None
        self.reference_text: str | None = None

        self.synthesized: list[SynthesizedSegment] = []

    # -- bundle --------------------------------------------------------------

    def load_bundle(self) -> None:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        version = self.manifest.get("metadata", {}).get("bundle_version")
        if version not in SUPPORTED_BUNDLE_VERSIONS:
            raise ValueError(
                f"Unsupported bundle version: {version}. "
                f"Supported: {sorted(SUPPORTED_BUNDLE_VERSIONS)}. "
                "IndicF5 conditions on the reference transcript and cannot "
                "clone without it; bundle 1.0 does not carry one."
            )

        if not self.request_path.exists():
            raise FileNotFoundError(f"Request not found: {self.request_path}")

        with open(self.request_path, "r", encoding="utf-8") as f:
            self.request = SynthesisRequest(**json.load(f))

        if not self.reference_path.exists():
            raise FileNotFoundError(
                f"Reference audio not found: {self.reference_path}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def prepare_reference(self) -> tuple[float, str]:
        """
        Measure the reference clip and normalise its transcript.

        Both are needed before the first segment: the duration goes into every
        `fix_duration`, and an empty transcript means the run cannot proceed
        at all.
        """
        import soundfile as sf

        from colab.indicf5_english import plain

        raw = (self.request.reference_text or "").strip()
        if not raw:
            raise ValueError(
                "The bundle carries no reference_text. IndicF5 conditions on "
                "reference audio and its transcript together and cannot clone "
                "from audio alone (FINDINGS §4). Re-export the bundle from a "
                "job whose transcript stage has run."
            )

        info = sf.info(str(self.reference_path))
        self.reference_seconds = float(info.duration)
        self.reference_text = plain(raw)

        if self.reference_seconds > REFERENCE_CLIP_THRESHOLD_S:
            print(
                f"  !! reference is {self.reference_seconds:.1f}s. IndicF5 "
                f"clips it to {REFERENCE_CLIP_THRESHOLD_S:.0f}s and does NOT "
                "shorten the transcript to match (FINDINGS §5d), so the text "
                "now describes audio the model cannot hear. Expect the "
                "gibberish of §5. Rebuild with "
                "src.stages.reference.reference_seconds('ai4bharat/indicf5')."
            )

        print(f"  reference   {self.reference_seconds:.2f}s, "
              f"{len(self.reference_text)} chars after normalisation")
        print(f"  ref_text    {self.reference_text[:96]}")

        return self.reference_seconds, self.reference_text

    def prepare_text(self, text: str) -> str:
        """
        The Devanagari the model will actually be asked to say.

        Only a loanword fold, and deliberately only that: a 13-entry table of
        English words whose Hindi spelling is conventional and was checked
        against the corpus. It is not a transliterator and must not become
        one — see src/text/loanwords.py. A Latin word outside the table is
        left alone and reported, because guessing its Devanagari spelling is
        exactly the drafting judgement §4f says belongs to a human.
        """
        from src.text.loanwords import fold_loanwords, latin_words

        folded = fold_loanwords(text or "", "hi")
        left = latin_words(folded)

        if left:
            print(f"  !! Latin words IndicF5 cannot say, and not in the "
                  f"loanword table: {left}. Add them to "
                  f"fixtures/loanwords_hi.json or expect nonsense here.")

        return folded

    # -- model ---------------------------------------------------------------

    def load_model(self):
        if self.model is not None:
            return self.model

        from colab.indicf5_check import load_indicf5
        from colab.indicf5_diagnose import install_patches, reset_mode

        # _mode persists across calls in a live kernel, so a previous run's
        # fix_duration is still sitting there. FINDINGS §13a: skipping this
        # produced 24 clips at a frozen 3.82s whatever the text.
        reset_mode()
        install_patches()

        print(f"Loading {MODEL_ID}...")
        self.model = load_indicf5()
        return self.model

    # -- synthesis -----------------------------------------------------------

    def synthesize_segment(self, segment: SynthesisSegment) -> SynthesizedSegment:
        import numpy as np
        import soundfile as sf
        import torch

        from colab.indicf5_diagnose import _calls, _mode

        slot = float(segment.end_ts - segment.start_ts)

        # Ask for the span the text was CHOSEN against, not the bare slot.
        # Length control picks a candidate that fits `budget_s`, which pools
        # the pause after the segment, and the assembly cascade is built to
        # absorb the overhang. Forcing the clip into `slot` would compress it
        # by budget/slot on top of whatever overflow the text already has —
        # on english.mov that is up to 1.17x of extra compression for nothing.
        # 1.1 bundles carry no budget, so they fall back to the slot.
        span = float(segment.budget_s) if segment.budget_s else slot

        sample_rate = self.request.output_sample_rate

        # fix_duration sets TOTAL frames, reference included (FINDINGS §5a).
        fixed = span <= MAX_FIXED_SPAN_S
        _mode["speed"] = None
        _mode["one_chunk"] = fixed
        _mode["fix_duration"] = (
            self.reference_seconds + span if fixed else None
        )
        _calls.clear()

        if not fixed:
            print(
                f"  segment {segment.segment_id} asks for {span:.1f}s, past "
                f"the {MAX_FIXED_SPAN_S:.0f}s single-chunk cap (§16a). "
                "Falling back to the model's own chunking, which will not "
                "hit the slot."
            )

        text = self.prepare_text(segment.text)

        print(
            f"Synthesizing segment {segment.segment_id} "
            f"(chunk {segment.chunk_id}, {len(text)} chars, "
            f"{slot:.2f}s slot, {span:.2f}s budget)..."
        )

        started = time.perf_counter()

        with torch.no_grad():
            audio = self.model(
                text,
                ref_audio_path=str(self.reference_path),
                ref_text=self.reference_text,
            )

        wav = np.asarray(audio, dtype=np.float32).reshape(-1)

        # IndicF5 returns int16-scaled floats on some paths and [-1, 1] on
        # others. Writing the former as float32 produces silence at best; a
        # peak well past unity is the tell.
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if peak > 1.0:
            wav = wav / 32768.0
            peak = float(np.max(np.abs(wav)))

        num_samples = int(wav.shape[-1])
        duration = num_samples / MODEL_SAMPLE_RATE

        if not _calls:
            raise InstrumentationError(
                "No call was intercepted, so fix_duration and one_chunk were "
                "ignored and this clip's length is whatever the byte ratio "
                "asked for (FINDINGS §5a, §13a). The instrumentation is not "
                "bound to the live f5_tts. RESTART THE RUNTIME and rerun; a "
                "re-import cannot repair it."
            )

        call = _calls[-1]
        requested = call.get("fix_duration")
        # The span the sampler was given, reference frames already subtracted.
        # This is the number that should equal the slot; `fix_duration` is it
        # plus the reference, and confusing the two is FINDINGS §5a's trap.
        generated_span = call.get("requested_s")

        output_path = self.output_dir / f"seg_{segment.segment_id:05d}.wav"
        # PCM_16 because the local side reads these with the stdlib `wave`
        # module, which handles integer PCM only.
        sf.write(str(output_path), wav, sample_rate, subtype="PCM_16")

        elapsed = time.perf_counter() - started
        ratio = duration / span if span else float("nan")

        print(
            f"  {duration:.2f}s audio for a {span:.2f}s budget "
            f"(ratio {ratio:.2f}), sampler span "
            f"{'-' if generated_span is None else f'{generated_span:.2f}s'}, "
            f"fix_duration "
            f"{'-' if requested is None else f'{requested:.2f}s'}, "
            f"peak {peak:.2f}, {elapsed:.1f}s on GPU"
        )

        return SynthesizedSegment(
            segment_id=segment.segment_id,
            chunk_id=segment.chunk_id,
            audio_path=str(output_path.relative_to(self.bundle_dir)),
            duration=duration,
            num_samples=num_samples,
            status="done",
            # IndicF5 has no GPT decoder and exposes no speaker encoder of its
            # own. Identity is genuinely unmeasured in the pipeline right now
            # (FINDINGS §17) and a fabricated number would be worse than none.
            gpt_tokens=None,
            speaker_similarity=None,
        )

    # -- result --------------------------------------------------------------

    def build_result(self) -> SynthesisResult:
        return SynthesisResult(
            job_id=self.request.job_id,
            sample_rate=self.request.output_sample_rate,
            model_id=MODEL_ID,
            params={
                "one_chunk": True,
                "fix_duration": "reference_seconds + slot_seconds",
                "reference_seconds": self.reference_seconds,
                "reference_text_normalization": "plain",
                "generated_text_normalization": "fold_loanwords",
                "max_fixed_span_s": MAX_FIXED_SPAN_S,
            },
            segments=sorted(self.synthesized, key=lambda s: s.segment_id),
        )

    def write_result(self) -> Path:
        result = self.build_result()

        with open(self.result_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)

        return self.result_path

    def run(self) -> SynthesisResult:
        """
        Execute the whole TTS job.

        Results are written after every segment so that a session timeout or
        one bad segment still leaves salvageable work on disk.
        """
        print("Loading bundle...")
        self.load_bundle()

        job_id = self.manifest.get("metadata", {}).get("job_id")
        segments = self.request.segments

        print(f"Bundle loaded: job {job_id}, {len(segments)} segments.")

        self.prepare_reference()
        self.load_model()

        self.synthesized = []

        for segment in segments:
            try:
                self.synthesized.append(self.synthesize_segment(segment))
            except InstrumentationError:
                self.write_result()
                raise
            except Exception as exc:
                print(f"  FAILED segment {segment.segment_id}: {exc}")

                with open(self.logs_dir / "errors.log", "a", encoding="utf-8") as f:
                    f.write(f"segment {segment.segment_id}: {exc}\n")
                    f.write(traceback.format_exc())
                    f.write("\n")

                self.synthesized.append(
                    SynthesizedSegment(
                        segment_id=segment.segment_id,
                        chunk_id=segment.chunk_id,
                        audio_path="",
                        duration=0.0,
                        status="failed",
                        error=repr(exc),
                    )
                )

            self.write_result()

        done = sum(1 for s in self.synthesized if s.status == "done")
        print(f"\nSynthesis complete: {done}/{len(segments)} segments.")
        print(f"Result written to {self.result_path}")

        return self.build_result()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run an IndicF5 synthesis bundle.")
    parser.add_argument(
        "--bundle",
        default="/kaggle/working/tts_bundle",
        help="Path to the extracted bundle directory.",
    )
    args = parser.parse_args()

    IndicF5Worker(Path(args.bundle)).run()
