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


MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"

SUPPORTED_BUNDLE_VERSIONS = {"1.0", "1.1"}

# Conditioning settings established in colab/xtts.md. These belong to
# get_conditioning_latents(); passing them to inference() silently does nothing.
CONDITIONING_PARAMS = {
    "gpt_cond_len": 8,
    "gpt_cond_chunk_len": 4,
    "max_ref_length": 10,
}

# Decoder settings established in colab/xtts.md. Greedy decoding is what makes
# output duration reproducible; sampling was the root cause of the 4s/7s/14s
# variation on identical input. temperature, top_k and top_p are inert under
# do_sample=False and must not be passed.
INFERENCE_PARAMS = {
    "do_sample": False,
    "repetition_penalty": 5.0,
    "enable_text_splitting": False,
}


class XTTSWorker:
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
        self.speaker_embedding = None
        self.gpt_cond_latent = None

        self.synthesized: list[SynthesizedSegment] = []

    def load_bundle(self) -> None:
        """
        Read the manifest and synthesis request from the bundle.
        """
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        version = self.manifest.get("metadata", {}).get("bundle_version")
        if version not in SUPPORTED_BUNDLE_VERSIONS:
            raise ValueError(
                f"Unsupported bundle version: {version}. "
                f"Supported: {sorted(SUPPORTED_BUNDLE_VERSIONS)}"
            )

        if not self.request_path.exists():
            raise FileNotFoundError(f"Request not found: {self.request_path}")

        with open(self.request_path, "r", encoding="utf-8") as f:
            self.request = SynthesisRequest(**json.load(f))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self):
        """
        Load the XTTS-v2 model.
        """
        if self.model is not None:
            return self.model

        from colab.preflight import PreflightValidator

        diagnostics = PreflightValidator().run()
        print("Preflight diagnostics:", diagnostics)

        from TTS.api import TTS

        print("Loading XTTS-v2...")
        self.model = TTS(MODEL_ID).to("cuda")
        return self.model

    @property
    def xtts(self):
        if self.model is None:
            self.load_model()
        return self.model.synthesizer.tts_model

    def compute_speaker_embedding(self):
        """
        Compute GPT conditioning latent and speaker embedding from reference audio.
        """
        if self.gpt_cond_latent is not None and self.speaker_embedding is not None:
            return self.gpt_cond_latent, self.speaker_embedding

        if not self.reference_path.exists():
            raise FileNotFoundError(
                f"Reference audio not found: {self.reference_path}"
            )

        print("Computing speaker embeddings...")
        gpt_cond_latent, speaker_embedding = self.xtts.get_conditioning_latents(
            audio_path=[str(self.reference_path)],
            **CONDITIONING_PARAMS,
        )

        self.gpt_cond_latent = gpt_cond_latent
        self.speaker_embedding = speaker_embedding

        return self.gpt_cond_latent, self.speaker_embedding

    def _speaker_similarity(self, wav_path: Path) -> float | None:
        """
        Cosine similarity between the reference speaker embedding and one
        recomputed from synthesized audio, using XTTS's own speaker encoder.

        This is the voice-cloning metric. Returns None if the clip is too
        short or the encoder rejects it, which must not fail the run.
        """
        import torch

        if self.speaker_embedding is None:
            return None

        try:
            _, embedding = self.xtts.get_conditioning_latents(
                audio_path=[str(wav_path)],
            )
        except Exception:
            return None

        reference = self.speaker_embedding.reshape(1, -1).float()
        candidate = embedding.reshape(1, -1).float().to(reference.device)

        return float(
            torch.nn.functional.cosine_similarity(reference, candidate).item()
        )

    def synthesize_segment(self, segment: SynthesisSegment) -> SynthesizedSegment:
        """
        Synthesize a single segment and return its typed result.
        """
        import torch
        import torchaudio

        self.compute_speaker_embedding()

        sample_rate = self.request.output_sample_rate

        print(
            f"Synthesizing segment {segment.segment_id} "
            f"(chunk {segment.chunk_id}, {len(segment.text)} chars)..."
        )

        started = time.perf_counter()

        with torch.no_grad():
            out = self.xtts.inference(
                text=segment.text,
                language=self.request.language,
                gpt_cond_latent=self.gpt_cond_latent,
                speaker_embedding=self.speaker_embedding,
                **INFERENCE_PARAMS,
            )

        wav = torch.tensor(out["wav"]).unsqueeze(0)
        num_samples = wav.shape[-1]
        duration = num_samples / sample_rate

        gpt_latents = out.get("gpt_latents")
        gpt_tokens = int(gpt_latents.shape[1]) if gpt_latents is not None else None

        # segment_id is globally unique; chunk_id is shared by every segment
        # cut from the same source chunk and would overwrite siblings.
        output_path = self.output_dir / f"seg_{segment.segment_id:05d}.wav"
        torchaudio.save(str(output_path), wav, sample_rate)

        similarity = self._speaker_similarity(output_path)

        slot = segment.end_ts - segment.start_ts
        elapsed = time.perf_counter() - started

        print(
            f"  {duration:.2f}s audio for a {slot:.2f}s slot "
            f"(ratio {duration / slot:.2f}), {gpt_tokens} tokens, "
            f"similarity {similarity if similarity is None else round(similarity, 3)}, "
            f"{elapsed:.1f}s on GPU"
        )

        return SynthesizedSegment(
            segment_id=segment.segment_id,
            chunk_id=segment.chunk_id,
            audio_path=str(output_path.relative_to(self.bundle_dir)),
            duration=duration,
            num_samples=num_samples,
            status="done",
            gpt_tokens=gpt_tokens,
            speaker_similarity=similarity,
        )

    def build_result(self) -> SynthesisResult:
        return SynthesisResult(
            job_id=self.request.job_id,
            sample_rate=self.request.output_sample_rate,
            model_id=MODEL_ID,
            params={**CONDITIONING_PARAMS, **INFERENCE_PARAMS},
            segments=sorted(self.synthesized, key=lambda s: s.segment_id),
        )

    def write_result(self) -> Path:
        """
        Write synthesis_result.json to the bundle's output directory.
        """
        result = self.build_result()

        with open(self.result_path, "w", encoding="utf-8") as f:
            json.dump(
                result.model_dump(),
                f,
                ensure_ascii=False,
                indent=2,
            )

        return self.result_path

    def run(self) -> SynthesisResult:
        """
        Execute the whole TTS job.

        Results are written after every segment so that a Colab timeout or a
        single bad segment still leaves salvageable work on disk.
        """
        print("Loading bundle...")
        self.load_bundle()

        job_id = self.manifest.get("metadata", {}).get("job_id")
        segments = self.request.segments

        print(f"Bundle loaded: job {job_id}, {len(segments)} segments.")
        print(f"Conditioning: {CONDITIONING_PARAMS}")
        print(f"Inference:    {INFERENCE_PARAMS}")

        self.load_model()
        self.compute_speaker_embedding()

        self.synthesized = []

        for segment in segments:
            try:
                self.synthesized.append(self.synthesize_segment(segment))
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

    parser = argparse.ArgumentParser(description="Run an XTTS synthesis bundle.")
    parser.add_argument(
        "--bundle",
        default="/content/tts_bundle",
        help="Path to the extracted bundle directory.",
    )
    args = parser.parse_args()

    XTTSWorker(Path(args.bundle)).run()
