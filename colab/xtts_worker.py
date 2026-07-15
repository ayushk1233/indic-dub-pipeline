import json
from pathlib import Path

from src.stages.tts.models import SynthesisRequest, SynthesisSegment


class XTTSWorker:
    def __init__(self, bundle_dir: Path):
        self.bundle_dir = bundle_dir
        self.manifest_path = bundle_dir / "manifest.json"
        self.request_path = bundle_dir / "request" / "synthesis_request.json"
        self.output_dir = bundle_dir / "output"
        self.logs_dir = bundle_dir / "logs"

        self.request: SynthesisRequest | None = None
        self.manifest = None
        self.model = None
        self.speaker_embedding = None
        self.gpt_cond_latent = None

    def load_bundle(self) -> None:
        """
        Read the manifest and synthesis request from the bundle.
        """
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
            
        version = self.manifest.get("metadata", {}).get("bundle_version")
        if version != "1.0":
            raise ValueError(f"Unsupported bundle version: {version}")

        if not self.request_path.exists():
            raise FileNotFoundError(f"Request not found: {self.request_path}")

        with open(self.request_path, "r", encoding="utf-8") as f:
            self.request = SynthesisRequest(**json.load(f))

    def load_model(self):
        """
        Load the XTTS-v2 model and compute the speaker embedding.
        """
        if self.model is not None:
            return self.model

        from colab.preflight import PreflightValidator
        
        validator = PreflightValidator()
        diagnostics = validator.run()
        print("Preflight diagnostics:", diagnostics)

        from TTS.api import TTS
        
        print("Loading XTTS-v2...")
        self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
        return self.model

    def compute_speaker_embedding(self):
        """
        Compute GPT conditioning latent and speaker embedding from reference audio.
        """
        if self.speaker_embedding is not None and getattr(self, "gpt_cond_latent", None) is not None:
            return self.gpt_cond_latent, self.speaker_embedding

        if self.model is None:
            self.load_model()

        reference_wav = self.bundle_dir / "request" / "reference.wav"
        if not reference_wav.exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_wav}")

        print("Computing speaker embeddings...")
        xtts_model = self.model.synthesizer.tts_model
        gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(
            audio_path=[str(reference_wav)]
        )

        self.gpt_cond_latent = gpt_cond_latent
        self.speaker_embedding = speaker_embedding

        return self.gpt_cond_latent, self.speaker_embedding

    def synthesize_segment(self, segment: SynthesisSegment) -> Path:
        """
        Synthesize a single audio segment.
        """
        import torch
        import torchaudio

        if self.model is None:
            self.load_model()
            
        if self.gpt_cond_latent is None or self.speaker_embedding is None:
            self.compute_speaker_embedding()

        print(f"Synthesizing segment {segment.segment_id} (chunk {segment.chunk_id})...")
        
        language = self.request.language
        xtts_model = self.model.synthesizer.tts_model
        
        with torch.no_grad():
            out = xtts_model.inference(
                text=segment.text,
                language=language,
                gpt_cond_latent=self.gpt_cond_latent,
                speaker_embedding=self.speaker_embedding,
                temperature=0.7,
            )

        wav = torch.tensor(out["wav"]).unsqueeze(0)
        
        output_filename = f"chunk_{segment.chunk_id:04d}.wav"
        output_path = self.output_dir / output_filename
        
        torchaudio.save(str(output_path), wav, 24000)
        
        return output_path

    def save_segment(self) -> None:
        """
        Save the synthesized segment to disk.
        """
        raise NotImplementedError

    def write_result(self) -> None:
        """
        Write the synthesis_result.json to the output directory.
        """
        raise NotImplementedError

    def run(self) -> None:
        """
        Execute the TTS job.
        """
        print("Loading bundle...")
        self.load_bundle()
        job_id = self.manifest.get("metadata", {}).get("job_id")
        num_segments = len(self.request.get("segments", []))
        print(f"Bundle loaded: Job {job_id} with {num_segments} segments.")
        print("Worker run finished (synthesis deferred).")
