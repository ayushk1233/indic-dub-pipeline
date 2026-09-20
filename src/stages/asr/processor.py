import json
from pathlib import Path

from src.stages.asr.backend import InferenceBackend
from src.stages.asr.models import TranscriptResult, TranscriptSegment


class ASRProcessor:
    def __init__(self, backend: InferenceBackend):
        self.backend = backend

    def transcribe(self, manifest_path: Path) -> TranscriptResult:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        job_id = manifest_path.parent.name
        
        language = "unknown"
        language_probability = 0.0
        
        transcript_segments = []
        segment_id = 0
        
        for i, chunk_data in enumerate(manifest):
            chunk_path = chunk_data["chunk_path"]
            chunk_start = float(chunk_data["start_ts"])
            
            # Run inference
            segments, info = self.backend.transcribe_audio(chunk_path)
            
            if i == 0:
                language = info.language
                language_probability = info.language_probability
                
            for seg in segments:
                transcript_segments.append(
                    TranscriptSegment(
                        segment_id=segment_id,
                        chunk_id=i,
                        start_ts=chunk_start + seg.start,
                        end_ts=chunk_start + seg.end,
                        text=seg.text,
                        avg_logprob=seg.avg_logprob,
                        no_speech_prob=seg.no_speech_prob,
                        compression_ratio=seg.compression_ratio,
                    )
                )
                segment_id += 1
                
        return TranscriptResult(
            job_id=job_id,
            language=language,
            language_probability=language_probability,
            segments=transcript_segments,
        )
