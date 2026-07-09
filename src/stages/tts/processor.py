import json
from pathlib import Path

from src.stages.translation.models import TranslationResult
from src.stages.tts.models import SynthesisRequest, SynthesisSegment


class TTSProcessor:
    def build_request(
        self,
        translation_result: TranslationResult,
        reference_audio: str,
    ) -> SynthesisRequest:
        segments = []
        for segment in translation_result.segments:
            segments.append(
                SynthesisSegment(
                    segment_id=segment.segment_id,
                    chunk_id=segment.chunk_id,
                    start_ts=segment.start_ts,
                    end_ts=segment.end_ts,
                    text=segment.translated_text,
                    reference_audio=reference_audio,
                )
            )

        return SynthesisRequest(
            job_id=translation_result.job_id,
            language=translation_result.target_language,
            segments=segments,
        )

    def export_request(
        self,
        request: SynthesisRequest,
        output_path: Path,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                request.model_dump(),
                f,
                ensure_ascii=False,
                indent=2,
            )

        return output_path
