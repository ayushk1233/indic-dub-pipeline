from src.stages.asr.models import TranscriptResult
from src.stages.translation.backend import TranslationBackend
from src.stages.translation.models import TranslatedSegment, TranslationResult


class TranslationProcessor:
    def __init__(
        self,
        backend: TranslationBackend,
    ):
        self.backend = backend

    def translate(
        self,
        transcript: TranscriptResult,
        target_language: str,
    ) -> TranslationResult:
        segments = []
        for segment in transcript.segments:
            translated_text = self.backend.translate(
                segment.text,
                transcript.language,
                target_language,
            )
            segments.append(
                TranslatedSegment(
                    segment_id=segment.segment_id,
                    chunk_id=segment.chunk_id,
                    start_ts=segment.start_ts,
                    end_ts=segment.end_ts,
                    source_text=segment.text,
                    translated_text=translated_text,
                    source_language=transcript.language,
                    target_language=target_language,
                )
            )

        return TranslationResult(
            job_id=transcript.job_id,
            source_language=transcript.language,
            target_language=target_language,
            segments=segments,
        )
