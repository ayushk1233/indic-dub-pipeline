from dataclasses import dataclass

from src.stages.translation.models import TranslationResult


@dataclass
class TranslationMetrics:
    num_segments: int
    avg_source_length: float
    avg_target_length: float
    expansion_ratio: float


def evaluate_translation(
    translation: TranslationResult,
) -> TranslationMetrics:
    num_segments = len(translation.segments)

    if num_segments == 0:
        return TranslationMetrics(
            num_segments=0,
            avg_source_length=0.0,
            avg_target_length=0.0,
            expansion_ratio=1.0,
        )

    total_source_chars = sum(len(s.source_text) for s in translation.segments)
    total_target_chars = sum(len(s.translated_text) for s in translation.segments)

    avg_source_length = total_source_chars / num_segments
    avg_target_length = total_target_chars / num_segments

    expansion_ratio = (
        total_target_chars / total_source_chars
        if total_source_chars > 0
        else 1.0
    )

    return TranslationMetrics(
        num_segments=num_segments,
        avg_source_length=avg_source_length,
        avg_target_length=avg_target_length,
        expansion_ratio=expansion_ratio,
    )
