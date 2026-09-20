from dataclasses import dataclass, field

import jiwer

from src.stages.asr.models import TranscriptResult


# Thresholds for the classic Whisper hallucination signature. All three fields
# are already carried on TranscriptSegment; nothing read them before.
NO_SPEECH_CEILING = 0.6
LOGPROB_FLOOR = -1.0
COMPRESSION_CEILING = 2.4


@dataclass
class ASRMetrics:
    """
    Transcription quality. Word and character error rate need a reference
    transcript; everything else is reference-free and always available.
    """

    wer: float | None
    cer: float | None
    avg_logprob: float
    avg_segment_duration: float

    num_segments: int = 0
    language: str = "unknown"
    language_probability: float = 0.0

    total_speech_s: float = 0.0

    # Source speaking rate in characters per second. This is the baseline the
    # translation feasibility check and the synthesis pace check measure against.
    source_cps: float = 0.0

    mean_no_speech_prob: float = 0.0
    max_no_speech_prob: float = 0.0
    mean_compression_ratio: float = 0.0
    max_compression_ratio: float = 0.0

    num_low_confidence: int = 0
    num_high_no_speech: int = 0
    num_repetitive: int = 0
    num_empty: int = 0

    suspect_segments: list[int] = field(default_factory=list)

    @property
    def pct_suspect(self) -> float:
        if not self.num_segments:
            return 0.0
        return 100.0 * len(self.suspect_segments) / self.num_segments


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_transcript(
    reference: str | None = None,
    transcript: TranscriptResult | None = None,
) -> ASRMetrics:
    """
    Evaluate a transcript, with or without a ground-truth reference.

    Pass reference=None to get the reference-free signals only.
    """
    if transcript is None:
        raise ValueError("transcript is required")

    segments = transcript.segments

    hypothesis = " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()

    wer = cer = None

    if reference and reference.strip() and hypothesis:
        wer = jiwer.wer(reference, hypothesis)
        cer = jiwer.cer(reference, hypothesis)

    logprobs = [s.avg_logprob for s in segments if s.avg_logprob is not None]
    no_speech = [s.no_speech_prob for s in segments if s.no_speech_prob is not None]
    compression = [
        s.compression_ratio for s in segments if s.compression_ratio is not None
    ]

    durations = [s.end_ts - s.start_ts for s in segments]
    total_speech = sum(durations)
    total_chars = sum(len(s.text.strip()) for s in segments)

    suspect = sorted(
        {
            s.segment_id
            for s in segments
            if (s.no_speech_prob is not None and s.no_speech_prob > NO_SPEECH_CEILING)
            or (s.avg_logprob is not None and s.avg_logprob < LOGPROB_FLOOR)
            or (
                s.compression_ratio is not None
                and s.compression_ratio > COMPRESSION_CEILING
            )
            or not s.text.strip()
        }
    )

    return ASRMetrics(
        wer=wer,
        cer=cer,
        avg_logprob=_mean(logprobs),
        avg_segment_duration=_mean(durations),
        num_segments=len(segments),
        language=transcript.language,
        language_probability=transcript.language_probability,
        total_speech_s=total_speech,
        source_cps=total_chars / total_speech if total_speech > 0 else 0.0,
        mean_no_speech_prob=_mean(no_speech),
        max_no_speech_prob=max(no_speech) if no_speech else 0.0,
        mean_compression_ratio=_mean(compression),
        max_compression_ratio=max(compression) if compression else 0.0,
        num_low_confidence=sum(
            1 for s in segments if s.avg_logprob is not None and s.avg_logprob < LOGPROB_FLOOR
        ),
        num_high_no_speech=sum(
            1
            for s in segments
            if s.no_speech_prob is not None and s.no_speech_prob > NO_SPEECH_CEILING
        ),
        num_repetitive=sum(
            1
            for s in segments
            if s.compression_ratio is not None
            and s.compression_ratio > COMPRESSION_CEILING
        ),
        num_empty=sum(1 for s in segments if not s.text.strip()),
        suspect_segments=suspect,
    )
