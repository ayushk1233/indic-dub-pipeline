from dataclasses import dataclass

import jiwer

from src.stages.asr.models import TranscriptResult


@dataclass
class ASRMetrics:
    wer: float
    cer: float
    avg_logprob: float
    avg_segment_duration: float


def evaluate_transcript(
    reference: str,
    transcript: TranscriptResult,
) -> ASRMetrics:
    hypothesis = " ".join(
        segment.text.strip()
        for segment in transcript.segments
    )

    wer = jiwer.wer(reference, hypothesis)
    cer = jiwer.cer(reference, hypothesis)

    logprobs = [
        s.avg_logprob
        for s in transcript.segments
        if s.avg_logprob is not None
    ]

    avg_logprob = (
        sum(logprobs) / len(logprobs)
        if logprobs
        else 0.0
    )

    durations = [
        s.end_ts - s.start_ts
        for s in transcript.segments
    ]

    avg_duration = (
        sum(durations) / len(durations)
        if durations
        else 0.0
    )

    return ASRMetrics(
        wer=wer,
        cer=cer,
        avg_logprob=avg_logprob,
        avg_segment_duration=avg_duration,
    )
