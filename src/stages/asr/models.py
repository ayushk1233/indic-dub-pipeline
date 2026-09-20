from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """
    A single transcription segment returned by the ASR engine.
    """

    segment_id: int

    chunk_id: int

    start_ts: float

    end_ts: float

    text: str

    avg_logprob: float | None = None

    no_speech_prob: float | None = None

    compression_ratio: float | None = None


class TranscriptResult(BaseModel):
    """
    Complete transcription result for a preprocessing job.
    """

    job_id: str

    language: str

    language_probability: float

    segments: list[TranscriptSegment] = Field(default_factory=list)
