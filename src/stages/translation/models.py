from pydantic import BaseModel, Field


class TranslatedSegment(BaseModel):
    """
    A translated transcript segment that preserves timing alignment.
    """

    segment_id: int
    chunk_id: int

    start_ts: float
    end_ts: float

    source_text: str
    translated_text: str

    source_language: str
    target_language: str


class TranslationResult(BaseModel):
    """
    Complete translation output for a transcription job.
    """

    job_id: str

    source_language: str
    target_language: str

    segments: list[TranslatedSegment] = Field(default_factory=list)
