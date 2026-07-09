from pydantic import BaseModel, Field


class AudioChunk(BaseModel):
    """
    Represents one speech chunk produced during preprocessing.
    """

    chunk_id: int

    chunk_path: str

    start_ts: float

    end_ts: float

    duration: float

    status: str = "pending"

    speaker: str | None = None

    language: str | None = None


class ChunkManifest(BaseModel):
    """
    Structured manifest shared across pipeline stages.
    """

    job_id: str

    source_file: str

    audio_file: str

    sample_rate: int

    channels: int

    chunks: list[AudioChunk] = Field(default_factory=list)
