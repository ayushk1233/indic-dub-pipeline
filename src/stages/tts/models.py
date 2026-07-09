from pydantic import BaseModel, Field


class SynthesisSegment(BaseModel):
    """
    One translated segment to synthesize.
    """

    segment_id: int
    chunk_id: int

    start_ts: float
    end_ts: float

    text: str

    reference_audio: str


class SynthesisRequest(BaseModel):
    """
    Complete XTTS synthesis request.
    """

    job_id: str

    language: str

    output_sample_rate: int = 24000

    segments: list[SynthesisSegment] = Field(default_factory=list)


class SynthesizedSegment(BaseModel):
    """
    Output produced by XTTS.
    """

    segment_id: int
    chunk_id: int

    audio_path: str

    duration: float


class SynthesisResult(BaseModel):
    """
    Complete synthesis output.
    """

    job_id: str

    sample_rate: int

    segments: list[SynthesizedSegment] = Field(default_factory=list)
