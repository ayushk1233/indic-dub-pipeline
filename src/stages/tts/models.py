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

    # How long this segment may take to speak, pooling the pause that
    # follows it (src.eval.translation_metrics.speaking_budgets). This, not
    # end_ts - start_ts, is what length control chose the text against, and
    # so it is what synthesis must be asked for: forcing the clip into the
    # bare slot over-compresses it and leaves the assembly cascade's stretch
    # and drift tiers nothing to absorb. Optional so bundle 1.1 still parses.
    budget_s: float | None = None


class SynthesisRequest(BaseModel):
    """
    Complete the earlier baseline synthesis request.
    """

    job_id: str

    language: str

    output_sample_rate: int = 24000

    # Transcript of request/reference.wav, in the source language. the earlier baseline does
    # not need it; IndicF5 conditions on reference audio and its transcript
    # together and cannot clone without it. Optional so that bundle 1.0
    # requests still parse.
    reference_text: str | None = None

    segments: list[SynthesisSegment] = Field(default_factory=list)


class SynthesizedSegment(BaseModel):
    """
    Output produced by the earlier baseline.

    Paths are stored relative to the bundle root, because the file is
    written on the GPU host and read somewhere else entirely.
    """

    segment_id: int
    chunk_id: int

    audio_path: str

    duration: float

    num_samples: int = 0

    status: str = "done"

    error: str | None = None

    # Number of GPT audio tokens generated. A value at the decoder's
    # ceiling means generation never terminated on its own.
    gpt_tokens: int | None = None

    # Cosine similarity between the reference speaker embedding and one
    # recomputed from this segment's audio. 1.0 is identical.
    speaker_similarity: float | None = None


class SynthesisResult(BaseModel):
    """
    Complete synthesis output.
    """

    job_id: str

    sample_rate: int

    model_id: str | None = None

    # The exact inference kwargs used. This project's central failure mode
    # was a decoder parameter silently changing the output, so the settings
    # that produced a given audio file belong in the artifact beside it.
    params: dict = Field(default_factory=dict)

    segments: list[SynthesizedSegment] = Field(default_factory=list)
