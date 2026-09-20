from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    RETRYING = "retrying"


class StageResult(BaseModel):
    stage_name: str
    status: StageStatus
    output_path: Optional[str] = None
    metrics: dict = Field(default_factory=dict)
    error: Optional[str] = None
    attempt: int = 1


class Job(BaseModel):
    job_id: str
    source_lang: str
    target_lang: str
    stages: dict[str, StageResult] = Field(default_factory=dict)
    created_at: str