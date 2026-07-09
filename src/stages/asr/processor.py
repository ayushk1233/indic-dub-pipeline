from abc import ABC, abstractmethod
from pathlib import Path

from src.stages.asr.models import TranscriptResult


class ASRProcessor(ABC):
    """
    Abstract interface for ASR engines.
    """

    @abstractmethod
    def transcribe(
        self,
        chunk_manifest: Path,
    ) -> TranscriptResult:
        """
        Transcribe all chunks described by the manifest.
        """
        raise NotImplementedError
