from abc import ABC, abstractmethod
from pathlib import Path

from src.stages.tts.models import (
    SynthesisRequest,
    SynthesisResult,
)


class TTSBackend(ABC):
    """
    Abstract interface for all TTS execution backends.
    """

    @abstractmethod
    def synthesize(
        self,
        request: SynthesisRequest,
    ) -> SynthesisResult:
        """
        Execute speech synthesis.

        Implementations may execute:
        - locally
        - remotely
        - via Colab
        - via cloud GPU
        """
        raise NotImplementedError


class ExternalExecutionBackend(TTSBackend):
    """
    Base class for external execution systems
    (Colab, HTTP service, etc.)
    """

    @abstractmethod
    def export_request(
        self,
        request: SynthesisRequest,
        output_dir: Path,
    ) -> Path:
        """
        Serialize request for external execution.
        """
        raise NotImplementedError

    @abstractmethod
    def import_result(
        self,
        result_path: Path,
    ) -> SynthesisResult:
        """
        Read synthesized results produced externally.
        """
        raise NotImplementedError
