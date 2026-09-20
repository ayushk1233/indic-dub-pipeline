from abc import ABC, abstractmethod

from src.orchestrator.models import StageResult


class PipelineStage(ABC):
    """
    Base contract implemented by every pipeline stage.

    Orchestration code communicates only with this interface,
    allowing stages to be swapped without changing the pipeline.
    """

    @abstractmethod
    def validate_input(self, input_path: str) -> bool:
        """
        Perform a lightweight validation before expensive work begins.
        """
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        input_path: str,
        job_id: str,
        cfg: dict,
    ) -> StageResult:
        """
        Execute the stage and return a structured StageResult.
        """
        raise NotImplementedError