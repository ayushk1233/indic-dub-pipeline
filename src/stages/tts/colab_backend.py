import json
from pathlib import Path

from src.stages.tts.backend import ExternalExecutionBackend
from src.stages.tts.models import (
    SynthesisRequest,
    SynthesisResult,
)


class ColabTTSBackend(ExternalExecutionBackend):
    def __init__(
        self,
        workspace: Path,
    ):
        self.workspace = workspace

    def export_request(
        self,
        request: SynthesisRequest,
        output_dir: Path,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        request_path = output_dir / "synthesis_request.json"

        with open(request_path, "w", encoding="utf-8") as f:
            json.dump(
                request.model_dump(),
                f,
                ensure_ascii=False,
                indent=2,
            )

        return request_path

    def import_result(
        self,
        result_path: Path,
    ) -> SynthesisResult:
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return SynthesisResult(**data)

    def synthesize(
        self,
        request: SynthesisRequest,
    ) -> SynthesisResult:
        self.export_request(request, self.workspace)
        print(f"Synthesis request exported to {self.workspace}/synthesis_request.json")
        print(f"Run the XTTS notebook and place synthesis_result.json in:\n  {self.workspace}/")
        
        raise NotImplementedError(
            "External execution required."
        )
