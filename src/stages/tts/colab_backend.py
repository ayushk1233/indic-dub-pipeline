import json
from pathlib import Path

from src.stages.tts.backend import ExternalExecutionBackend
from src.stages.tts.bundle.importer import BundleImporter, ImportedBundle
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
        """
        Read back what the GPU host produced.

        Accepts a whole bundle (directory or zip), which is the normal case and
        gets the full verification in `BundleImporter`, or a bare
        synthesis_result.json, which cannot be verified because the audio it
        refers to is only locatable relative to a bundle root.
        """
        result_path = Path(result_path)

        if result_path.is_dir() or result_path.suffix == ".zip":
            return self.import_bundle(result_path).result

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return SynthesisResult(**data)

    def import_bundle(self, path: Path) -> ImportedBundle:
        """
        Import a bundle with its audio resolved and checked.

        Prefer this over `import_result` wherever the caller will go on to read
        the synthesized audio, because only this reports which segments are
        actually usable.
        """
        return BundleImporter().import_bundle(Path(path))

    def synthesize(
        self,
        request: SynthesisRequest,
    ) -> SynthesisResult:
        self.export_request(request, self.workspace)
        print(f"Synthesis request exported to {self.workspace}/synthesis_request.json")
        print(f"Run the synthesis notebook and place synthesis_result.json in:\n  {self.workspace}/")
        
        raise NotImplementedError(
            "External execution required."
        )
