import json
import shutil
from datetime import datetime
from pathlib import Path

from src.stages.tts.bundle.models import (
    BundleManifest,
    BundleMetadata,
    BundlePaths,
)
from src.stages.tts.models import SynthesisRequest


class BundleExporter:
    def export(
        self,
        request: SynthesisRequest,
        reference_audio: Path,
        output_dir: Path,
    ) -> Path:
        # Create the directory structure
        request_dir = output_dir / "request"
        output_sub_dir = output_dir / "output"
        logs_dir = output_dir / "logs"

        request_dir.mkdir(parents=True, exist_ok=True)
        output_sub_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Copy reference audio
        bundle_ref_audio = request_dir / "reference.wav"
        shutil.copy2(reference_audio, bundle_ref_audio)

        # Write synthesis request
        bundle_request_json = request_dir / "synthesis_request.json"
        with open(bundle_request_json, "w", encoding="utf-8") as f:
            json.dump(
                request.model_dump(),
                f,
                ensure_ascii=False,
                indent=2,
            )

        # Build and write manifest
        manifest = BundleManifest(
            metadata=BundleMetadata(
                bundle_version="1.0",
                job_id=request.job_id,
                created_at=datetime.utcnow().isoformat() + "Z",
            ),
            paths=BundlePaths(
                request_json="request/synthesis_request.json",
                reference_audio="request/reference.wav",
            ),
        )

        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                manifest.model_dump(),
                f,
                ensure_ascii=False,
                indent=2,
            )

        return output_dir
