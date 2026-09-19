import json
import shutil
import zipfile
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

        # Clear any previous run's output before re-exporting. Two reasons,
        # both of which have bitten this project: `package_bundle` zips
        # everything under the bundle root, so leftover audio rides along to
        # the GPU host; and a segment that fails there leaves the stale file
        # in place, where the importer accepts it as a real result. A bundle
        # must describe exactly one synthesis attempt.
        for stale_dir in (output_sub_dir, logs_dir):
            if stale_dir.exists():
                shutil.rmtree(stale_dir)

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
                # 1.1 adds SynthesisRequest.reference_text.
                # 1.2 adds SynthesisSegment.budget_s. Both are additive and
                # optional, so an older reader ignores the extra field and a
                # newer one falls back when it is absent.
                bundle_version="1.2",
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

    def package_bundle(
        self,
        bundle_dir: Path,
    ) -> Path:
        if not bundle_dir.exists() or not bundle_dir.is_dir():
            raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")

        manifest_path = bundle_dir / "manifest.json"
        request_json_path = bundle_dir / "request" / "synthesis_request.json"
        reference_audio_path = bundle_dir / "request" / "reference.wav"

        for required_file in [manifest_path, request_json_path, reference_audio_path]:
            if not required_file.exists():
                raise FileNotFoundError(f"Missing required bundle file: {required_file}")

        zip_path = bundle_dir.with_name(bundle_dir.name + ".zip")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in bundle_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(bundle_dir)
                    zipf.write(file_path, arcname)

        return zip_path
