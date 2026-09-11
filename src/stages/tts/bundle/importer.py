"""
Read back a synthesis bundle that was executed somewhere else.

This is the return half of the GPU boundary. `BundleExporter` writes a bundle,
some GPU host runs it, and this brings the result home. It is deliberately
suspicious of what comes back, for three reasons learned the hard way:

- The audio was written on a different machine, so `audio_path` is relative to
  the bundle root and must be resolved against it rather than the working
  directory, and must be checked for escaping that root.
- The remote result records a duration that the remote code computed. Whether
  that matches the audio actually on disk is itself worth knowing, so the
  duration is re-derived here and disagreement is reported, not smoothed over.
- A partial run is normal. Colab times out and the worker writes results after
  every segment. Missing segments are a finding to report, never an exception.
"""

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from src.stages.tts.models import SynthesisRequest, SynthesisResult


# Every bundle layout this importer understands. Kept in step with the
# exporter and with colab/xtts_worker.py, which checks the same set.
SUPPORTED_BUNDLE_VERSIONS = {"1.0", "1.1", "1.2"}

# A re-derived duration further than this from the recorded one means the
# result file and the audio beside it disagree about what was produced.
DURATION_TOLERANCE_S = 0.05


@dataclass
class SegmentIssue:
    segment_id: int
    kind: str
    detail: str


@dataclass
class ImportedBundle:
    """
    A synthesis bundle that has been read back and checked.
    """

    bundle_dir: Path
    job_id: str
    bundle_version: str

    request: SynthesisRequest
    result: SynthesisResult

    # Absolute, verified paths for every segment whose audio was found.
    audio_paths: dict[int, Path] = field(default_factory=dict)

    issues: list[SegmentIssue] = field(default_factory=list)

    @property
    def requested_ids(self) -> set[int]:
        return {segment.segment_id for segment in self.request.segments}

    @property
    def usable_ids(self) -> set[int]:
        return set(self.audio_paths)

    @property
    def missing_ids(self) -> list[int]:
        return sorted(self.requested_ids - self.usable_ids)

    @property
    def complete(self) -> bool:
        return not self.missing_ids

    def issues_of(self, kind: str) -> list[SegmentIssue]:
        return [issue for issue in self.issues if issue.kind == kind]

    def summary(self) -> str:
        return (
            f"job {self.job_id}: {len(self.usable_ids)}/{len(self.requested_ids)} "
            f"segments usable, {len(self.issues)} issues"
        )


class BundleImporter:
    """
    Mirror of `BundleExporter`. Where the exporter writes a bundle for a GPU
    host to consume, this reads back what that host produced.
    """

    def extract(self, archive_path: Path, destination: Path) -> Path:
        """
        Unpack a bundle zip, refusing any member that would write outside the
        destination. A zip is an untrusted container even when you made it.
        """
        archive_path = Path(archive_path)
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        resolved_root = destination.resolve()

        with zipfile.ZipFile(archive_path, "r") as archive:
            for member in archive.namelist():
                target = (destination / member).resolve()

                if not target.is_relative_to(resolved_root):
                    raise ValueError(
                        f"Refusing to extract {member!r}: it escapes the bundle root."
                    )

            archive.extractall(destination)

        return destination

    def import_bundle(
        self,
        path: Path,
        extract_to: Path | None = None,
    ) -> ImportedBundle:
        """
        Read a bundle directory, or a bundle zip which is unpacked first.
        """
        path = Path(path)

        if path.is_file() and path.suffix == ".zip":
            destination = (
                Path(extract_to)
                if extract_to is not None
                else path.with_suffix("")
            )
            bundle_dir = self.extract(path, destination)
        else:
            bundle_dir = path

        if not bundle_dir.is_dir():
            raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")

        manifest = self._read_json(bundle_dir / "manifest.json", "manifest")
        metadata = manifest.get("metadata", {})
        version = metadata.get("bundle_version")

        if version not in SUPPORTED_BUNDLE_VERSIONS:
            raise ValueError(
                f"Unsupported bundle version: {version!r}. "
                f"Supported: {sorted(SUPPORTED_BUNDLE_VERSIONS)}"
            )

        request = SynthesisRequest(
            **self._read_json(
                bundle_dir / "request" / "synthesis_request.json",
                "synthesis request",
            )
        )

        result_path = bundle_dir / "output" / "synthesis_result.json"

        if not result_path.exists():
            raise FileNotFoundError(
                f"No synthesis result in bundle: {result_path}. "
                "The GPU side has not run, or its output was not copied back."
            )

        result = SynthesisResult(**self._read_json(result_path, "synthesis result"))

        imported = ImportedBundle(
            bundle_dir=bundle_dir,
            job_id=metadata.get("job_id", request.job_id),
            bundle_version=version,
            request=request,
            result=result,
        )

        self._resolve_segments(imported)
        self._note_missing(imported)

        return imported

    def _read_json(self, path: Path, what: str) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Bundle {what} not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_segments(self, imported: ImportedBundle) -> None:
        root = imported.bundle_dir.resolve()
        requested = imported.requested_ids

        for produced in imported.result.segments:
            segment_id = produced.segment_id

            if segment_id not in requested:
                imported.issues.append(
                    SegmentIssue(
                        segment_id,
                        "unexpected",
                        "result contains a segment that was never requested",
                    )
                )
                continue

            if produced.status != "done":
                imported.issues.append(
                    SegmentIssue(
                        segment_id,
                        "failed",
                        produced.error or f"status is {produced.status!r}",
                    )
                )
                continue

            if not produced.audio_path:
                imported.issues.append(
                    SegmentIssue(segment_id, "no-path", "status is done but no audio path")
                )
                continue

            candidate = (imported.bundle_dir / produced.audio_path).resolve()

            if not candidate.is_relative_to(root):
                imported.issues.append(
                    SegmentIssue(
                        segment_id,
                        "path-escape",
                        f"{produced.audio_path!r} resolves outside the bundle",
                    )
                )
                continue

            if not candidate.exists():
                imported.issues.append(
                    SegmentIssue(
                        segment_id,
                        "audio-missing",
                        f"{produced.audio_path!r} is not in the bundle",
                    )
                )
                continue

            self._check_duration(imported, segment_id, candidate, produced.duration)
            imported.audio_paths[segment_id] = candidate

    def _check_duration(
        self,
        imported: ImportedBundle,
        segment_id: int,
        audio_path: Path,
        recorded: float,
    ) -> None:
        """
        Re-derive duration from the audio rather than trusting the remote
        number. Reading the file is also the cheapest proof it is real audio.
        """
        from src.eval.tts_metrics import read_wav

        try:
            samples, rate = read_wav(audio_path)
        except Exception as exc:
            imported.issues.append(
                SegmentIssue(segment_id, "unreadable", f"{type(exc).__name__}: {exc}")
            )
            return

        if rate <= 0 or samples.size == 0:
            imported.issues.append(
                SegmentIssue(segment_id, "empty-audio", "file contains no samples")
            )
            return

        actual = samples.size / rate

        if abs(actual - recorded) > DURATION_TOLERANCE_S:
            imported.issues.append(
                SegmentIssue(
                    segment_id,
                    "duration-mismatch",
                    f"result says {recorded:.3f}s, audio is {actual:.3f}s",
                )
            )

        if rate != imported.result.sample_rate:
            imported.issues.append(
                SegmentIssue(
                    segment_id,
                    "sample-rate",
                    f"result declares {imported.result.sample_rate} Hz, file is {rate} Hz",
                )
            )

    def _note_missing(self, imported: ImportedBundle) -> None:
        produced_ids = {segment.segment_id for segment in imported.result.segments}

        for segment_id in sorted(imported.requested_ids - produced_ids):
            imported.issues.append(
                SegmentIssue(
                    segment_id,
                    "missing",
                    "requested but absent from the result; the run did not finish it",
                )
            )
