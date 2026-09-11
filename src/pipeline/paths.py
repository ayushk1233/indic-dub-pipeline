"""
Where a job's artifacts live.

Every stage previously built its own paths relative to the working directory,
which meant the pipeline only worked when run from the repo root and could not
be pointed at a scratch directory. One place to ask makes both possible.
"""

from dataclasses import dataclass
from pathlib import Path


DEFAULT_ROOT = Path("artifacts")


@dataclass(frozen=True)
class JobPaths:
    """
    Every path one job reads or writes.
    """

    job_id: str
    root: Path = DEFAULT_ROOT

    @property
    def job_dir(self) -> Path:
        return self.root / self.job_id

    # Preprocessing
    @property
    def audio(self) -> Path:
        return self.job_dir / "audio.wav"

    @property
    def chunks(self) -> Path:
        return self.job_dir / "chunks"

    @property
    def manifest(self) -> Path:
        return self.job_dir / "manifest.json"

    # Intermediate stage results, each the input of the next
    @property
    def transcript(self) -> Path:
        return self.job_dir / "transcript.json"

    @property
    def translation(self) -> Path:
        return self.job_dir / "translation.json"

    @property
    def candidates(self) -> Path:
        return self.job_dir / "translation_candidates.json"

    # The GPU boundary
    @property
    def bundle(self) -> Path:
        return self.job_dir / "tts_bundle"

    @property
    def bundle_zip(self) -> Path:
        return self.job_dir / "tts_bundle.zip"

    @property
    def reference_audio(self) -> Path:
        return self.job_dir / "reference.wav"

    # Output
    @property
    def dubbed_audio(self) -> Path:
        return self.job_dir / "dubbed.wav"

    @property
    def dubbed_video(self) -> Path:
        return self.job_dir / "dubbed.mp4"

    @property
    def timeline(self) -> Path:
        return self.job_dir / "timeline.json"

    @property
    def report_json(self) -> Path:
        return self.job_dir / "report.json"

    def ensure(self) -> "JobPaths":
        self.job_dir.mkdir(parents=True, exist_ok=True)
        return self
