import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path


# Segments shorter than this carry too little audio to transcribe reliably.
MIN_USEFUL_SEGMENT_S = 0.4

# Whisper's mel window. A segment longer than this gets split internally and
# loses the timing correspondence the dub depends on.
ASR_WINDOW_S = 30.0


@dataclass
class PreprocessMetrics:
    """
    Shape of the segmentation, which determines every timing budget downstream.

    If the dub cannot hit its timing, this is the first place to look: slots
    that are too tight here cannot be rescued by translation or synthesis.
    """

    num_segments: int

    total_duration_s: float
    total_speech_s: float
    speech_ratio: float

    min_segment_s: float
    median_segment_s: float
    max_segment_s: float

    min_gap_s: float
    median_gap_s: float
    max_gap_s: float

    num_short_segments: int
    num_oversized_segments: int
    num_missing_chunks: int

    segment_durations: list[float] = field(default_factory=list)
    gaps: list[float] = field(default_factory=list)

    @property
    def mean_segment_s(self) -> float:
        return (
            sum(self.segment_durations) / len(self.segment_durations)
            if self.segment_durations
            else 0.0
        )


def evaluate_manifest(
    manifest_path: Path,
    total_duration: float | None = None,
) -> PreprocessMetrics:
    """
    Measure a preprocessing manifest.

    `total_duration` is the source media length; without it the speech ratio
    is computed against the span the segments themselves cover.
    """
    manifest_path = Path(manifest_path)

    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    if not entries:
        return PreprocessMetrics(
            num_segments=0,
            total_duration_s=total_duration or 0.0,
            total_speech_s=0.0,
            speech_ratio=0.0,
            min_segment_s=0.0,
            median_segment_s=0.0,
            max_segment_s=0.0,
            min_gap_s=0.0,
            median_gap_s=0.0,
            max_gap_s=0.0,
            num_short_segments=0,
            num_oversized_segments=0,
            num_missing_chunks=0,
        )

    entries = sorted(entries, key=lambda e: float(e["start_ts"]))

    durations = [
        float(e["end_ts"]) - float(e["start_ts"])
        for e in entries
    ]

    gaps = [
        float(entries[i + 1]["start_ts"]) - float(entries[i]["end_ts"])
        for i in range(len(entries) - 1)
    ]

    span = float(entries[-1]["end_ts"])
    total = total_duration if total_duration is not None else span

    missing = 0
    for entry in entries:
        chunk = Path(entry["chunk_path"])
        if not chunk.is_absolute():
            chunk = manifest_path.parent.parent.parent / chunk
        if not Path(entry["chunk_path"]).exists() and not chunk.exists():
            missing += 1

    return PreprocessMetrics(
        num_segments=len(entries),
        total_duration_s=total,
        total_speech_s=sum(durations),
        speech_ratio=sum(durations) / total if total > 0 else 0.0,
        min_segment_s=min(durations),
        median_segment_s=statistics.median(durations),
        max_segment_s=max(durations),
        min_gap_s=min(gaps) if gaps else 0.0,
        median_gap_s=statistics.median(gaps) if gaps else 0.0,
        max_gap_s=max(gaps) if gaps else 0.0,
        num_short_segments=sum(1 for d in durations if d < MIN_USEFUL_SEGMENT_S),
        num_oversized_segments=sum(1 for d in durations if d > ASR_WINDOW_S),
        num_missing_chunks=missing,
        segment_durations=durations,
        gaps=gaps,
    )
