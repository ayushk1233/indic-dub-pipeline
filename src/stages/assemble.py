"""
Assemble synthesized segments into one dubbed audio track.

This is the step that makes a dub a dub. Synthesis returns a pile of clips
that are each roughly the right words but rarely the right length, and this
module decides where each one goes and what to do when it does not fit.

The fitting cascade, cheapest and least damaging first:

  Tier 1, trim and place.   Generated speech usually carries trailing silence
                            the decoder added after the words ended. Cutting
                            it is free and often enough on its own.
  Tier 2, absorb the gap.   A segment may run past its own slot into the pause
                            that follows without colliding with anything, so
                            the budget is the slot plus that pause.
  Tier 3, time-stretch.     Speed the clip up, but only to MAX_TEMPO. Past
                            roughly 1.25x, speech stops sounding like speech,
                            and a dub that is intelligible but obviously rushed
                            is worse than one that drifts slightly.
  Tier 4, bounded drift.    When even a capped stretch does not fit, let the
                            segment run long and push what follows, tracking
                            the accumulated lateness and giving it back at the
                            next pause big enough to absorb it.

The track is built as one NumPy buffer rather than an FFmpeg filtergraph, for
two reasons. FFmpeg's `amix` renormalizes levels, which silently undoes the
gain work, and a buffer lets a test assert that a segment landed at an exact
sample index rather than approximately somewhere.
"""

import json
import subprocess
import tempfile
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from src.eval.tts_metrics import MAX_TEMPO, analyze_audio, read_wav
from src.eval.translation_metrics import MIN_GAP_S


# Fade applied to each end of every placed segment, to stop the buffer joins
# from clicking.
FADE_S = 0.005

# How far a segment may push the ones after it before the cascade gives up and
# simply lets it overlap.
MAX_DRIFT_S = 0.25

# A gap at least this long is a real pause, and any accumulated drift is
# handed back there rather than carried through the rest of the track.
DRIFT_RESET_GAP_S = 0.35

# atempo below this is not worth a process spawn.
TEMPO_EPSILON = 0.01


@dataclass
class PlacedSegment:
    """
    Where one synthesized segment ended up, and what it cost to get it there.
    """

    segment_id: int

    # What the timeline asked for.
    target_start_s: float
    slot_s: float
    budget_s: float

    # What synthesis produced, before and after trimming.
    raw_duration_s: float
    trimmed_duration_s: float

    # What was done about the difference.
    tier: str
    tempo: float
    final_duration_s: float

    # Where it actually landed.
    actual_start_s: float
    drift_s: float

    overlaps_next: bool = False

    @property
    def fits(self) -> bool:
        return self.tier in {"placed", "trimmed", "gap-absorbed"}


@dataclass
class AssemblyResult:
    job_id: str
    sample_rate: int
    total_duration_s: float
    output_path: str

    num_segments: int = 0
    num_placed: int = 0
    num_stretched: int = 0
    num_drifted: int = 0
    num_missing: int = 0

    max_tempo_used: float = 1.0
    max_drift_s: float = 0.0

    segments: list[PlacedSegment] = field(default_factory=list)

    @property
    def pct_untouched(self) -> float:
        if not self.num_segments:
            return 0.0
        return 100.0 * self.num_placed / self.num_segments


def trim_trailing_silence(
    samples: np.ndarray,
    sample_rate: int,
    keep_s: float = 0.05,
) -> np.ndarray:
    """
    Cut silence the decoder appended after the words ended, leaving a short
    natural tail.

    This is tier one of the cascade and the only one that costs nothing: the
    removed audio carries no speech, so removing it cannot hurt intelligibility.
    """
    if samples.size == 0:
        return samples

    health = analyze_audio(samples, sample_rate)

    if health.trailing_silence_s <= keep_s:
        return samples

    keep_samples = int(keep_s * sample_rate)
    cut = int(health.trailing_silence_s * sample_rate) - keep_samples

    if cut <= 0:
        return samples

    return samples[: max(samples.size - cut, 1)]


def time_stretch(
    samples: np.ndarray,
    sample_rate: int,
    tempo: float,
) -> np.ndarray:
    """
    Speed audio up by `tempo` without changing pitch, via FFmpeg's atempo.

    atempo only accepts 0.5 to 2.0 per instance, which is well outside
    anything this cascade asks for, so a single filter is always enough here.
    """
    if abs(tempo - 1.0) < TEMPO_EPSILON or samples.size == 0:
        return samples

    with tempfile.TemporaryDirectory() as workspace:
        source = Path(workspace) / "in.wav"
        stretched = Path(workspace) / "out.wav"

        write_wav_samples(source, samples, sample_rate)

        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-filter:a",
                f"atempo={tempo:.6f}",
                "-y",
                str(stretched),
            ],
            check=True,
            capture_output=True,
        )

        result, _ = read_wav(stretched)

    return result


def write_wav_samples(
    path: Path,
    samples: np.ndarray,
    sample_rate: int,
) -> Path:
    """
    Write a mono float array in [-1, 1] as 16-bit PCM.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())

    return path


def apply_fades(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Short linear fades on both ends, so adding a segment into the buffer does
    not produce a step discontinuity.
    """
    fade = int(FADE_S * sample_rate)

    if samples.size < 2 * fade or fade == 0:
        return samples

    faded = samples.copy()
    ramp = np.linspace(0.0, 1.0, fade, dtype=samples.dtype)

    faded[:fade] *= ramp
    faded[-fade:] *= ramp[::-1]

    return faded


def _budget_for(
    index: int,
    segments: list,
    total_duration: float,
) -> tuple[float, float, float]:
    """
    Slot, budget, and the gap that follows, for one requested segment.

    The budget pools the following pause, which is the same arithmetic the
    feasibility check uses in `evaluate_segment_fitness`. Keeping the two in
    agreement matters: a segment predicted to fit must actually be given the
    room that prediction assumed.
    """
    segment = segments[index]

    if index + 1 < len(segments):
        next_start = segments[index + 1].start_ts
    else:
        next_start = max(total_duration, segment.end_ts)

    slot = max(segment.end_ts - segment.start_ts, 1e-6)
    gap = max(next_start - segment.end_ts, 0.0)
    budget = max(next_start - segment.start_ts - MIN_GAP_S, slot)

    return slot, budget, gap


def assemble(
    imported,
    total_duration: float,
    output_path: Path,
    sample_rate: int | None = None,
) -> AssemblyResult:
    """
    Build one dubbed track from an imported bundle.

    `imported` is an `ImportedBundle`. Only segments it marked usable are
    placed; everything else leaves silence, which is the honest outcome when
    synthesis did not produce audio.
    """
    request = imported.request
    segments = request.segments
    rate = sample_rate or imported.result.sample_rate or request.output_sample_rate

    total_samples = max(int(total_duration * rate), 1)
    track = np.zeros(total_samples, dtype=np.float32)

    placements: list[PlacedSegment] = []
    drift = 0.0

    for index, segment in enumerate(segments):
        slot, budget, gap = _budget_for(index, segments, total_duration)
        audio_path = imported.audio_paths.get(segment.segment_id)

        if audio_path is None:
            placements.append(
                PlacedSegment(
                    segment_id=segment.segment_id,
                    target_start_s=segment.start_ts,
                    slot_s=slot,
                    budget_s=budget,
                    raw_duration_s=0.0,
                    trimmed_duration_s=0.0,
                    tier="missing",
                    tempo=1.0,
                    final_duration_s=0.0,
                    actual_start_s=segment.start_ts,
                    drift_s=drift,
                )
            )
            continue

        samples, file_rate = read_wav(audio_path)
        raw_duration = samples.size / file_rate if file_rate else 0.0

        samples = trim_trailing_silence(samples, file_rate)
        trimmed_duration = samples.size / file_rate if file_rate else 0.0

        # Decide the tier.
        if trimmed_duration <= slot:
            tier = "trimmed" if trimmed_duration < raw_duration else "placed"
            tempo = 1.0
        elif trimmed_duration <= budget:
            tier = "gap-absorbed"
            tempo = 1.0
        else:
            required = trimmed_duration / budget
            tempo = min(required, MAX_TEMPO)
            tier = "stretched" if required <= MAX_TEMPO else "drifted"

        if tempo > 1.0 + TEMPO_EPSILON:
            samples = time_stretch(samples, file_rate, tempo)

        if file_rate != rate and samples.size:
            samples = _resample(samples, file_rate, rate)

        final_duration = samples.size / rate if rate else 0.0

        # Place it, carrying any drift accumulated so far.
        start = segment.start_ts + drift
        start_index = max(int(start * rate), 0)
        end_index = min(start_index + samples.size, total_samples)

        if end_index > start_index:
            window = apply_fades(samples[: end_index - start_index], rate)
            track[start_index:end_index] += window

        overrun = max((start + final_duration) - (segment.start_ts + budget), 0.0)

        if tier == "drifted":
            drift = min(drift + overrun, MAX_DRIFT_S)
        elif gap >= DRIFT_RESET_GAP_S:
            # A real pause: hand back whatever lateness we accumulated rather
            # than letting it compound across the whole track.
            drift = 0.0

        placements.append(
            PlacedSegment(
                segment_id=segment.segment_id,
                target_start_s=segment.start_ts,
                slot_s=slot,
                budget_s=budget,
                raw_duration_s=raw_duration,
                trimmed_duration_s=trimmed_duration,
                tier=tier,
                tempo=tempo,
                final_duration_s=final_duration,
                actual_start_s=start,
                drift_s=drift,
                overlaps_next=overrun > 0.0,
            )
        )

    peak = float(np.abs(track).max()) if track.size else 0.0

    if peak > 1.0:
        track /= peak

    output_path = Path(output_path)
    write_wav_samples(output_path, track, rate)

    return AssemblyResult(
        job_id=request.job_id,
        sample_rate=rate,
        total_duration_s=total_samples / rate,
        output_path=str(output_path),
        num_segments=len(placements),
        num_placed=sum(1 for p in placements if p.fits),
        num_stretched=sum(1 for p in placements if p.tier == "stretched"),
        num_drifted=sum(1 for p in placements if p.tier == "drifted"),
        num_missing=sum(1 for p in placements if p.tier == "missing"),
        max_tempo_used=max((p.tempo for p in placements), default=1.0),
        max_drift_s=max((abs(p.drift_s) for p in placements), default=0.0),
        segments=placements,
    )


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """
    Linear resample. Only used when a segment's file rate disagrees with the
    track rate, which is a configuration bug the importer already flags; this
    keeps assembly working rather than producing a track at the wrong speed.
    """
    if source_rate == target_rate or samples.size == 0:
        return samples

    duration = samples.size / source_rate
    target_size = max(int(duration * target_rate), 1)

    source_positions = np.linspace(0.0, samples.size - 1, num=samples.size)
    target_positions = np.linspace(0.0, samples.size - 1, num=target_size)

    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def write_timeline(result: AssemblyResult, path: Path) -> Path:
    """
    Persist which tier fixed each segment.

    This is the quality-control record the project is named for: it says, per
    segment, whether the dub fit naturally, needed the pause after it, was sped
    up, or was allowed to run long.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)

    return path


def render_timeline(result: AssemblyResult) -> str:
    lines: list[str] = []

    lines.append("=" * 78)
    lines.append(f"ASSEMBLY   {result.output_path}")
    lines.append("=" * 78)
    lines.append("")
    lines.append(
        f"  {result.num_segments} segments, {result.num_placed} fit untouched "
        f"({result.pct_untouched:.0f}%), {result.num_stretched} stretched, "
        f"{result.num_drifted} drifted, {result.num_missing} missing"
    )
    lines.append(
        f"  max tempo {result.max_tempo_used:.2f}x   "
        f"max drift {result.max_drift_s:.2f}s   "
        f"track {result.total_duration_s:.2f}s at {result.sample_rate} Hz"
    )
    lines.append("")
    lines.append(
        f"  {'seg':>4} {'start':>8} {'slot':>7} {'budget':>7} "
        f"{'raw':>7} {'final':>7} {'tempo':>6} {'drift':>6}  tier"
    )

    for placed in result.segments:
        lines.append(
            f"  {placed.segment_id:>4} {placed.actual_start_s:>7.2f}s "
            f"{placed.slot_s:>6.2f}s {placed.budget_s:>6.2f}s "
            f"{placed.raw_duration_s:>6.2f}s {placed.final_duration_s:>6.2f}s "
            f"{placed.tempo:>5.2f}x {placed.drift_s:>5.2f}s  {placed.tier}"
        )

    lines.append("")
    lines.append("=" * 78)

    return "\n".join(lines)
