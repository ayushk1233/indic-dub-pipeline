import math
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.eval.translation_metrics import MIN_GAP_S, natural_cps
from src.stages.tts.models import SynthesisRequest, SynthesisResult


# Level below which a sample is treated as silence when measuring the
# leading and trailing tails.
SILENCE_DBFS = -40.0

# A trailing tail longer than this is the fingerprint of generation that did
# not terminate when the text ended.
RUNAWAY_TAIL_S = 0.5

# Realized speaking rate below this fraction of natural means the model is
# drawling or padding rather than speaking the text.
DRAWL_RATIO = 0.7

# Cosine similarity against the reference speaker below which the clone is
# not recognisably the same voice.
SIMILARITY_FLOOR = 0.75

# Tempo beyond which time-stretching stops sounding like speech.
MAX_TEMPO = 1.25


def _dbfs(value: float) -> float:
    return 20.0 * math.log10(value) if value > 1e-9 else -120.0


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """
    Read a PCM wav into a mono float array in [-1, 1].

    Uses the standard library so that reading synthesized audio needs no
    audio dependency beyond numpy.
    """
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(width)

    if dtype is None:
        raise ValueError(f"Unsupported sample width: {width} bytes")

    samples = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    samples /= float(np.iinfo(dtype).max)

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    return samples, rate


@dataclass
class AudioHealth:
    peak_dbfs: float
    rms_dbfs: float
    clipped_fraction: float
    dc_offset: float
    leading_silence_s: float
    trailing_silence_s: float


def analyze_audio(samples: np.ndarray, sample_rate: int) -> AudioHealth:
    if samples.size == 0:
        return AudioHealth(-120.0, -120.0, 0.0, 0.0, 0.0, 0.0)

    magnitude = np.abs(samples)
    threshold = 10.0 ** (SILENCE_DBFS / 20.0)
    voiced = np.flatnonzero(magnitude > threshold)

    if voiced.size:
        leading = voiced[0] / sample_rate
        trailing = (samples.size - 1 - voiced[-1]) / sample_rate
    else:
        leading = samples.size / sample_rate
        trailing = 0.0

    return AudioHealth(
        peak_dbfs=_dbfs(float(magnitude.max())),
        rms_dbfs=_dbfs(float(np.sqrt(np.mean(np.square(samples))))),
        clipped_fraction=float(np.mean(magnitude >= 0.99)),
        dc_offset=float(np.mean(samples)),
        leading_silence_s=float(leading),
        trailing_silence_s=float(trailing),
    )


@dataclass
class SegmentSynthesisMetrics:
    """
    How one synthesized segment behaved, against the slot it must fill.
    """

    segment_id: int
    status: str

    chars: int
    slot_s: float
    budget_s: float
    duration_s: float

    # duration divided by the budget. Above 1.0 the segment does not fit.
    duration_ratio: float

    # The atempo factor needed to make it fit, 1.0 when it already does.
    required_tempo: float

    # Characters per second actually delivered, and that rate as a fraction
    # of natural speech. Well below 1.0 means the model padded the output.
    realized_cps: float
    pace_ratio: float

    speaker_similarity: float | None
    gpt_tokens: int | None

    audio: AudioHealth | None

    flags: list[str] = field(default_factory=list)

    @property
    def fits(self) -> bool:
        return self.duration_ratio <= 1.0

    @property
    def stretchable(self) -> bool:
        return self.required_tempo <= MAX_TEMPO


@dataclass
class SynthesisMetrics:
    job_id: str
    sample_rate: int
    model_id: str | None
    params: dict

    num_segments: int
    num_done: int
    num_failed: int
    num_missing: int

    mean_speaker_similarity: float | None
    min_speaker_similarity: float | None
    num_below_similarity_floor: int

    num_fits: int
    num_stretchable: int
    num_overrun: int

    mean_duration_ratio: float
    mean_pace_ratio: float

    total_synth_duration_s: float

    segments: list[SegmentSynthesisMetrics] = field(default_factory=list)

    @property
    def flagged(self) -> list[SegmentSynthesisMetrics]:
        return [s for s in self.segments if s.flags]


def evaluate_synthesis(
    request: SynthesisRequest,
    result: SynthesisResult,
    bundle_dir: Path,
    total_duration: float | None = None,
) -> SynthesisMetrics:
    """
    Measure synthesized audio against the request that produced it.

    Durations are re-derived from the audio rather than trusted from the
    remote result, because the two disagreeing is itself a defect worth
    catching.
    """
    bundle_dir = Path(bundle_dir)
    language = request.language

    by_id = {s.segment_id: s for s in result.segments}
    requested = request.segments

    metrics: list[SegmentSynthesisMetrics] = []

    for index, segment in enumerate(requested):
        if index + 1 < len(requested):
            next_start = requested[index + 1].start_ts
        elif total_duration is not None:
            next_start = total_duration
        else:
            next_start = segment.end_ts

        slot = max(segment.end_ts - segment.start_ts, 1e-6)
        budget = max(next_start - segment.start_ts - MIN_GAP_S, slot)

        chars = len(segment.text.strip())
        produced = by_id.get(segment.segment_id)
        flags: list[str] = []

        if produced is None:
            metrics.append(
                SegmentSynthesisMetrics(
                    segment_id=segment.segment_id,
                    status="missing",
                    chars=chars,
                    slot_s=slot,
                    budget_s=budget,
                    duration_s=0.0,
                    duration_ratio=0.0,
                    required_tempo=1.0,
                    realized_cps=0.0,
                    pace_ratio=0.0,
                    speaker_similarity=None,
                    gpt_tokens=None,
                    audio=None,
                    flags=["missing"],
                )
            )
            continue

        if produced.status != "done":
            flags.append("failed")

        audio = None
        duration = produced.duration

        wav_path = bundle_dir / produced.audio_path

        if produced.audio_path and wav_path.exists():
            try:
                samples, rate = read_wav(wav_path)
                audio = analyze_audio(samples, rate)
                duration = samples.size / rate

                if rate != result.sample_rate:
                    flags.append(f"sample-rate-{rate}")

                if abs(duration - produced.duration) > 0.05:
                    flags.append("duration-mismatch")

                if audio.trailing_silence_s > RUNAWAY_TAIL_S:
                    flags.append("long-tail")

                if audio.clipped_fraction > 0.001:
                    flags.append("clipping")

                if audio.peak_dbfs < -30.0:
                    flags.append("too-quiet")
            except Exception as exc:
                flags.append(f"unreadable: {exc}")
        elif produced.status == "done":
            flags.append("audio-missing")

        duration = max(duration, 1e-6)
        duration_ratio = duration / budget
        realized_cps = chars / duration
        pace_ratio = realized_cps / natural_cps(language)

        if duration_ratio > 1.0:
            flags.append("overruns-slot")

        if duration_ratio > MAX_TEMPO:
            flags.append("beyond-stretch")

        if pace_ratio < DRAWL_RATIO:
            flags.append("drawling")

        if (
            produced.speaker_similarity is not None
            and produced.speaker_similarity < SIMILARITY_FLOOR
        ):
            flags.append("voice-drift")

        metrics.append(
            SegmentSynthesisMetrics(
                segment_id=segment.segment_id,
                status=produced.status,
                chars=chars,
                slot_s=slot,
                budget_s=budget,
                duration_s=duration,
                duration_ratio=duration_ratio,
                required_tempo=max(duration_ratio, 1.0),
                realized_cps=realized_cps,
                pace_ratio=pace_ratio,
                speaker_similarity=produced.speaker_similarity,
                gpt_tokens=produced.gpt_tokens,
                audio=audio,
                flags=flags,
            )
        )

    similarities = [
        s.speaker_similarity for s in metrics if s.speaker_similarity is not None
    ]
    scored = [s for s in metrics if s.status == "done"]

    return SynthesisMetrics(
        job_id=result.job_id,
        sample_rate=result.sample_rate,
        model_id=result.model_id,
        params=result.params,
        num_segments=len(metrics),
        num_done=sum(1 for s in metrics if s.status == "done"),
        num_failed=sum(1 for s in metrics if s.status == "failed"),
        num_missing=sum(1 for s in metrics if s.status == "missing"),
        mean_speaker_similarity=(
            sum(similarities) / len(similarities) if similarities else None
        ),
        min_speaker_similarity=min(similarities) if similarities else None,
        num_below_similarity_floor=sum(
            1 for v in similarities if v < SIMILARITY_FLOOR
        ),
        num_fits=sum(1 for s in scored if s.fits),
        num_stretchable=sum(1 for s in scored if not s.fits and s.stretchable),
        num_overrun=sum(1 for s in scored if not s.stretchable),
        mean_duration_ratio=(
            sum(s.duration_ratio for s in scored) / len(scored) if scored else 0.0
        ),
        mean_pace_ratio=(
            sum(s.pace_ratio for s in scored) / len(scored) if scored else 0.0
        ),
        total_synth_duration_s=sum(s.duration_s for s in scored),
        segments=metrics,
    )


def evaluate_intelligibility(
    request: SynthesisRequest,
    result: SynthesisResult,
    bundle_dir: Path,
    backend,
) -> dict[int, float]:
    """
    Round-trip check: transcribe the synthesized audio and score it against
    the text that was fed to the model.

    This is the only metric that catches gibberish, truncation and repetition,
    because all three produce audio that is perfectly healthy by every other
    measure here. `backend` is any loaded ASR backend exposing
    transcribe_audio(); the repo's FasterWhisperBackend fits.
    """
    import jiwer

    texts = {s.segment_id: s.text for s in request.segments}
    scores: dict[int, float] = {}

    for produced in result.segments:
        reference = texts.get(produced.segment_id, "").strip()
        wav_path = Path(bundle_dir) / produced.audio_path

        if produced.status != "done" or not reference or not wav_path.exists():
            continue

        segments, _ = backend.transcribe_audio(str(wav_path))
        hypothesis = " ".join(s.text for s in segments).strip()

        scores[produced.segment_id] = (
            jiwer.cer(reference, hypothesis) if hypothesis else 1.0
        )

    return scores
