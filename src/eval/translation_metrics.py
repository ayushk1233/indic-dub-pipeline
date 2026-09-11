from dataclasses import dataclass, field

from src.stages.translation.models import TranslationResult


# Approximate natural conversational speaking rate in characters per second.
#
# These are heuristic defaults, not measurements. They exist so that a segment
# can be judged feasible before any GPU time is spent on it: if the translated
# text needs a rate far above these, no TTS setting will make it fit and the
# fix belongs at translation or segmentation time.
NATURAL_CPS = {
    "en": 15.0,
    "hi": 13.0,
    "mr": 13.0,
    "ur": 13.0,
    "bn": 12.0,
    "gu": 12.0,
    "pa": 12.0,
    "or": 12.0,
    "ta": 11.0,
    "te": 11.0,
    "kn": 11.0,
    "ml": 11.0,
    "as": 12.0,
}

DEFAULT_NATURAL_CPS = 13.0

# Unicode ranges per target script, used to detect untranslated passthrough
# and wrong-script output.
SCRIPT_RANGES = {
    "hi": [(0x0900, 0x097F)],
    "mr": [(0x0900, 0x097F)],
    "bn": [(0x0980, 0x09FF)],
    "as": [(0x0980, 0x09FF)],
    "pa": [(0x0A00, 0x0A7F)],
    "gu": [(0x0A80, 0x0AFF)],
    "or": [(0x0B00, 0x0B7F)],
    "ta": [(0x0B80, 0x0BFF)],
    "te": [(0x0C00, 0x0C7F)],
    "kn": [(0x0C80, 0x0CFF)],
    "ml": [(0x0D00, 0x0D7F)],
    "ur": [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "en": [(0x0041, 0x005A), (0x0061, 0x007A)],
}

# A segment needing at most this multiple of the natural rate is treated as
# absorbable by the following pause or a gentle time-stretch.
TIGHT_THRESHOLD = 1.15

# Above this multiple, no amount of stretching sounds like speech.
IMPOSSIBLE_THRESHOLD = 1.5

# Minimum pause preserved between utterances when pooling a following gap.
MIN_GAP_S = 0.12


def natural_cps(language: str) -> float:
    return NATURAL_CPS.get(language, DEFAULT_NATURAL_CPS)


def script_ratio(text: str, language: str) -> float:
    """
    Fraction of letter characters that fall in the target script.

    Punctuation, digits and whitespace are ignored, so a correct translation
    scores near 1.0 regardless of how it is punctuated.
    """
    ranges = SCRIPT_RANGES.get(language)

    if not ranges:
        return 1.0

    letters = [c for c in text if c.isalpha()]

    if not letters:
        return 0.0

    in_script = sum(
        1
        for c in letters
        if any(low <= ord(c) <= high for low, high in ranges)
    )

    return in_script / len(letters)


def _digits(text: str) -> list[str]:
    return [c for c in text if c.isdigit()]


@dataclass
class SegmentFitness:
    """
    Whether one translated segment can physically be spoken in its slot.
    """

    segment_id: int

    source_chars: int
    target_chars: int

    # The original speech window this segment must land in.
    slot_s: float

    # The slot plus any silence before the next segment starts.
    budget_s: float

    source_cps: float
    required_cps: float

    # required_cps divided by the natural rate for the target language.
    # 1.0 means a normal speaker could just manage it.
    pace_ratio: float

    verdict: str

    script_ratio: float
    digits_preserved: bool
    is_empty: bool
    is_copy: bool

    @property
    def needs_attention(self) -> bool:
        return (
            self.verdict in {"infeasible", "impossible"}
            or self.script_ratio < 0.8
            or not self.digits_preserved
            or self.is_empty
            or self.is_copy
        )


@dataclass
class TranslationMetrics:
    num_segments: int
    avg_source_length: float
    avg_target_length: float
    expansion_ratio: float

    # Aggregate feasibility. These are the numbers that predict whether the
    # dub can hit its timing before synthesis runs.
    mean_pace_ratio: float = 0.0
    max_pace_ratio: float = 0.0
    num_fits: int = 0
    num_tight: int = 0
    num_infeasible: int = 0
    num_impossible: int = 0

    mean_script_ratio: float = 1.0
    num_wrong_script: int = 0
    num_empty: int = 0
    num_copied: int = 0
    num_digits_dropped: int = 0

    segments: list[SegmentFitness] = field(default_factory=list)

    @property
    def pct_needs_attention(self) -> float:
        if not self.segments:
            return 0.0
        flagged = sum(1 for s in self.segments if s.needs_attention)
        return 100.0 * flagged / len(self.segments)


def _verdict(pace_ratio: float) -> str:
    if pace_ratio <= 1.0:
        return "fits"
    if pace_ratio <= TIGHT_THRESHOLD:
        return "tight"
    if pace_ratio <= IMPOSSIBLE_THRESHOLD:
        return "infeasible"
    return "impossible"


def evaluate_segment_fitness(
    segment,
    budget_s: float,
    target_language: str,
) -> SegmentFitness:
    """
    Judge one translated segment against the time it has to be spoken in.
    """
    slot_s = max(segment.end_ts - segment.start_ts, 1e-6)
    budget_s = max(budget_s, 1e-6)

    source_text = segment.source_text or ""
    target_text = segment.translated_text or ""

    source_chars = len(source_text.strip())
    target_chars = len(target_text.strip())

    required_cps = target_chars / budget_s
    pace_ratio = required_cps / natural_cps(target_language)

    return SegmentFitness(
        segment_id=segment.segment_id,
        source_chars=source_chars,
        target_chars=target_chars,
        slot_s=slot_s,
        budget_s=budget_s,
        source_cps=source_chars / slot_s,
        required_cps=required_cps,
        pace_ratio=pace_ratio,
        verdict=_verdict(pace_ratio),
        script_ratio=script_ratio(target_text, target_language),
        digits_preserved=sorted(_digits(source_text)) == sorted(_digits(target_text)),
        is_empty=target_chars == 0,
        is_copy=target_chars > 0 and target_text.strip() == source_text.strip(),
    )


def evaluate_translation(
    translation: TranslationResult,
    total_duration: float | None = None,
) -> TranslationMetrics:
    """
    Structural and feasibility metrics for a translation result.

    `total_duration` lets the final segment pool the trailing silence into its
    budget the same way earlier segments pool the gap before the next one.
    """
    segments = translation.segments
    num_segments = len(segments)

    if num_segments == 0:
        return TranslationMetrics(
            num_segments=0,
            avg_source_length=0.0,
            avg_target_length=0.0,
            expansion_ratio=1.0,
        )

    total_source_chars = sum(len(s.source_text) for s in segments)
    total_target_chars = sum(len(s.translated_text) for s in segments)

    fitness: list[SegmentFitness] = []

    for index, segment in enumerate(segments):
        if index + 1 < num_segments:
            next_start = segments[index + 1].start_ts
        elif total_duration is not None:
            next_start = total_duration
        else:
            next_start = segment.end_ts

        budget = max(next_start - segment.start_ts - MIN_GAP_S, segment.end_ts - segment.start_ts)

        fitness.append(
            evaluate_segment_fitness(
                segment,
                budget_s=budget,
                target_language=translation.target_language,
            )
        )

    pace_ratios = [f.pace_ratio for f in fitness]
    verdicts = [f.verdict for f in fitness]

    return TranslationMetrics(
        num_segments=num_segments,
        avg_source_length=total_source_chars / num_segments,
        avg_target_length=total_target_chars / num_segments,
        expansion_ratio=(
            total_target_chars / total_source_chars
            if total_source_chars > 0
            else 1.0
        ),
        mean_pace_ratio=sum(pace_ratios) / num_segments,
        max_pace_ratio=max(pace_ratios),
        num_fits=verdicts.count("fits"),
        num_tight=verdicts.count("tight"),
        num_infeasible=verdicts.count("infeasible"),
        num_impossible=verdicts.count("impossible"),
        mean_script_ratio=sum(f.script_ratio for f in fitness) / num_segments,
        num_wrong_script=sum(1 for f in fitness if f.script_ratio < 0.8),
        num_empty=sum(1 for f in fitness if f.is_empty),
        num_copied=sum(1 for f in fitness if f.is_copy),
        num_digits_dropped=sum(1 for f in fitness if not f.digits_preserved),
        segments=fitness,
    )
