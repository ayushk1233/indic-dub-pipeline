"""
Measure real speaking rates and dub expansion from FLEURS.

The pipeline's feasibility verdicts all rest on one number per language: how
many characters a speaker gets through in a second. Those numbers were typed
in by hand and the code said so. This script replaces them with measurements,
and records the provenance so the next person can tell the difference.

    ./venv/bin/python -m src.data.measure --languages hi ta bn mr
"""

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.data.fleurs import SpeechSample, load_samples, pair_by_sentence


OUTPUT_DIR = Path("artifacts") / "measurements"


@dataclass
class RateStats:
    """
    Distribution of speaking rate for one language.
    """

    language: str
    num_samples: int

    median_cps: float
    mean_cps: float
    p25_cps: float
    p75_cps: float

    median_duration_s: float
    median_chars: float

    # Least-squares fit of duration against character count. The slope is the
    # articulation rate with recording overhead removed; the intercept is that
    # overhead. Reporting both is what separates a measured rate from a guess.
    seconds_per_char: float
    intercept_s: float


@dataclass
class ExpansionStats:
    """
    How much longer the same sentence takes in the target language.
    """

    source_language: str
    target_language: str
    num_pairs: int

    median_duration_ratio: float
    p25_duration_ratio: float
    p75_duration_ratio: float

    median_char_ratio: float


@dataclass
class MeasurementReport:
    split: str
    limit: int | None
    rates: list[RateStats] = field(default_factory=list)
    expansions: list[ExpansionStats] = field(default_factory=list)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(fraction * (len(ordered) - 1)), len(ordered) - 1)
    return ordered[index]


def _linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """
    Ordinary least squares slope and intercept, without pulling in numpy for
    two sums. Returns (slope, intercept).
    """
    n = len(xs)

    if n < 2:
        return 0.0, 0.0

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    variance = sum((x - mean_x) ** 2 for x in xs)

    if variance == 0:
        return 0.0, mean_y

    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / variance

    return slope, mean_y - slope * mean_x


def measure_rate(samples: list[SpeechSample]) -> RateStats:
    rates = [s.cps for s in samples if s.cps > 0]
    durations = [s.duration_s for s in samples]
    chars = [float(s.chars) for s in samples]

    slope, intercept = _linear_fit(chars, durations)

    return RateStats(
        language=samples[0].language if samples else "unknown",
        num_samples=len(samples),
        median_cps=statistics.median(rates) if rates else 0.0,
        mean_cps=statistics.fmean(rates) if rates else 0.0,
        p25_cps=_percentile(rates, 0.25),
        p75_cps=_percentile(rates, 0.75),
        median_duration_s=statistics.median(durations) if durations else 0.0,
        median_chars=statistics.median(chars) if chars else 0.0,
        seconds_per_char=slope,
        intercept_s=intercept,
    )


def measure_expansion(
    source: list[SpeechSample],
    target: list[SpeechSample],
) -> ExpansionStats:
    pairs = pair_by_sentence(source, target)

    duration_ratios = [
        tgt.duration_s / src.duration_s
        for src, tgt in pairs
        if src.duration_s > 0
    ]
    char_ratios = [
        tgt.chars / src.chars
        for src, tgt in pairs
        if src.chars > 0
    ]

    return ExpansionStats(
        source_language=source[0].language if source else "unknown",
        target_language=target[0].language if target else "unknown",
        num_pairs=len(pairs),
        median_duration_ratio=(
            statistics.median(duration_ratios) if duration_ratios else 0.0
        ),
        p25_duration_ratio=_percentile(duration_ratios, 0.25),
        p75_duration_ratio=_percentile(duration_ratios, 0.75),
        median_char_ratio=statistics.median(char_ratios) if char_ratios else 0.0,
    )


def render(report: MeasurementReport) -> str:
    lines: list[str] = []

    lines.append("=" * 78)
    lines.append(
        f"MEASURED SPEAKING RATES   FLEURS {report.split} split"
        + (f", capped at {report.limit} samples" if report.limit else "")
    )
    lines.append("=" * 78)
    lines.append("")
    lines.append(
        f"  {'lang':>5} {'n':>5} {'median':>8} {'mean':>7} "
        f"{'p25':>7} {'p75':>7} {'s/char':>8} {'overhead':>9}"
    )

    for rate in report.rates:
        lines.append(
            f"  {rate.language:>5} {rate.num_samples:>5} "
            f"{rate.median_cps:>7.2f}  {rate.mean_cps:>6.2f} "
            f"{rate.p25_cps:>6.2f}  {rate.p75_cps:>6.2f} "
            f"{rate.seconds_per_char:>8.4f} {rate.intercept_s:>8.2f}s"
        )

    lines.append("")
    lines.append("  median/mean/p25/p75 are characters per second over the whole clip.")
    lines.append("  s/char is the fitted articulation rate, overhead the fitted intercept.")

    if report.expansions:
        lines.append("")
        lines.append("-" * 78)
        lines.append("DUB EXPANSION   same sentence, both languages, matched by id")
        lines.append("-" * 78)
        lines.append("")
        lines.append(
            f"  {'pair':>9} {'n':>5} {'duration':>9} {'p25':>7} "
            f"{'p75':>7} {'chars':>8}"
        )

        for exp in report.expansions:
            pair = f"{exp.source_language}>{exp.target_language}"
            lines.append(
                f"  {pair:>9} {exp.num_pairs:>5} "
                f"{exp.median_duration_ratio:>8.3f}x "
                f"{exp.p25_duration_ratio:>6.3f}x {exp.p75_duration_ratio:>6.3f}x "
                f"{exp.median_char_ratio:>7.3f}x"
            )

        lines.append("")
        lines.append("  duration above 1.0 means the dub needs more time than the slot holds.")

    lines.append("")
    lines.append("=" * 78)

    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Measure speaking rates and dub expansion from FLEURS."
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["hi"],
        help="Two-letter target codes, e.g. hi ta bn mr.",
    )
    parser.add_argument("--source", default="en", help="Source language code.")
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap samples per language. Omit to scan the whole split.",
    )
    args = parser.parse_args()

    languages = [args.source] + [
        lang for lang in args.languages if lang != args.source
    ]

    loaded: dict[str, list[SpeechSample]] = {}

    for language in languages:
        print(f"Loading {language}...", flush=True)

        try:
            loaded[language] = load_samples(
                language,
                split=args.split,
                limit=args.limit,
            )
        except Exception as exc:
            # Streaming a language can fail on a network timeout partway
            # through a long run. Losing the languages already measured to
            # one flaky download would be the wrong trade, so record the
            # failure and keep the rest.
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            loaded[language] = []
            continue

        print(f"  {len(loaded[language])} sentences", flush=True)

    report = MeasurementReport(split=args.split, limit=args.limit)

    for language in languages:
        if loaded[language]:
            report.rates.append(measure_rate(loaded[language]))

    source_samples = loaded.get(args.source, [])

    for language in languages:
        if language == args.source or not loaded[language] or not source_samples:
            continue
        report.expansions.append(
            measure_expansion(source_samples, loaded[language])
        )

    rendered = render(report)
    print()
    print(rendered)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "speaking_rates.json"
    text_path = OUTPUT_DIR / "speaking_rates.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, ensure_ascii=False, indent=2)

    with open(text_path, "w", encoding="utf-8") as f:
        f.write(rendered)
        f.write("\n")

    print(f"\nWritten to {json_path} and {text_path}")


if __name__ == "__main__":
    main()
