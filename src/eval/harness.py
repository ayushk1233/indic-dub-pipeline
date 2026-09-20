"""
Aggregate per-stage metrics for one job into a report you can read.

Every stage is optional. The harness reports on whatever artifacts exist,
so it is useful long before the pipeline runs end to end.
"""

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

from src.eval.preprocess_metrics import evaluate_manifest
from src.eval.translation_metrics import evaluate_translation
from src.eval.tts_metrics import evaluate_synthesis
from src.stages.translation.models import TranslatedSegment, TranslationResult
from src.stages.tts.models import SynthesisRequest, SynthesisResult


def _load(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _serialize(value):
    if is_dataclass(value):
        return asdict(value)
    return value


def _is_preprocess_manifest(path: Path) -> bool:
    """
    Is this the preprocess chunk manifest, or some other manifest.json?

    The harness is often pointed at a bundle directory, which carries its own
    manifest.json describing the bundle version and paths. That file is a JSON
    object, not the list of segment entries this stage reports on, and feeding
    it to `evaluate_manifest` iterates the keys and fails on a string index.
    Existence is not enough; check the shape.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    if not isinstance(entries, list):
        return False

    return not entries or all(
        isinstance(e, dict) and "start_ts" in e and "end_ts" in e for e in entries
    )


def _translation_from_request(request: SynthesisRequest) -> TranslationResult:
    """
    Derive a translation result from a synthesis request.

    The request carries the translated text and the slot it must fill, which
    is everything the feasibility check needs. This lets the pace metrics run
    against an existing bundle before the pipeline persists translation.json.
    """
    return TranslationResult(
        job_id=request.job_id,
        source_language="en",
        target_language=request.language,
        segments=[
            TranslatedSegment(
                segment_id=s.segment_id,
                chunk_id=s.chunk_id,
                start_ts=s.start_ts,
                end_ts=s.end_ts,
                source_text="",
                translated_text=s.text,
                source_language="en",
                target_language=request.language,
            )
            for s in request.segments
        ],
    )


# A segment whose round-trip CER is worse than this is reported as bad rather
# than merely noisy. The measured floor for this speaker is ~0.05 (the project notes
# §4d); the digit corruption that prompted this check scored far above it,
# with whole words replaced. Halfway between is a threshold that catches
# gibberish without firing on ASR variance.
INTELLIGIBILITY_CER_LIMIT = 0.25


def build_report(
    job_dir: Path,
    bundle_dir: Path | None = None,
    total_duration: float | None = None,
    asr_backend=None,
) -> dict:
    """
    Collect every metric available for a job.

    `asr_backend` is optional and opt-in because it costs a 3 GB model and
    minutes of CPU. Without it the report is timing only — which is exactly
    how a run with nine gibberish segments once passed every check it had. The
    cheap guard against that case lives in `BundleExporter`; this is the one
    that reads the audio.
    """
    job_dir = Path(job_dir)

    if bundle_dir is None:
        candidate = job_dir / "tts_bundle"
        bundle_dir = candidate if candidate.exists() else None

    report: dict = {"job_dir": str(job_dir), "stages": {}}

    manifest_path = job_dir / "manifest.json"
    if manifest_path.exists() and _is_preprocess_manifest(manifest_path):
        report["stages"]["preprocess"] = _serialize(
            evaluate_manifest(manifest_path, total_duration=total_duration)
        )

    request = result = None

    if bundle_dir is not None:
        request_path = Path(bundle_dir) / "request" / "synthesis_request.json"
        result_path = Path(bundle_dir) / "output" / "synthesis_result.json"

        if request_path.exists():
            request = SynthesisRequest(**_load(request_path))
        if result_path.exists():
            result = SynthesisResult(**_load(result_path))

    translation_path = job_dir / "translation.json"

    if translation_path.exists():
        translation = TranslationResult(**_load(translation_path))
    elif request is not None:
        translation = _translation_from_request(request)
    else:
        translation = None

    if translation is not None:
        report["stages"]["translation"] = _serialize(
            evaluate_translation(translation, total_duration=total_duration)
        )

    if request is not None and result is not None:
        report["stages"]["tts"] = _serialize(
            evaluate_synthesis(
                request,
                result,
                bundle_dir=bundle_dir,
                total_duration=total_duration,
            )
        )

        if asr_backend is not None:
            report["stages"]["content"] = _evaluate_content(
                request, result, bundle_dir, asr_backend
            )

    return report


def _evaluate_content(request, result, bundle_dir, asr_backend) -> dict:
    """
    Transcribe the synthesized audio and score it against what was asked for.

    The only metric here that catches gibberish, truncation and repetition —
    all three produce audio that every timing measure scores as healthy.
    """
    from src.eval.tts_metrics import evaluate_intelligibility

    scores = evaluate_intelligibility(request, result, Path(bundle_dir), asr_backend)

    if not scores:
        return {"num_scored": 0, "segments": [], "num_bad": 0}

    ordered = sorted(scores.items())
    bad = [segment_id for segment_id, cer in ordered if cer > INTELLIGIBILITY_CER_LIMIT]

    return {
        "num_scored": len(ordered),
        "cer_limit": INTELLIGIBILITY_CER_LIMIT,
        "mean_cer": sum(cer for _, cer in ordered) / len(ordered),
        "max_cer": max(cer for _, cer in ordered),
        "num_bad": len(bad),
        "bad_segments": bad,
        "segments": [
            {"segment_id": segment_id, "cer": cer} for segment_id, cer in ordered
        ],
    }


def render_report(report: dict) -> str:
    """
    Render the report as a plain table.
    """
    lines: list[str] = []
    stages = report["stages"]

    lines.append("=" * 78)
    lines.append(f"DUBBING QC REPORT  {report['job_dir']}")
    lines.append("=" * 78)

    if not stages:
        lines.append("\nNo stage artifacts found.")
        return "\n".join(lines)

    pre = stages.get("preprocess")
    if pre:
        lines.append("\nPREPROCESS")
        lines.append(
            f"  {pre['num_segments']} segments, "
            f"{pre['total_speech_s']:.1f}s speech of {pre['total_duration_s']:.1f}s "
            f"({pre['speech_ratio'] * 100:.0f}% speech)"
        )
        lines.append(
            f"  segment length  min {pre['min_segment_s']:.2f}s  "
            f"median {pre['median_segment_s']:.2f}s  max {pre['max_segment_s']:.2f}s"
        )
        lines.append(
            f"  gap between     min {pre['min_gap_s']:.2f}s  "
            f"median {pre['median_gap_s']:.2f}s  max {pre['max_gap_s']:.2f}s"
        )
        warnings = []
        if pre["num_short_segments"]:
            warnings.append(f"{pre['num_short_segments']} too short")
        if pre["num_oversized_segments"]:
            warnings.append(f"{pre['num_oversized_segments']} over ASR window")
        if pre["num_missing_chunks"]:
            warnings.append(f"{pre['num_missing_chunks']} chunks missing")
        if warnings:
            lines.append("  WARN: " + ", ".join(warnings))

    tr = stages.get("translation")
    if tr:
        lines.append("\nTRANSLATION  (can the text be spoken in the time available?)")
        lines.append(
            f"  expansion {tr['expansion_ratio']:.2f}x   "
            f"pace mean {tr['mean_pace_ratio']:.2f}x  max {tr['max_pace_ratio']:.2f}x"
        )
        lines.append(
            f"  fits {tr['num_fits']}   tight {tr['num_tight']}   "
            f"infeasible {tr['num_infeasible']}   impossible {tr['num_impossible']}"
        )
        if tr["num_wrong_script"] or tr["num_empty"] or tr["num_copied"]:
            lines.append(
                f"  WARN: wrong script {tr['num_wrong_script']}, "
                f"empty {tr['num_empty']}, untranslated {tr['num_copied']}"
            )
        lines.append("")
        lines.append(
            f"  {'seg':>4} {'chars':>6} {'slot':>7} {'budget':>7} "
            f"{'need cps':>9} {'pace':>6}  verdict"
        )
        for s in tr["segments"]:
            lines.append(
                f"  {s['segment_id']:>4} {s['target_chars']:>6} "
                f"{s['slot_s']:>6.2f}s {s['budget_s']:>6.2f}s "
                f"{s['required_cps']:>9.1f} {s['pace_ratio']:>5.2f}x  {s['verdict']}"
            )

    tts = stages.get("tts")
    if tts:
        lines.append("\nSYNTHESIS  (voice cloning and delivered pace)")
        lines.append(
            f"  {tts['num_done']} done, {tts['num_failed']} failed, "
            f"{tts['num_missing']} missing   model {tts['model_id']}"
        )
        if tts["params"]:
            lines.append(f"  params {tts['params']}")

        similarity = tts["mean_speaker_similarity"]
        if similarity is not None:
            lines.append(
                f"  speaker similarity  mean {similarity:.3f}  "
                f"min {tts['min_speaker_similarity']:.3f}  "
                f"below floor {tts['num_below_similarity_floor']}"
            )
        else:
            lines.append("  speaker similarity  not recorded")

        lines.append(
            f"  fits {tts['num_fits']}   stretchable {tts['num_stretchable']}   "
            f"overrun {tts['num_overrun']}   "
            f"mean duration ratio {tts['mean_duration_ratio']:.2f}x"
        )
        lines.append("")
        lines.append(
            f"  {'seg':>4} {'dur':>7} {'budget':>7} {'ratio':>6} {'tempo':>6} "
            f"{'cps':>6} {'sim':>6} {'tok':>5}  flags"
        )
        for s in tts["segments"]:
            sim = s["speaker_similarity"]
            lines.append(
                f"  {s['segment_id']:>4} {s['duration_s']:>6.2f}s "
                f"{s['budget_s']:>6.2f}s {s['duration_ratio']:>5.2f}x "
                f"{s['required_tempo']:>5.2f}x {s['realized_cps']:>6.1f} "
                f"{(f'{sim:.3f}' if sim is not None else '    -'):>6} "
                f"{str(s['gpt_tokens'] or '-'):>5}  {','.join(s['flags'])}"
            )

    lines.append("")
    lines.append("=" * 78)

    content = stages.get("content")
    if content and content.get("num_scored"):
        lines.append("")
        lines.append("CONTENT  (what the audio actually says)")
        lines.append(
            f"  {content['num_scored']} scored   "
            f"mean CER {content['mean_cer']:.3f}   "
            f"max {content['max_cer']:.3f}   "
            f"limit {content['cer_limit']:.2f}"
        )
        if content["num_bad"]:
            lines.append(
                f"  {content['num_bad']} SEGMENT(S) OVER THE LIMIT: "
                f"{', '.join(str(i) for i in content['bad_segments'])}"
            )
            lines.append(
                "  Transcribe them against the request text before shipping; "
                "timing metrics do not see this."
            )
        else:
            lines.append("  every segment within the limit")

    return "\n".join(lines)


def write_report(job_dir: Path, report: dict) -> tuple[Path, Path]:
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    json_path = job_dir / "report.json"
    text_path = job_dir / "report.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(text_path, "w", encoding="utf-8") as f:
        f.write(render_report(report))
        f.write("\n")

    return json_path, text_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Report dubbing QC metrics for a job.")
    parser.add_argument("--job-dir", required=True, help="artifacts/<job_id>")
    parser.add_argument("--bundle-dir", default=None, help="Defaults to <job-dir>/tts_bundle")
    parser.add_argument(
        "--total-duration",
        type=float,
        default=None,
        help="Source media duration in seconds, so the last segment can pool its tail.",
    )
    parser.add_argument("--no-write", action="store_true", help="Print only.")
    parser.add_argument(
        "--check-content",
        action="store_true",
        help=(
            "Transcribe the synthesized audio and score it against the request "
            "text. The only check that catches gibberish; costs an ASR model "
            "and minutes of CPU, so it is opt-in."
        ),
    )
    args = parser.parse_args()

    asr_backend = None

    if args.check_content:
        import yaml

        from src.stages.asr.faster_whisper_backend import FasterWhisperBackend

        with open("config/pipeline.yaml", "r", encoding="utf-8") as f:
            asr_cfg = yaml.safe_load(f)["asr"]

        asr_backend = FasterWhisperBackend(
            model_name=asr_cfg["model"],
            device=asr_cfg["device"],
            compute_type=asr_cfg["compute_type"],
            language=asr_cfg.get("language", "hi"),
        )
        asr_backend.load()

    report = build_report(
        Path(args.job_dir),
        bundle_dir=Path(args.bundle_dir) if args.bundle_dir else None,
        total_duration=args.total_duration,
        asr_backend=asr_backend,
    )

    print(render_report(report))

    if not args.no_write:
        json_path, text_path = write_report(Path(args.job_dir), report)
        print(f"\nWritten to {json_path} and {text_path}")


if __name__ == "__main__":
    main()
