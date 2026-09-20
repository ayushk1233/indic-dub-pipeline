"""
One command from a video to a dubbed video.

    ./venv/bin/python -m src.cli --input test.mp4 --target-lang hi --job-id demo

Synthesis runs on a different machine, so a first run stops at the GPU
boundary with a bundle zip to carry there. Once the results come back, resume:

    ./venv/bin/python -m src.cli --input test.mp4 --job-id demo --from-stage import
"""

import argparse
import os
import sys
from pathlib import Path

# faster-whisper forks after tokenizers has been used, and HuggingFace prints a
# five-line warning every time it happens — dozens of identical paragraphs that
# bury the stage output a user is actually reading. Set before any import that
# reaches transformers. `setdefault`, so an explicit value still wins.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import yaml

from src.orchestrator.models import StageStatus
from src.pipeline.paths import DEFAULT_ROOT, JobPaths
from src.pipeline.runner import GPU_BOUNDARY, STAGES, PipelineRunner, RunResult


DEFAULT_CONFIG = "config/pipeline.yaml"


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="src.cli",
        description="Dub a video into an Indic language.",
    )
    parser.add_argument("--input", required=True, help="Source video or audio file.")
    parser.add_argument("--job-id", required=True, help="Names the artifact directory.")
    parser.add_argument("--target-lang", default="hi", help="Two-letter target code.")
    parser.add_argument(
        "--source-lang",
        default=None,
        help=(
            "Two-letter source code for ASR. Defaults to asr.language in the "
            "config, which is 'en'. Set it to transcribe a source that is not "
            "English; --source-lang hi --target-lang hi is the cloning leg, "
            "and skips translation."
        ),
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--artifacts-root",
        default=str(DEFAULT_ROOT),
        help="Where job directories are written.",
    )
    parser.add_argument(
        "--from-stage",
        default="preprocess",
        choices=STAGES,
        help="Resume from this stage, reusing what earlier stages already wrote.",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=1,
        help=(
            "Translation candidates per segment. Above 1 turns on length "
            "control, which picks the candidate that fits the slot."
        ),
    )
    parser.add_argument(
        "--duration-model",
        default=None,
        help="Fitted duration model JSON, used by length control.",
    )
    parser.add_argument(
        "--no-fidelity",
        action="store_true",
        help=(
            "Skip semantic scoring of candidates. Faster and avoids a 1.8 GB "
            "download, but length control then chooses on duration alone."
        ),
    )
    parser.add_argument(
        "--reference-audio",
        default=None,
        help="Voice reference for cloning. Defaults to the longest source chunk.",
    )
    parser.add_argument(
        "--reference-text",
        default=None,
        help=(
            "What is said in the reference clip, in the TARGET script. "
            "Literal text, or a path to a .json/.txt holding it. IndicF5 "
            "conditions on audio and transcript together, and a script "
            "mismatch between them is the largest measured cause of leading "
            "gibberish (§16e) — the export refuses one."
        ),
    )
    parser.add_argument(
        "--bundle",
        default=None,
        help="Bundle to import when resuming. Defaults to the job's own bundle.",
    )
    return parser


def _reference_text(value: str | None) -> str | None:
    """
    Resolve --reference-text, which may be the text itself or a file holding
    it.

    A JSON file is read for a "devanagari" key first and a "text" key second,
    which is the shape `fixtures/xlit/reference_deva.json` and
    `fixtures/reference_text.json` already use — so a reviewed transcript can
    be passed by path rather than pasted into a shell.
    """
    if not value:
        return None

    path = Path(value)

    if not path.exists():
        return value

    if path.suffix == ".json":
        import json

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        for key in ("devanagari", "text"):
            if isinstance(payload, dict) and payload.get(key):
                return payload[key]

        raise SystemExit(
            f"{path} has no 'devanagari' or 'text' key to read a transcript from."
        )

    return path.read_text(encoding="utf-8").strip()


def _load_duration_model(path: str | None, use_fidelity: bool):
    from src.eval.duration_model import DEFAULT_MODEL_PATH, DurationModelSet

    candidate = Path(path) if path else DEFAULT_MODEL_PATH

    if candidate.exists():
        print(f"Duration model: {candidate}")
        return DurationModelSet.load(candidate)

    print(
        f"Duration model: none at {candidate}, falling back to measured "
        "constant rates. Fit one with src.eval.duration_model."
    )
    return DurationModelSet()


def _load_fidelity_scorer(disabled: bool):
    if disabled:
        from src.stages.translation.fidelity import NullFidelityScorer

        return NullFidelityScorer()

    from src.stages.translation.fidelity import FidelityScorer

    return FidelityScorer()


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    cfg = load_config(args.config)

    if args.source_lang:
        # ASR's language is config, not a per-run argument, because every run
        # so far has been English. A Hindi source needs it per run, and a
        # wrong value here does not raise — Whisper transcribes Hindi speech
        # into confident English words.
        cfg.setdefault("asr", {})["language"] = args.source_lang

    paths = JobPaths(args.job_id, root=Path(args.artifacts_root)).ensure()

    duration_model = None
    fidelity_scorer = None

    if args.candidates > 1:
        duration_model = _load_duration_model(args.duration_model, not args.no_fidelity)
        fidelity_scorer = _load_fidelity_scorer(args.no_fidelity)

    runner = PipelineRunner(
        cfg=cfg,
        paths=paths,
        target_language=args.target_lang,
        num_candidates=args.candidates,
        duration_model=duration_model,
        fidelity_scorer=fidelity_scorer,
    )

    run = RunResult(job_id=args.job_id)
    start_index = STAGES.index(args.from_stage)

    def should_run(stage: str) -> bool:
        return STAGES.index(stage) >= start_index

    total_duration = None

    try:
        if should_run("preprocess"):
            print("\n[preprocess]")
            result = run.add(runner.preprocess(args.input))
            _echo(result)

            if result.status != StageStatus.DONE:
                return _finish(run, 1)

        if should_run("asr"):
            print("\n[asr]")
            _echo(run.add(runner.transcribe()))

        if should_run("translate"):
            print("\n[translate]")
            if args.candidates > 1:
                print(f"  length control on, {args.candidates} candidates per segment")
            _echo(run.add(runner.translate()))

        if should_run("export"):
            print("\n[export]")
            result = run.add(
                runner.export_bundle(
                    args.reference_audio,
                    args.input,
                    reference_text=_reference_text(args.reference_text),
                )
            )
            _echo(result)

        # Everything past here needs synthesized audio back from the GPU host.
        bundle_path = Path(args.bundle) if args.bundle else paths.bundle
        result_file = bundle_path / "output" / "synthesis_result.json"

        if not result_file.exists():
            run.stopped_at = GPU_BOUNDARY
            run.message = (
                f"\nStopped at the GPU boundary — this is step 1 of 3, and it "
                f"finished.\n\n"
                f"  2. open notebooks/kaggle_synthesis.ipynb or "
                f"notebooks/colab_synthesis.ipynb\n"
                f"     and give it  {paths.bundle_zip}\n\n"
                f"  3. unzip the result into {bundle_path / 'output'}\n"
                f"     then re-run this command with  --from-stage import"
            )
            print(f"\n{run.message}")
            return _finish(run, 0)

        from src.stages.remux import probe_duration

        total_duration = probe_duration(Path(args.input))

        if should_run("import"):
            print("\n[import]")
            imported = runner.import_bundle(bundle_path)
            print(f"  {imported.summary()}")

            for issue in imported.issues:
                print(f"    segment {issue.segment_id}: {issue.kind} — {issue.detail}")
        else:
            imported = runner.import_bundle(bundle_path)

        if should_run("assemble"):
            print("\n[assemble]")
            _echo(run.add(runner.assemble(imported, total_duration)))

        if should_run("remux"):
            print("\n[remux]")
            result = run.add(runner.remux(args.input))
            _echo(result)

        if should_run("report"):
            print("\n[report]")
            _echo(run.add(runner.report(total_duration)))

    except Exception as exc:
        # Flush first: the stages print to stdout and this prints to stderr,
        # and without it the error lands above the stage that raised it.
        sys.stdout.flush()
        print(f"\nFAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return _finish(run, 1)

    if run.ok and run.stopped_at is None:
        print(f"\nDubbed video: {paths.dubbed_video}")

    return _finish(run, 0 if run.ok else 1)


def _echo(result) -> None:
    status = result.status.value

    if result.error:
        print(f"  {status}: {result.error}")
        return

    print(f"  {status}  {result.output_path or ''}")

    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.2f}")
        else:
            print(f"    {key}: {value}")


def _finish(run: RunResult, code: int) -> int:
    failed = [s.stage_name for s in run.stages if s.status != StageStatus.DONE]

    if failed:
        print(f"\nFailed stages: {', '.join(failed)}", file=sys.stderr)

    return code


if __name__ == "__main__":
    raise SystemExit(main())
