"""
Run the whole pipeline, from a video to a dubbed video.

The stages were all built before anything connected them, so this is the spine
they were missing. It deliberately stays a straight line: each step writes a
file, the next step reads it, and any step can be the starting point because
its input is already on disk from last time.

That resumability is not a luxury here. Synthesis happens on a different
machine, so every run necessarily stops at the GPU boundary and picks up again
once the bundle comes back. Making every stage resumable costs nothing extra
once the first one is.
"""

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.orchestrator.models import StageResult, StageStatus
from src.pipeline.paths import JobPaths
from src.stages.reference import (
    SPAN_PADDING_S,
    build_reference,
    reference_ceiling,
    reference_seconds,
)


# Stage names, in execution order. `--from-stage` names one of these.
STAGES = (
    "preprocess",
    "asr",
    "translate",
    "export",
    "import",
    "assemble",
    "remux",
    "report",
)

# Stages after this one need the GPU result to exist, so a run that has not
# had one comes back here.
GPU_BOUNDARY = "import"


@dataclass
class RunResult:
    job_id: str
    stages: list[StageResult] = field(default_factory=list)
    stopped_at: str | None = None
    message: str = ""

    @property
    def ok(self) -> bool:
        return all(s.status == StageStatus.DONE for s in self.stages)

    def add(self, result: StageResult) -> StageResult:
        self.stages.append(result)
        return result


def _done(name: str, output: Path | None = None, **metrics) -> StageResult:
    return StageResult(
        stage_name=name,
        status=StageStatus.DONE,
        output_path=str(output) if output else None,
        metrics=metrics,
    )


def _failed(name: str, error: str) -> StageResult:
    return StageResult(stage_name=name, status=StageStatus.FAILED, error=error)


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return path


class PipelineRunner:
    """
    Walks the stages in order, writing each result where the next one expects it.
    """

    def __init__(
        self,
        cfg: dict,
        paths: JobPaths,
        target_language: str = "hi",
        num_candidates: int = 1,
        duration_model=None,
        fidelity_scorer=None,
    ):
        self.cfg = cfg
        self.paths = paths
        self.target_language = target_language
        self.num_candidates = num_candidates
        self.duration_model = duration_model
        self.fidelity_scorer = fidelity_scorer

    # -- individual stages ---------------------------------------------------

    def preprocess(self, input_path: str) -> StageResult:
        from src.stages.preprocess import FFmpegPreprocessStage

        return FFmpegPreprocessStage().run(
            input_path,
            self.paths.job_id,
            self.cfg,
        )

    def transcribe(self) -> StageResult:
        from src.stages.asr.faster_whisper_backend import FasterWhisperBackend
        from src.stages.asr.processor import ASRProcessor

        asr_cfg = self.cfg["asr"]

        backend = FasterWhisperBackend(
            model_name=asr_cfg["model"],
            device=asr_cfg["device"],
            compute_type=asr_cfg["compute_type"],
            language=asr_cfg["language"],
        )
        backend.load()

        started = time.perf_counter()
        transcript = ASRProcessor(backend).transcribe(self.paths.manifest)

        _write_json(self.paths.transcript, transcript.model_dump())

        return _done(
            "asr",
            self.paths.transcript,
            latency_ms=(time.perf_counter() - started) * 1000,
            num_segments=len(transcript.segments),
            language=transcript.language,
        )

    def translate(self) -> StageResult:
        from src.stages.asr.models import TranscriptResult
        from src.stages.translation.indictrans2_backend import IndicTrans2Backend
        from src.stages.translation.processor import TranslationProcessor

        with open(self.paths.transcript, "r", encoding="utf-8") as f:
            transcript = TranscriptResult(**json.load(f))

        translate_cfg = self.cfg["translate"]

        backend = IndicTrans2Backend(
            model_name=translate_cfg["model"],
            device=translate_cfg["device"],
        )
        backend.load()

        started = time.perf_counter()

        if self.num_candidates > 1:
            translation, selections = self._translate_with_length_control(
                transcript,
                backend,
            )
            _write_json(self.paths.candidates, selections)
        else:
            translation = TranslationProcessor(backend).translate(
                transcript,
                self.target_language,
            )
            selections = []

        _write_json(self.paths.translation, translation.model_dump())

        return _done(
            "translate",
            self.paths.translation,
            latency_ms=(time.perf_counter() - started) * 1000,
            num_segments=len(translation.segments),
            num_candidates=self.num_candidates,
            length_controlled=bool(selections),
        )

    def _translate_with_length_control(self, transcript, backend):
        """
        Translate each segment by generating several candidates and keeping the
        one that fits its slot without losing the meaning.
        """
        from src.eval.duration_model import DurationModelSet
        from src.stages.translation.length_control import choose_translation
        from src.stages.translation.models import (
            TranslatedSegment,
            TranslationResult,
        )
        from src.eval.translation_metrics import MIN_GAP_S

        duration_model = self.duration_model or DurationModelSet()
        segments = transcript.segments

        translated: list[TranslatedSegment] = []
        selections: list[dict] = []

        for index, segment in enumerate(segments):
            # Budget pools the pause that follows, matching the arithmetic the
            # feasibility check and the assembly cascade both use.
            if index + 1 < len(segments):
                next_start = segments[index + 1].start_ts
            else:
                next_start = segment.end_ts

            slot = max(segment.end_ts - segment.start_ts, 1e-6)
            budget = max(next_start - segment.start_ts - MIN_GAP_S, slot)

            texts = backend.translate_candidates(
                segment.text,
                transcript.language,
                self.target_language,
                num_candidates=self.num_candidates,
            )

            fidelities = None

            if self.fidelity_scorer is not None:
                fidelities = self.fidelity_scorer.score(segment.text, texts)

            selection = choose_translation(
                texts,
                budget_s=budget,
                language=self.target_language,
                duration_model=duration_model,
                fidelities=fidelities,
            )

            translated.append(
                TranslatedSegment(
                    segment_id=segment.segment_id,
                    chunk_id=segment.chunk_id,
                    start_ts=segment.start_ts,
                    end_ts=segment.end_ts,
                    source_text=segment.text,
                    translated_text=selection.chosen.text,
                    source_language=transcript.language,
                    target_language=self.target_language,
                )
            )

            selections.append(
                {
                    "segment_id": segment.segment_id,
                    "budget_s": budget,
                    "reason": selection.reason,
                    "fidelity_cost": selection.fidelity_cost,
                    "fit_gain": selection.fit_gain,
                    "chosen": {
                        "text": selection.chosen.text,
                        "chars": len(selection.chosen.text),
                        "predicted_duration_s": selection.chosen.predicted_duration_s,
                        "fit_ratio": selection.chosen.fit_ratio,
                        "fidelity": selection.chosen.fidelity,
                    },
                    "candidates": [
                        {
                            "text": c.text,
                            "chars": len(c.text),
                            "fit_ratio": c.fit_ratio,
                            "fidelity": c.fidelity,
                            "off_language": c.off_language,
                        }
                        for c in selection.candidates
                    ],
                }
            )

        return (
            TranslationResult(
                job_id=transcript.job_id,
                source_language=transcript.language,
                target_language=self.target_language,
                segments=translated,
            ),
            selections,
        )

    def export_bundle(
        self,
        reference_audio: Path | None = None,
        source_media: str | Path | None = None,
    ) -> StageResult:
        from src.stages.translation.models import TranslationResult
        from src.stages.tts.bundle.exporter import BundleExporter
        from src.stages.tts.processor import TTSProcessor

        with open(self.paths.translation, "r", encoding="utf-8") as f:
            translation = TranslationResult(**json.load(f))

        if reference_audio:
            reference, reference_text = Path(reference_audio), None
        else:
            reference, reference_text = self._build_reference(source_media)

        request = TTSProcessor().build_request(
            translation,
            reference_audio="request/reference.wav",
            reference_text=reference_text,
        )

        exporter = BundleExporter()
        exporter.export(request, reference, self.paths.bundle)
        archive = exporter.package_bundle(self.paths.bundle)

        return _done(
            "export",
            archive,
            num_segments=len(request.segments),
            reference_audio=str(reference),
        )

    def _build_reference(
        self, source_media: str | Path | None
    ) -> tuple[Path, str | None]:
        """
        Build the reference clip and its transcript.

        Returns the transcript alongside the audio because IndicF5 conditions
        on both and cannot clone from audio alone. XTTS ignores it.
        """
        if self.paths.reference_audio.exists():
            return self.paths.reference_audio, self._transcript_for(None)

        spans = self._reference_spans()

        if source_media is None or not spans:
            return self._pick_reference(source_media), None

        try:
            path = build_reference(
                source_media, spans, self.paths.reference_audio
            )
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
            # A worse reference is recoverable; a failed export is not.
            return self._pick_reference(source_media), None

        return path, self._transcript_for(spans)

    def _reference_spans(self) -> list[tuple[float, float]]:
        """
        Choose which stretches of the source become the reference.

        Longest first, because long uninterrupted speech conditions better
        than the same seconds chopped up, then replayed in time order so the
        clip sounds like continuous speech rather than a shuffle.
        """
        entries = self._manifest_entries()

        if not entries:
            return []

        spans = sorted(
            ((float(e["start_ts"]), float(e["end_ts"])) for e in entries),
            key=lambda span: span[1] - span[0],
            reverse=True,
        )

        target = self._reference_seconds()
        ceiling = reference_ceiling(self._tts_model())

        # build_reference pads every span outward on both sides, so the file
        # is longer than the spans by 2 * SPAN_PADDING_S each. Selecting on
        # the raw span length and then landing over the ceiling is exactly
        # the mistake this is here to prevent.
        overhead = 2 * SPAN_PADDING_S

        chosen: list[tuple[float, float]] = []
        total = 0.0

        for span in spans:
            if total >= target:
                break

            predicted = total + (span[1] - span[0]) + overhead

            if ceiling is not None and predicted > ceiling:
                if chosen:
                    # A shorter span later in the list may still fit. Taking
                    # this one would overshoot by up to its whole length.
                    continue
                # Nothing chosen yet and even the longest span is too long,
                # so trim it rather than ship a reference the model will clip
                # out from under its own transcript.
                span = (span[0], span[0] + max(ceiling - overhead, 0.0))
                predicted = ceiling

            chosen.append(span)
            total = predicted

        return sorted(chosen)

    def _reference_seconds(self) -> float:
        """
        How much reference audio this job's TTS model wants.

        Model-specific because IndicF5 silently clips a reference over 15 s
        without shortening its transcript (FINDINGS §5d), and XTTS wants more
        than IndicF5 can take. See `src.stages.reference.REFERENCE_SECONDS`.
        """
        return reference_seconds(self._tts_model())

    def _tts_model(self) -> str | None:
        return (self.cfg.get("tts") or {}).get("model")

    def _manifest_entries(self) -> list[dict]:
        if not self.paths.manifest.exists():
            return []

        try:
            with open(self.paths.manifest, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except (OSError, json.JSONDecodeError):
            return []

        if not isinstance(entries, list):
            return []

        return [e for e in entries if isinstance(e, dict) and "start_ts" in e]

    def _transcript_for(self, spans: list[tuple[float, float]] | None) -> str | None:
        """
        The source-language text spoken during `spans`, or all of it when
        `spans` is None.
        """
        if not self.paths.transcript.exists():
            return None

        try:
            with open(self.paths.transcript, "r", encoding="utf-8") as f:
                transcript = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

        parts = []

        for segment in transcript.get("segments", []):
            start = float(segment.get("start_ts", 0.0))
            end = float(segment.get("end_ts", 0.0))

            if spans is not None and not any(
                start < span_end and end > span_start
                for span_start, span_end in spans
            ):
                continue

            text = (segment.get("text") or "").strip()

            if text:
                parts.append(text)

        return " ".join(parts) or None

    def _pick_reference(self, source_media: str | Path | None = None) -> Path:
        """
        Choose a voice reference from the speaker's own audio.

        The longest chunk is used, because the synthesis research in
        colab/xtts.md found that greedy decoding degrades with short reference
        audio, and the longest available clip is the safest default.

        The chunks themselves are the wrong file to hand XTTS. They are cut at
        the ASR sample rate, 16 kHz, and XTTS conditions at 22.05 kHz, so it
        upsamples them and clones a voice with nothing above 8 kHz. That band
        is where much of a speaker's identity lives. They are also at whatever
        level the source happened to be. Measured on the first real run: a
        16 kHz reference peaking at 0.21 produced a mean speaker similarity of
        0.478 against a floor of 0.75.

        So when the source media is available, re-cut the same span from it at
        REFERENCE_SAMPLE_RATE and normalize the level. Falling back to the raw
        chunk keeps the older callers working.
        """
        if self.paths.reference_audio.exists():
            return self.paths.reference_audio

        chunks = sorted(self.paths.chunks.glob("chunk_*.wav"))

        if not chunks:
            raise FileNotFoundError(
                f"No reference audio and no chunks to build one from in "
                f"{self.paths.chunks}. Run preprocessing first."
            )

        chunk = max(chunks, key=lambda p: p.stat().st_size)

        if source_media is None:
            return chunk

        span = self._span_of(chunk)

        if span is None:
            return chunk

        try:
            return build_reference(
                source_media, span, self.paths.reference_audio
            )
        except (OSError, RuntimeError, subprocess.SubprocessError):
            # A bad reference is recoverable; a failed export is not.
            return chunk

    def _span_of(self, chunk: Path) -> tuple[float, float] | None:
        """
        Find a chunk's start and end in the source, from the manifest.
        """
        if not self.paths.manifest.exists():
            return None

        try:
            with open(self.paths.manifest, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

        if not isinstance(entries, list):
            return None

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if Path(str(entry.get("chunk_path", ""))).name == chunk.name:
                return float(entry["start_ts"]), float(entry["end_ts"])

        return None

    def import_bundle(self, bundle_path: Path | None = None):
        from src.stages.tts.bundle.importer import BundleImporter

        path = Path(bundle_path) if bundle_path else self.paths.bundle

        return BundleImporter().import_bundle(path)

    def assemble(self, imported, total_duration: float) -> StageResult:
        from src.stages.assemble import assemble, write_timeline

        started = time.perf_counter()

        result = assemble(imported, total_duration, self.paths.dubbed_audio)
        write_timeline(result, self.paths.timeline)

        return _done(
            "assemble",
            self.paths.dubbed_audio,
            latency_ms=(time.perf_counter() - started) * 1000,
            num_placed=result.num_placed,
            num_stretched=result.num_stretched,
            num_drifted=result.num_drifted,
            num_missing=result.num_missing,
            max_tempo=result.max_tempo_used,
        )

    def remux(self, video_path: str) -> StageResult:
        from src.stages.remux import remux

        result = remux(
            Path(video_path),
            self.paths.dubbed_audio,
            self.paths.dubbed_video,
        )

        if not result.in_sync:
            return _failed(
                "remux",
                f"audio and video durations differ by {result.drift_s:.3f}s",
            )

        return _done(
            "remux",
            self.paths.dubbed_video,
            drift_s=result.drift_s,
            video_duration_s=result.video_duration_s,
        )

    def report(self, total_duration: float | None = None) -> StageResult:
        from src.eval.harness import build_report, write_report

        report = build_report(
            self.paths.job_dir,
            bundle_dir=self.paths.bundle if self.paths.bundle.exists() else None,
            total_duration=total_duration,
        )

        json_path, _ = write_report(self.paths.job_dir, report)

        return _done("report", json_path, stages=sorted(report.get("stages", {})))
