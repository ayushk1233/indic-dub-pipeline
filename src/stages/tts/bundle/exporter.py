import json
import shutil
import wave
import zipfile
from datetime import datetime
from pathlib import Path

from src.stages.tts.bundle.models import (
    BundleManifest,
    BundleMetadata,
    BundlePaths,
)
from src.eval.translation_metrics import script_ratio
from src.stages.tts.models import SynthesisRequest
from src.text.normalize import unspeakable_digits


# A reference transcript below this fraction of target-script letters is
# refused.
#
# The threshold is set from the two real cases, not from a round number.
# hi_dub's own transcript scores 0.97 with two Latin loanwords in it, because
# Hindi ASR writes `project` and `fit` that way; en_dub's English transcript
# scores 0.00. What this guard exists to catch is the second kind, and the gap
# between them is enormous, so it sits low enough that ordinary ASR output can
# never trip it. A guard that fires on good input is one somebody disables.
#
# Latin islands inside an otherwise Devanagari transcript are a smaller, real
# problem, and they are handled where the table lives — the worker folds them
# through fixtures/loanwords_hi.json and reports what survives.
MIN_REFERENCE_SCRIPT_RATIO = 0.80

# Characters per second of reference audio, above which the transcript cannot
# be describing that clip. Measured natural Hindi is 12.19 cps and the fastest
# this project has recorded from the model is about 18, so 25 is double the
# natural rate and still well clear of anything real. The case it catches ran
# at 45.6.
MAX_REFERENCE_CPS = 25.0


class UnspeakableText(ValueError):
    """
    A segment carries text the model cannot pronounce.

    Raised rather than warned because the failure it prevents is silent: a
    digit synthesizes into confident nonsense that every timing metric scores
    as healthy. One run shipped two dubbed videos that way.
    """


class ReferenceScriptMismatch(ValueError):
    """
    The reference transcript is not in the script the model will generate.

    IndicF5 conditions on `ref_audio + ref_text` as one sequence and then
    strips exactly `ref_audio_len` frames off the front, with no alignment
    check behind the slice. If it could not align the transcript to the audio,
    the remainder is spoken at the start of the kept region — §5.

    §16e measured the cost directly: the same speaker and audio with a Latin
    reference transcript produced a leading prefix on 4 clips in 24 and CER
    0.080; with a Devanagari transcript, 0 in 24 and 0.052. It is the largest
    single quality lever in this pipeline and it fails silently, so it is a
    refusal rather than a warning. A run has already shipped this way.
    """


class ReferenceLengthMismatch(ValueError):
    """
    The reference transcript describes more speech than the clip contains.

    IndicF5 is handed `ref_audio + ref_text` as one sequence and then has
    exactly `ref_audio_len` frames stripped off the front, with no alignment
    check. A transcript far longer than its audio cannot be spoken inside
    those frames, so the remainder lands in the generated region — and past
    some margin the generation stops tracking the requested text at all.

    This shipped. `_build_reference` passed `None` to `_transcript_for` when
    the reference clip already existed, which means "the whole media", and a
    13.9s clip went out with a 634-character transcript at 45.6 cps. Every
    segment of that job came back as fluent Hindi assembled from the wrong
    sentences, with one clip repeating a single syllable fourteen times.
    """


class BundleExporter:
    def export(
        self,
        request: SynthesisRequest,
        reference_audio: Path,
        output_dir: Path,
    ) -> Path:
        self._refuse_unspeakable_text(request)
        self._refuse_mismatched_reference(request)
        self._refuse_overlong_reference(request, reference_audio)

        # Create the directory structure
        request_dir = output_dir / "request"
        output_sub_dir = output_dir / "output"
        logs_dir = output_dir / "logs"

        # Clear any previous run's output before re-exporting. Two reasons,
        # both of which have bitten this project: `package_bundle` zips
        # everything under the bundle root, so leftover audio rides along to
        # the GPU host; and a segment that fails there leaves the stale file
        # in place, where the importer accepts it as a real result. A bundle
        # must describe exactly one synthesis attempt.
        for stale_dir in (output_sub_dir, logs_dir):
            if stale_dir.exists():
                shutil.rmtree(stale_dir)

        request_dir.mkdir(parents=True, exist_ok=True)
        output_sub_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Copy reference audio
        bundle_ref_audio = request_dir / "reference.wav"
        shutil.copy2(reference_audio, bundle_ref_audio)

        # Write synthesis request
        bundle_request_json = request_dir / "synthesis_request.json"
        with open(bundle_request_json, "w", encoding="utf-8") as f:
            json.dump(
                request.model_dump(),
                f,
                ensure_ascii=False,
                indent=2,
            )

        # Build and write manifest
        manifest = BundleManifest(
            metadata=BundleMetadata(
                # 1.1 adds SynthesisRequest.reference_text.
                # 1.2 adds SynthesisSegment.budget_s. Both are additive and
                # optional, so an older reader ignores the extra field and a
                # newer one falls back when it is absent.
                bundle_version="1.2",
                job_id=request.job_id,
                created_at=datetime.utcnow().isoformat() + "Z",
            ),
            paths=BundlePaths(
                request_json="request/synthesis_request.json",
                reference_audio="request/reference.wav",
            ),
        )

        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                manifest.model_dump(),
                f,
                ensure_ascii=False,
                indent=2,
            )

        return output_dir

    def _refuse_unspeakable_text(self, request: SynthesisRequest) -> None:
        """
        Stop a bundle whose text still contains digits.

        `normalize_for_speech` has no ceiling, so a digit reaching here means
        the text arrived on a path that did not normalise it — not that the
        number was too large. IndicF5 renders a digit as noise while
        corrupting the head of the segment with it. Failing the export is
        cheap. The alternative is finding out by ear after a GPU round trip,
        which is how this check came to exist.
        """
        offenders = [
            (segment.segment_id, digits)
            for segment in request.segments
            if (digits := unspeakable_digits(segment.text))
        ]

        if offenders:
            detail = "; ".join(
                f"segment {segment_id}: {', '.join(digits)}"
                for segment_id, digits in offenders
            )
            raise UnspeakableText(
                f"{len(offenders)} segment(s) still contain digits the model "
                f"cannot say — {detail}. Spell them out before exporting; see "
                f"src/text/normalize.py."
            )

    def _refuse_mismatched_reference(self, request: SynthesisRequest) -> None:
        """
        Stop a bundle whose reference transcript is in the wrong script.

        The pipeline builds this transcript by transcribing the reference
        clip, so for an `en -> hi` job it comes back in Latin — describing
        English audio to a model about to generate Devanagari. There is no
        automatic English-to-Devanagari transliteration in this repo yet
        (§4f), so the fix is to supply a reviewed pair with
        `--reference-audio` and `--reference-text`.
        """
        text = (request.reference_text or "").strip()

        if not text:
            return

        ratio = script_ratio(text, request.language)

        if ratio < MIN_REFERENCE_SCRIPT_RATIO:
            raise ReferenceScriptMismatch(
                f"The reference transcript is {ratio:.0%} in the script for "
                f"'{request.language}' and needs at least "
                f"{MIN_REFERENCE_SCRIPT_RATIO:.0%}. IndicF5 conditions on the "
                f"reference audio and its transcript together, and a script "
                f"boundary between them is the largest measured cause of "
                f"invented speech at the start of a clip. Pass a reviewed "
                f"pair with --reference-audio and --reference-text. "
                f"Got: {text[:70]!r}"
            )

    def _refuse_overlong_reference(
        self, request: SynthesisRequest, reference_audio: Path
    ) -> None:
        """
        Stop a bundle whose reference transcript is too long for its audio.

        Measured in characters per second rather than absolute length,
        because the reference clip's duration is model-specific (10s for
        IndicF5, 25s for the earlier baseline) and the transcript has to scale with it.
        """
        text = (request.reference_text or "").strip()

        if not text:
            return

        try:
            with wave.open(str(reference_audio), "rb") as handle:
                seconds = handle.getnframes() / handle.getframerate()
        except (OSError, wave.Error):
            # Not a readable wav. The reference is checked elsewhere; this
            # guard has nothing to measure against and says nothing.
            return

        if seconds <= 0:
            return

        cps = len(text) / seconds

        if cps > MAX_REFERENCE_CPS:
            raise ReferenceLengthMismatch(
                f"The reference transcript is {len(text)} characters for "
                f"{seconds:.1f}s of audio — {cps:.1f} characters per second, "
                f"against a natural {12.19:.2f} and a {MAX_REFERENCE_CPS:.0f} "
                f"limit. It is describing about {len(text) / 12.19:.0f}s of "
                f"speech, so it is not a transcript of this clip. IndicF5 "
                f"cannot align them and the generation collapses."
            )

    def package_bundle(
        self,
        bundle_dir: Path,
    ) -> Path:
        if not bundle_dir.exists() or not bundle_dir.is_dir():
            raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")

        manifest_path = bundle_dir / "manifest.json"
        request_json_path = bundle_dir / "request" / "synthesis_request.json"
        reference_audio_path = bundle_dir / "request" / "reference.wav"

        for required_file in [manifest_path, request_json_path, reference_audio_path]:
            if not required_file.exists():
                raise FileNotFoundError(f"Missing required bundle file: {required_file}")

        zip_path = bundle_dir.with_name(bundle_dir.name + ".zip")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in bundle_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(bundle_dir)
                    zipf.write(file_path, arcname)

        return zip_path
