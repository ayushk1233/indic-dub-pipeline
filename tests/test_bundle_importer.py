"""
Tests for reading a synthesis bundle back from a GPU host.

The importer's job is to be suspicious: the audio was written elsewhere, the
run may have been cut short, and the paths inside the result are attacker-
shaped even when you wrote them yourself.
"""

import json
import wave
import zipfile

import numpy as np
import pytest

from src.stages.tts.bundle.exporter import BundleExporter
from src.stages.tts.bundle.importer import BundleImporter
from src.stages.tts.models import (
    SynthesisRequest,
    SynthesisResult,
    SynthesisSegment,
    SynthesizedSegment,
)


SAMPLE_RATE = 24000


def write_wav(path, duration_s):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(duration_s * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    pcm = (0.3 * np.sin(2 * np.pi * 220.0 * t) * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def make_request(num_segments=2):
    return SynthesisRequest(
        job_id="t",
        language="hi",
        output_sample_rate=SAMPLE_RATE,
        segments=[
            SynthesisSegment(
                segment_id=i,
                chunk_id=0,
                start_ts=float(i),
                end_ts=float(i) + 1.0,
                text="क" * 12,
                reference_audio="request/reference.wav",
            )
            for i in range(num_segments)
        ],
    )


def build_bundle(tmp_path, result, num_segments=2, audio_for=(0, 1), durations=None):
    """
    Write a bundle on disk the way the exporter plus a GPU run would leave it.
    """
    bundle = tmp_path / "tts_bundle"

    reference = tmp_path / "reference.wav"
    write_wav(reference, 1.0)

    BundleExporter().export(
        make_request(num_segments),
        reference,
        bundle,
    )

    for segment_id in audio_for:
        duration = (durations or {}).get(segment_id, 1.0)
        write_wav(bundle / "output" / f"seg_{segment_id:05d}.wav", duration)

    with open(bundle / "output" / "synthesis_result.json", "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)

    return bundle


def done(segment_id, duration=1.0, path=None):
    return SynthesizedSegment(
        segment_id=segment_id,
        chunk_id=0,
        audio_path=path or f"output/seg_{segment_id:05d}.wav",
        duration=duration,
        num_samples=int(duration * SAMPLE_RATE),
    )


def test_a_complete_bundle_imports_cleanly(tmp_path):
    result = SynthesisResult(
        job_id="t",
        sample_rate=SAMPLE_RATE,
        model_id="ai4bharat/IndicF5",
        segments=[done(0), done(1)],
    )

    imported = BundleImporter().import_bundle(build_bundle(tmp_path, result))

    assert imported.complete
    assert imported.issues == []
    assert imported.usable_ids == {0, 1}
    assert imported.job_id == "t"

    # Paths come back absolute and pointing inside the bundle.
    for path in imported.audio_paths.values():
        assert path.is_absolute()
        assert path.exists()


def test_a_partial_run_is_reported_not_raised(tmp_path):
    # The worker got through segment 0 and then Colab timed out.
    result = SynthesisResult(
        job_id="t",
        sample_rate=SAMPLE_RATE,
        segments=[done(0)],
    )

    imported = BundleImporter().import_bundle(
        build_bundle(tmp_path, result, audio_for=(0,))
    )

    assert not imported.complete
    assert imported.missing_ids == [1]
    assert imported.usable_ids == {0}
    assert [i.kind for i in imported.issues] == ["missing"]


def test_a_failed_segment_is_kept_out_of_the_usable_set(tmp_path):
    result = SynthesisResult(
        job_id="t",
        sample_rate=SAMPLE_RATE,
        segments=[
            done(0),
            SynthesizedSegment(
                segment_id=1,
                chunk_id=0,
                audio_path="",
                duration=0.0,
                status="failed",
                error="CUDA out of memory",
            ),
        ],
    )

    imported = BundleImporter().import_bundle(
        build_bundle(tmp_path, result, audio_for=(0,))
    )

    assert imported.usable_ids == {0}
    failures = imported.issues_of("failed")
    assert len(failures) == 1
    assert "out of memory" in failures[0].detail


def test_duration_is_re_derived_from_the_audio(tmp_path):
    # The result claims 3.38s; the file on disk is 1.0s.
    result = SynthesisResult(
        job_id="t",
        sample_rate=SAMPLE_RATE,
        segments=[done(0, duration=3.38), done(1)],
    )

    imported = BundleImporter().import_bundle(build_bundle(tmp_path, result))

    mismatches = imported.issues_of("duration-mismatch")
    assert len(mismatches) == 1
    assert mismatches[0].segment_id == 0
    # A disagreement is a finding, not a disqualification.
    assert 0 in imported.usable_ids


def test_a_path_escaping_the_bundle_is_refused(tmp_path):
    secret = tmp_path / "outside.wav"
    write_wav(secret, 1.0)

    result = SynthesisResult(
        job_id="t",
        sample_rate=SAMPLE_RATE,
        segments=[done(0, path="../outside.wav"), done(1)],
    )

    imported = BundleImporter().import_bundle(
        build_bundle(tmp_path, result, audio_for=(1,))
    )

    escapes = imported.issues_of("path-escape")
    assert len(escapes) == 1
    assert escapes[0].segment_id == 0
    assert 0 not in imported.usable_ids


def test_audio_named_but_absent_is_reported(tmp_path):
    result = SynthesisResult(
        job_id="t",
        sample_rate=SAMPLE_RATE,
        segments=[done(0), done(1)],
    )

    bundle = build_bundle(tmp_path, result, audio_for=(0,))
    imported = BundleImporter().import_bundle(bundle)

    assert [i.kind for i in imported.issues] == ["audio-missing"]
    assert imported.usable_ids == {0}


def test_an_unsupported_bundle_version_is_refused(tmp_path):
    result = SynthesisResult(job_id="t", sample_rate=SAMPLE_RATE, segments=[done(0)])
    bundle = build_bundle(tmp_path, result, audio_for=(0,))

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"]["bundle_version"] = "99.0"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="Unsupported bundle version"):
        BundleImporter().import_bundle(bundle)


def test_a_bundle_with_no_result_says_so(tmp_path):
    bundle = tmp_path / "tts_bundle"
    reference = tmp_path / "reference.wav"
    write_wav(reference, 1.0)
    BundleExporter().export(make_request(), reference, bundle)

    with pytest.raises(FileNotFoundError, match="No synthesis result"):
        BundleImporter().import_bundle(bundle)


def test_a_zip_bundle_is_unpacked_and_imported(tmp_path):
    result = SynthesisResult(
        job_id="t",
        sample_rate=SAMPLE_RATE,
        segments=[done(0), done(1)],
    )

    bundle = build_bundle(tmp_path, result)
    archive = BundleExporter().package_bundle(bundle)

    imported = BundleImporter().import_bundle(
        archive,
        extract_to=tmp_path / "unpacked",
    )

    assert imported.complete
    assert imported.bundle_dir == tmp_path / "unpacked"


def test_a_zip_member_escaping_the_root_is_refused(tmp_path):
    archive = tmp_path / "evil.zip"

    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../escaped.txt", "nope")

    with pytest.raises(ValueError, match="escapes the bundle root"):
        BundleImporter().extract(archive, tmp_path / "unpacked")


def test_a_result_segment_nobody_asked_for_is_flagged(tmp_path):
    result = SynthesisResult(
        job_id="t",
        sample_rate=SAMPLE_RATE,
        segments=[done(0), done(1), done(7)],
    )

    bundle = build_bundle(tmp_path, result)
    write_wav(bundle / "output" / "seg_00007.wav", 1.0)

    imported = BundleImporter().import_bundle(bundle)

    unexpected = imported.issues_of("unexpected")
    assert [i.segment_id for i in unexpected] == [7]
    assert 7 not in imported.usable_ids


def test_re_exporting_clears_a_previous_runs_output(tmp_path):
    # A bundle describes one synthesis attempt. Re-exporting over a bundle
    # that already holds results must not leave them behind: `package_bundle`
    # would ship them to the GPU host, and any segment that fails there would
    # be silently satisfied by the stale file.
    bundle = tmp_path / "tts_bundle"
    reference = tmp_path / "reference.wav"
    write_wav(reference, 1.0)

    BundleExporter().export(make_request(), reference, bundle)

    stale_audio = bundle / "output" / "seg_00000.wav"
    stale_result = bundle / "output" / "synthesis_result.json"
    stale_log = bundle / "logs" / "worker.log"
    write_wav(stale_audio, 1.0)
    stale_result.write_text("{}", encoding="utf-8")
    stale_log.write_text("old run", encoding="utf-8")

    BundleExporter().export(make_request(), reference, bundle)

    assert not stale_audio.exists()
    assert not stale_result.exists()
    assert not stale_log.exists()

    # The directories themselves survive, since the worker writes into them.
    assert (bundle / "output").is_dir()
    assert (bundle / "logs").is_dir()

    archive = BundleExporter().package_bundle(bundle)

    with zipfile.ZipFile(archive) as zf:
        assert not [name for name in zf.namelist() if name.startswith("output/")]
