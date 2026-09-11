"""
End-to-end test of the spine, from a translation to a dubbed video file.

Synthesis is stubbed with tones of known length, which is the point: it makes
the timing assertions exact. The stages either side of it are real — the same
exporter, importer, assembly cascade and remux the pipeline uses — so this is
the test that would have caught "nothing runs end to end".

ASR and machine translation are deliberately not in the loop. Both need
multi-gigabyte models and neither affects whether the audio lands in the right
place, which is what this file is about.
"""

import json
import wave

import numpy as np
import pytest

from src.pipeline.paths import JobPaths
from src.pipeline.runner import PipelineRunner
from src.stages.assemble import write_wav_samples
from src.stages.remux import probe_duration, probe_stream_durations
from src.stages.translation.models import TranslatedSegment, TranslationResult
from src.stages.tts.models import SynthesisRequest, SynthesisResult, SynthesizedSegment


TEST_VIDEO = "test.mp4"
SAMPLE_RATE = 24000


@pytest.fixture(scope="module")
def video_duration():
    return probe_duration(TEST_VIDEO)


def tone(duration_s, amplitude=0.3):
    n = max(int(duration_s * SAMPLE_RATE), 1)
    t = np.arange(n) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def write_translation(paths, segments, target_language="hi"):
    """
    segments: list of (segment_id, start_ts, end_ts, text)
    """
    translation = TranslationResult(
        job_id=paths.job_id,
        source_language="en",
        target_language=target_language,
        segments=[
            TranslatedSegment(
                segment_id=segment_id,
                chunk_id=segment_id,
                start_ts=start,
                end_ts=end,
                source_text="source text",
                translated_text=text,
                source_language="en",
                target_language=target_language,
            )
            for segment_id, start, end, text in segments
        ],
    )

    paths.job_dir.mkdir(parents=True, exist_ok=True)

    with open(paths.translation, "w", encoding="utf-8") as f:
        json.dump(translation.model_dump(), f, ensure_ascii=False, indent=2)

    return translation


def fake_synthesis(paths, durations):
    """
    Stand in for the GPU host: write one tone per segment plus the result file
    the worker would have produced.

    `durations` maps segment_id to the audio length the model 'generated'.
    """
    bundle = paths.bundle
    output = bundle / "output"
    output.mkdir(parents=True, exist_ok=True)

    with open(bundle / "request" / "synthesis_request.json", encoding="utf-8") as f:
        request = SynthesisRequest(**json.load(f))

    produced = []

    for segment in request.segments:
        duration = durations.get(segment.segment_id)

        if duration is None:
            continue

        name = f"seg_{segment.segment_id:05d}.wav"
        samples = tone(duration)
        write_wav_samples(output / name, samples, SAMPLE_RATE)

        produced.append(
            SynthesizedSegment(
                segment_id=segment.segment_id,
                chunk_id=segment.chunk_id,
                audio_path=f"output/{name}",
                duration=samples.size / SAMPLE_RATE,
                num_samples=samples.size,
            )
        )

    result = SynthesisResult(
        job_id=request.job_id,
        sample_rate=SAMPLE_RATE,
        model_id="stub",
        params={"stub": True},
        segments=produced,
    )

    with open(output / "synthesis_result.json", "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)

    return result


def build_runner(tmp_path, job_id="e2e"):
    paths = JobPaths(job_id, root=tmp_path / "artifacts").ensure()
    cfg = {"audio": {"sample_rate": 16000, "channels": 1}}
    return PipelineRunner(cfg=cfg, paths=paths), paths


def test_translation_to_dubbed_video(tmp_path, video_duration):
    runner, paths = build_runner(tmp_path)

    write_translation(
        paths,
        [
            (0, 0.0, 2.0, "क" * 16),
            (1, 3.0, 7.0, "क" * 60),
            (2, 9.0, 12.0, "क" * 40),
        ],
    )

    reference = tmp_path / "reference.wav"
    write_wav_samples(reference, tone(10.0), SAMPLE_RATE)

    export = runner.export_bundle(reference)
    assert export.status.value == "done"
    assert paths.bundle_zip.exists()

    # Segment 1 deliberately overruns its 4s slot; the cascade must handle it.
    fake_synthesis(paths, {0: 1.5, 1: 5.0, 2: 2.5})

    imported = runner.import_bundle()
    assert imported.complete
    assert imported.issues == []

    assemble = runner.assemble(imported, video_duration)
    assert assemble.status.value == "done"
    assert paths.dubbed_audio.exists()

    remux = runner.remux(TEST_VIDEO)
    assert remux.status.value == "done", remux.error
    assert paths.dubbed_video.exists()

    # The whole point: audio and video agree on how long the file is.
    streams = probe_stream_durations(paths.dubbed_video)
    assert abs(streams["video"] - video_duration) < 0.1
    assert abs(streams["audio"] - video_duration) < 0.1


def test_the_timeline_records_which_tier_fixed_each_segment(tmp_path, video_duration):
    runner, paths = build_runner(tmp_path, "timeline")

    # Every segment is followed immediately by another, so no segment can
    # borrow time from a trailing pause. The last one is only there to close
    # segment 2's budget: the final segment's budget always runs to the end of
    # the video, which would let anything fit.
    write_translation(
        paths,
        [
            (0, 0.0, 5.0, "क" * 20),   # fits easily
            (1, 5.0, 7.0, "क" * 20),   # needs stretching
            (2, 7.0, 9.0, "क" * 20),   # hopeless, must drift
            (3, 9.0, 11.0, "क" * 20),  # closes segment 2's budget
        ],
    )

    reference = tmp_path / "reference.wav"
    write_wav_samples(reference, tone(10.0), SAMPLE_RATE)
    runner.export_bundle(reference)

    fake_synthesis(paths, {0: 2.0, 1: 2.3, 2: 6.0, 3: 1.0})

    runner.assemble(runner.import_bundle(), video_duration)

    with open(paths.timeline, encoding="utf-8") as f:
        timeline = json.load(f)

    tiers = {s["segment_id"]: s["tier"] for s in timeline["segments"]}

    assert tiers[0] in {"placed", "trimmed"}
    assert tiers[1] == "stretched"
    assert tiers[2] == "drifted"

    assert timeline["max_tempo_used"] <= 1.25


def test_a_partial_synthesis_still_produces_a_video(tmp_path, video_duration):
    runner, paths = build_runner(tmp_path, "partial")

    write_translation(
        paths,
        [
            (0, 0.0, 2.0, "क" * 16),
            (1, 3.0, 6.0, "क" * 40),
        ],
    )

    reference = tmp_path / "reference.wav"
    write_wav_samples(reference, tone(10.0), SAMPLE_RATE)
    runner.export_bundle(reference)

    # The GPU host only finished the first segment.
    fake_synthesis(paths, {0: 1.5})

    imported = runner.import_bundle()
    assert not imported.complete
    assert imported.missing_ids == [1]

    assemble = runner.assemble(imported, video_duration)
    assert assemble.metrics["num_missing"] == 1

    remux = runner.remux(TEST_VIDEO)
    assert remux.status.value == "done"
    assert paths.dubbed_video.exists()


def test_the_dubbed_track_is_silent_where_a_segment_is_missing(tmp_path, video_duration):
    runner, paths = build_runner(tmp_path, "gaps")

    write_translation(paths, [(0, 0.0, 2.0, "क" * 16), (1, 10.0, 12.0, "क" * 16)])

    reference = tmp_path / "reference.wav"
    write_wav_samples(reference, tone(10.0), SAMPLE_RATE)
    runner.export_bundle(reference)
    fake_synthesis(paths, {0: 1.5, 1: 1.5})

    runner.assemble(runner.import_bundle(), video_duration)

    with wave.open(str(paths.dubbed_audio), "rb") as handle:
        rate = handle.getframerate()
        track = np.frombuffer(
            handle.readframes(handle.getnframes()), dtype=np.int16
        ).astype(np.float32) / 32767.0

    # Audio at both placements, silence in the long gap between them.
    assert np.abs(track[int(0.2 * rate) : int(1.2 * rate)]).max() > 0.01
    assert np.abs(track[int(4.0 * rate) : int(9.0 * rate)]).max() < 0.01
    assert np.abs(track[int(10.2 * rate) : int(11.2 * rate)]).max() > 0.01


def test_a_reference_is_chosen_from_the_speakers_own_chunks(tmp_path):
    runner, paths = build_runner(tmp_path, "reference")

    paths.chunks.mkdir(parents=True, exist_ok=True)
    write_wav_samples(paths.chunks / "chunk_0000.wav", tone(1.0), SAMPLE_RATE)
    write_wav_samples(paths.chunks / "chunk_0001.wav", tone(9.0), SAMPLE_RATE)
    write_wav_samples(paths.chunks / "chunk_0002.wav", tone(3.0), SAMPLE_RATE)

    # The longest chunk wins, because greedy decoding degrades on short
    # reference audio per the synthesis research.
    assert runner._pick_reference().name == "chunk_0001.wav"


def test_missing_chunks_and_no_reference_is_an_explicit_error(tmp_path):
    runner, _ = build_runner(tmp_path, "noref")

    with pytest.raises(FileNotFoundError, match="No reference audio"):
        runner._pick_reference()
