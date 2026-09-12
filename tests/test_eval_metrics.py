import json
import wave

import numpy as np

from src.eval.asr_metrics import evaluate_transcript
from src.eval.preprocess_metrics import evaluate_manifest
from src.eval.translation_metrics import evaluate_translation, script_ratio
from src.eval.tts_metrics import analyze_audio, evaluate_synthesis
from src.stages.asr.models import TranscriptResult, TranscriptSegment
from src.stages.translation.models import TranslatedSegment, TranslationResult
from src.stages.tts.models import (
    SynthesisRequest,
    SynthesisResult,
    SynthesisSegment,
    SynthesizedSegment,
)


SAMPLE_RATE = 24000


def write_wav(path, duration_s, trailing_silence_s=0.0, amplitude=0.3):
    """
    Write a tone of `duration_s` followed by `trailing_silence_s` of silence.
    """
    tone_samples = int(duration_s * SAMPLE_RATE)
    silence_samples = int(trailing_silence_s * SAMPLE_RATE)

    t = np.arange(tone_samples) / SAMPLE_RATE
    tone = amplitude * np.sin(2 * np.pi * 220.0 * t)
    signal = np.concatenate([tone, np.zeros(silence_samples)])

    pcm = (signal * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def make_translation(pairs, target_language="hi"):
    """
    pairs: list of (start_ts, end_ts, translated_text)
    """
    return TranslationResult(
        job_id="t",
        source_language="en",
        target_language=target_language,
        segments=[
            TranslatedSegment(
                segment_id=i,
                chunk_id=i,
                start_ts=start,
                end_ts=end,
                source_text="source",
                translated_text=text,
                source_language="en",
                target_language=target_language,
            )
            for i, (start, end, text) in enumerate(pairs)
        ],
    )


def test_pace_verdicts_span_the_range():
    # 13 chars in 10s is easy; 200 chars in 2s is impossible.
    translation = make_translation(
        [
            (0.0, 10.0, "क" * 13),
            (10.0, 12.0, "क" * 200),
        ]
    )

    metrics = evaluate_translation(translation, total_duration=12.0)

    assert metrics.segments[0].verdict == "fits"
    assert metrics.segments[1].verdict == "impossible"
    assert metrics.num_fits == 1
    assert metrics.num_impossible == 1
    assert metrics.segments[1].needs_attention


def test_budget_pools_the_following_gap():
    # A 2s slot followed by 3s of silence before the next segment starts.
    translation = make_translation(
        [
            (0.0, 2.0, "क" * 40),
            (5.0, 6.0, "क" * 5),
        ]
    )

    metrics = evaluate_translation(translation, total_duration=6.0)

    first = metrics.segments[0]

    assert first.slot_s == 2.0
    # Budget reaches toward the next start, minus the preserved pause.
    assert 4.5 < first.budget_s < 5.0
    assert first.required_cps < first.target_chars / first.slot_s


def test_wrong_script_is_detected():
    # Devanagari output for a Tamil target is the IndicTrans2 failure mode.
    assert script_ratio("नमस्कार", "hi") == 1.0
    assert script_ratio("नमस्कार", "ta") == 0.0

    translation = make_translation([(0.0, 5.0, "नमस्कार")], target_language="ta")
    metrics = evaluate_translation(translation)

    assert metrics.num_wrong_script == 1


def test_untranslated_and_empty_are_flagged():
    translation = make_translation([(0.0, 5.0, "source"), (5.0, 10.0, "")])
    metrics = evaluate_translation(translation, total_duration=10.0)

    assert metrics.num_copied == 1
    assert metrics.num_empty == 1


def test_digits_must_survive_translation():
    translation = make_translation([(0.0, 5.0, "मुझे 42 चाहिए")])
    metrics = evaluate_translation(translation)
    assert metrics.segments[0].digits_preserved is False  # source has no digits

    translation = make_translation([(0.0, 5.0, "कोई अंक नहीं")])
    metrics = evaluate_translation(translation)
    assert metrics.segments[0].digits_preserved is True


def test_analyze_audio_finds_the_trailing_tail():
    samples = np.concatenate(
        [
            0.4 * np.sin(2 * np.pi * 220 * np.arange(SAMPLE_RATE) / SAMPLE_RATE),
            np.zeros(SAMPLE_RATE),
        ]
    ).astype(np.float32)

    health = analyze_audio(samples, SAMPLE_RATE)

    assert 0.9 < health.trailing_silence_s < 1.1
    assert health.leading_silence_s < 0.05
    assert health.clipped_fraction == 0.0
    assert -12.0 < health.peak_dbfs < -6.0


def test_synthesis_metrics_flag_runaway_generation(tmp_path):
    """
    The real measurement from the Colab notebook: 16 characters of Hindi in a
    2s slot came back as 3.38s of audio. That must surface as both an overrun
    and a drawl, which together are the runaway-generation signature.
    """
    bundle = tmp_path / "bundle"
    (bundle / "output").mkdir(parents=True)

    write_wav(bundle / "output" / "seg_00000.wav", 2.4, trailing_silence_s=0.98)

    request = SynthesisRequest(
        job_id="t",
        language="hi",
        output_sample_rate=SAMPLE_RATE,
        segments=[
            SynthesisSegment(
                segment_id=0,
                chunk_id=0,
                start_ts=0.0,
                end_ts=2.0,
                text="सभी को नमस्कार ।",
                reference_audio="request/reference.wav",
            )
        ],
    )

    result = SynthesisResult(
        job_id="t",
        sample_rate=SAMPLE_RATE,
        model_id="xtts_v2",
        params={"do_sample": False},
        segments=[
            SynthesizedSegment(
                segment_id=0,
                chunk_id=0,
                audio_path="output/seg_00000.wav",
                duration=3.38,
                num_samples=81152,
                gpt_tokens=73,
                speaker_similarity=0.62,
            )
        ],
    )

    metrics = evaluate_synthesis(request, result, bundle_dir=bundle, total_duration=2.0)

    segment = metrics.segments[0]

    assert "overruns-slot" in segment.flags
    assert "beyond-stretch" in segment.flags
    assert "drawling" in segment.flags
    assert "long-tail" in segment.flags
    assert "voice-drift" in segment.flags

    assert metrics.num_overrun == 1
    assert metrics.num_fits == 0
    assert metrics.min_speaker_similarity == 0.62


def test_synthesis_metrics_accept_a_good_segment(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "output").mkdir(parents=True)

    write_wav(bundle / "output" / "seg_00000.wav", 1.8)

    request = SynthesisRequest(
        job_id="t",
        language="hi",
        output_sample_rate=SAMPLE_RATE,
        segments=[
            SynthesisSegment(
                segment_id=0,
                chunk_id=0,
                start_ts=0.0,
                end_ts=2.0,
                text="क" * 23,
                reference_audio="request/reference.wav",
            )
        ],
    )

    result = SynthesisResult(
        job_id="t",
        sample_rate=SAMPLE_RATE,
        segments=[
            SynthesizedSegment(
                segment_id=0,
                chunk_id=0,
                audio_path="output/seg_00000.wav",
                duration=1.8,
                speaker_similarity=0.92,
            )
        ],
    )

    metrics = evaluate_synthesis(request, result, bundle_dir=bundle, total_duration=2.0)

    assert metrics.segments[0].flags == []
    assert metrics.num_fits == 1


def test_missing_segment_is_reported_not_dropped(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "output").mkdir(parents=True)

    request = SynthesisRequest(
        job_id="t",
        language="hi",
        segments=[
            SynthesisSegment(
                segment_id=i,
                chunk_id=0,
                start_ts=float(i),
                end_ts=float(i) + 1.0,
                text="क" * 10,
                reference_audio="r.wav",
            )
            for i in range(3)
        ],
    )

    result = SynthesisResult(job_id="t", sample_rate=SAMPLE_RATE, segments=[])

    metrics = evaluate_synthesis(request, result, bundle_dir=bundle)

    assert metrics.num_missing == 3
    assert all("missing" in s.flags for s in metrics.segments)


def test_asr_metrics_flag_hallucination_signature():
    transcript = TranscriptResult(
        job_id="t",
        language="en",
        language_probability=0.98,
        segments=[
            TranscriptSegment(
                segment_id=0,
                chunk_id=0,
                start_ts=0.0,
                end_ts=2.0,
                text="hello there",
                avg_logprob=-0.3,
                no_speech_prob=0.05,
                compression_ratio=1.2,
            ),
            TranscriptSegment(
                segment_id=1,
                chunk_id=1,
                start_ts=2.0,
                end_ts=4.0,
                text="thank you thank you thank you",
                avg_logprob=-1.6,
                no_speech_prob=0.91,
                compression_ratio=3.1,
            ),
        ],
    )

    metrics = evaluate_transcript(transcript=transcript)

    assert metrics.wer is None  # no reference supplied
    assert metrics.suspect_segments == [1]
    assert metrics.num_high_no_speech == 1
    assert metrics.num_repetitive == 1
    assert metrics.num_low_confidence == 1
    assert metrics.source_cps > 0


def test_asr_metrics_with_reference_still_work():
    transcript = TranscriptResult(
        job_id="t",
        language="en",
        language_probability=0.99,
        segments=[
            TranscriptSegment(
                segment_id=0,
                chunk_id=0,
                start_ts=0.0,
                end_ts=1.0,
                text="hello world",
                avg_logprob=-0.2,
                no_speech_prob=0.01,
                compression_ratio=1.1,
            )
        ],
    )

    metrics = evaluate_transcript("hello world", transcript)

    assert metrics.wer == 0.0
    assert metrics.cer == 0.0


def test_preprocess_metrics_from_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {"chunk_path": "a.wav", "start_ts": 0.0, "end_ts": 2.0},
                {"chunk_path": "b.wav", "start_ts": 3.0, "end_ts": 4.0},
                {"chunk_path": "c.wav", "start_ts": 5.0, "end_ts": 5.2},
            ]
        )
    )

    metrics = evaluate_manifest(manifest, total_duration=10.0)

    assert metrics.num_segments == 3
    assert metrics.total_speech_s == 3.2
    assert metrics.speech_ratio == 0.32
    assert metrics.num_short_segments == 1  # the 0.2s one
    assert metrics.gaps == [1.0, 1.0]
    assert metrics.num_missing_chunks == 3


def test_a_bundle_manifest_is_not_mistaken_for_a_preprocess_manifest(tmp_path):
    # Pointing the harness at a bundle directory is normal on the GPU side,
    # where the preprocess manifest was never copied across. The bundle's own
    # manifest.json is an object, not a list of segments, and must be skipped
    # rather than parsed as one.
    from src.eval.harness import build_report

    bundle = tmp_path / "tts_bundle"
    (bundle / "request").mkdir(parents=True)

    with open(bundle / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {"bundle_version": "1.0", "job_id": "t"},
                "paths": {"request_json": "request/synthesis_request.json"},
            },
            f,
        )

    report = build_report(job_dir=bundle, bundle_dir=bundle)

    assert "preprocess" not in report["stages"]
