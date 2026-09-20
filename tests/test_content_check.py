"""
The QC report can say what the audio actually says.

`evaluate_intelligibility` existed and was never called from `build_report`.
That gap is why a run with nine gibberish segments passed every check it had:
the corrupted clips were the right length, landed on budget, and assembled at
tempo 1.00, so every timing metric read green.

These tests use a stub backend, so they need neither a GPU nor a 3 GB model.
"""

import json
import wave

from src.eval.harness import INTELLIGIBILITY_CER_LIMIT, build_report, render_report


class Heard:
    """One transcribed line, shaped like faster-whisper's segment."""

    def __init__(self, text):
        self.text = text


class StubASR:
    """
    Returns whatever was registered for a wav path, by segment number.

    Deliberately not a Mock: the interface `evaluate_intelligibility` needs is
    one method returning a (segments, info) pair, and spelling it out keeps the
    test honest about what a real backend has to provide.
    """

    def __init__(self, by_segment):
        self.by_segment = by_segment

    def transcribe_audio(self, audio_path):
        index = int(str(audio_path).rsplit("seg_", 1)[1].split(".")[0])

        return [Heard(self.by_segment[index])], None


def write_wav(path, seconds=1.0, rate=24000):
    path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))

    return path


def build_job(tmp_path, asked):
    """A job directory holding a bundle that has come back from synthesis."""
    job_dir = tmp_path / "job"
    bundle = job_dir / "tts_bundle"

    write_wav(bundle / "request" / "reference.wav")

    request = {
        "job_id": "content",
        "language": "hi",
        "output_sample_rate": 24000,
        "reference_text": "सो लेट मी टेल यू",
        "segments": [
            {
                "segment_id": index,
                "chunk_id": 0,
                "start_ts": float(index),
                "end_ts": float(index) + 1.0,
                "text": text,
                "reference_audio": "request/reference.wav",
                "budget_s": 1.0,
            }
            for index, text in enumerate(asked)
        ],
    }

    result = {
        "job_id": "content",
        "model_id": "ai4bharat/IndicF5",
        "sample_rate": 24000,
        "segments": [
            {
                "segment_id": index,
                "chunk_id": 0,
                "audio_path": f"output/seg_{index:05d}.wav",
                "duration": 1.0,
                "num_samples": 24000,
                "status": "done",
                "error": None,
            }
            for index in range(len(asked))
        ],
    }

    for index in range(len(asked)):
        write_wav(bundle / "output" / f"seg_{index:05d}.wav")

    (bundle / "request" / "synthesis_request.json").write_text(
        json.dumps(request, ensure_ascii=False), encoding="utf-8"
    )
    (bundle / "output" / "synthesis_result.json").write_text(
        json.dumps(result, ensure_ascii=False), encoding="utf-8"
    )

    return job_dir, bundle


ASKED = ["सात असंभव थे", "लगभग बीस प्रतिशत अधिक समय"]


def test_without_a_backend_the_report_has_no_content_section(tmp_path):
    """
    The check costs an ASR model, so it stays opt-in. This pins the default,
    which is also the behaviour that let the bad run through — recorded here
    deliberately rather than left implicit.
    """
    job_dir, bundle = build_job(tmp_path, ASKED)

    report = build_report(job_dir, bundle_dir=bundle)

    assert "content" not in report["stages"]


def test_clean_audio_scores_within_the_limit(tmp_path):
    job_dir, bundle = build_job(tmp_path, ASKED)

    report = build_report(
        job_dir, bundle_dir=bundle, asr_backend=StubASR(dict(enumerate(ASKED)))
    )
    content = report["stages"]["content"]

    assert content["num_scored"] == 2
    assert content["num_bad"] == 0
    assert content["max_cer"] < INTELLIGIBILITY_CER_LIMIT


def test_the_real_corruption_is_caught(tmp_path):
    """
    The second row is what segment 6 actually produced: `20%` spoken as लतक.
    This is the case the whole check exists for.
    """
    job_dir, bundle = build_job(tmp_path, ASKED)

    heard = {0: "सात असंभव थे", 1: "लगभग लतक अधिक समय"}
    report = build_report(job_dir, bundle_dir=bundle, asr_backend=StubASR(heard))
    content = report["stages"]["content"]

    assert content["num_bad"] == 1
    assert content["bad_segments"] == [1]


def test_a_head_prefix_is_caught_too(tmp_path):
    """
    Digit segments also opened with invented speech — `यू`, `ते`, `एंड`. A
    short segment with a prefix is exactly what a duration-only report calls
    healthy.
    """
    job_dir, bundle = build_job(tmp_path, ["सात असंभव थे"])

    report = build_report(
        job_dir, bundle_dir=bundle, asr_backend=StubASR({0: "यह ऐड असंभव थे"})
    )

    assert report["stages"]["content"]["num_bad"] == 1


def test_the_rendered_report_names_the_bad_segments(tmp_path):
    """A number buried in JSON is not a warning anybody acts on."""
    job_dir, bundle = build_job(tmp_path, ASKED)

    heard = {0: "सात असंभव थे", 1: "लगभग लतक अधिक समय"}
    report = build_report(job_dir, bundle_dir=bundle, asr_backend=StubASR(heard))

    text = render_report(report)

    assert "CONTENT" in text
    assert "OVER THE LIMIT" in text
    assert "1" in text


def test_a_clean_run_says_so_rather_than_staying_silent(tmp_path):
    job_dir, bundle = build_job(tmp_path, ASKED)

    report = build_report(
        job_dir, bundle_dir=bundle, asr_backend=StubASR(dict(enumerate(ASKED)))
    )

    assert "every segment within the limit" in render_report(report)
