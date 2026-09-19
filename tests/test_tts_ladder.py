"""
The duration ladder must vary duration and nothing else.

Two ways this probe can produce a plausible, wrong answer, and neither raises:

  - **The ladder varies pace instead of duration.** If a long sentence is also
    asked at a different characters-per-second than a short one, every row
    differs in two variables and the report reads as a duration effect. The
    sentences were written naturally and their targets derived from one rate,
    so asked-cps is constant by construction — which is a property worth
    pinning, because editing one sentence's text without its target silently
    breaks it.

  - **The run is not comparable to the run it cites.** This script has no
    content floor — nothing in it was ever read aloud — so every number is read
    against the control block and against FINDINGS §15. If the control drifts,
    or if the constants here stop matching what §15 recorded, the comparison is
    to a number that no longer exists.

No audio, no model, no GPU.
"""

import json
import unicodedata
from pathlib import Path

import pytest

from src.eval.translation_metrics import script_ratio

LADDER = Path("fixtures/sentences/tts_ladder_hi.json")
SLOTS = Path("fixtures/hi_speaker/slots.json")


@pytest.fixture
def ladder():
    return json.loads(LADDER.read_text(encoding="utf-8"))


def test_asked_cps_is_constant_across_the_ladder(ladder):
    """
    The whole design. A row that is longer AND faster measures two things and
    reports one.
    """
    rates = [s["chars"] / s["target_s"] for s in ladder["sentences"]]
    assert max(rates) - min(rates) < 0.15, [round(r, 2) for r in rates]


def test_the_rate_is_his_own_and_not_a_corpus_figure(ladder):
    """
    Targets come from the speaker's measured in-slot rate, not from FLEURS.
    FINDINGS §10 records that the FLEURS-fitted model is about 20% off
    synthesis, so a ladder built on it would ask for durations nobody uses.
    """
    measured = json.loads(SLOTS.read_text(encoding="utf-8"))["measured_cps"]
    assert ladder["rate_cps"] == measured
    assert ladder["rate_source"].endswith("#measured_cps")


def test_the_ladder_actually_spans_a_range(ladder):
    targets = [s["target_s"] for s in ladder["sentences"]]
    assert targets == sorted(targets), "rows are not in increasing duration"
    assert targets[0] < 2.0
    assert targets[-1] > 20.0
    assert len(targets) >= 10


def test_every_bucket_has_rows():
    """A bucket with nothing in it is a row of dashes that reads as a result."""
    probe = pytest.importorskip("colab.indicf5_tts_probe")

    _, sentences = probe.load_ladder()
    covered = {probe.bucket_of(s["target_s"]) for s in sentences}
    assert covered == {name for _, _, name in probe.BUCKETS}


def test_the_script_is_devanagari_and_nfc(ladder):
    for row in ladder["sentences"]:
        assert script_ratio(row["text"], "hi") == 1.0
        assert unicodedata.normalize("NFC", row["text"]) == row["text"]
        assert row["chars"] == len(row["text"])
        assert row["bytes"] == len(row["text"].encode("utf-8"))


def test_the_file_says_it_has_no_floor(ladder):
    """
    The one thing a reader must not assume. §15's numbers are anchored to his
    own recordings; nothing in this file was ever read aloud.
    """
    assert ladder["floor"] is None
    assert "NO CONTENT FLOOR" in ladder["_note"]
    assert ladder["control_block"].endswith("fixture7_hi.json")


def test_the_control_block_is_the_one_with_a_floor():
    probe = pytest.importorskip("colab.indicf5_tts_probe")

    control = probe.control_sentences()
    assert len(control) == 7
    slots = {s["id"]: s["duration_s"] for s in
             json.loads(SLOTS.read_text(encoding="utf-8"))["slots"]}
    for row in control:
        assert row["target_s"] == slots[row["id"]], "control lost his own slot"


def test_the_control_constants_match_what_findings_recorded():
    """
    These are quoted in the report as the thing this session is compared to.
    If §15 is ever re-measured and these are not updated, the probe compares a
    live number to a dead one and says they agree.
    """
    probe = pytest.importorskip("colab.indicf5_tts_probe")

    findings = Path("FINDINGS.md").read_text(encoding="utf-8")
    body = findings[findings.index("## 15."):findings.index("## 16.")]
    assert f"{probe.CONTROL_ARM_CER:.3f}" in body
    assert f"{probe.CONTROL_FLOOR_CER:.3f}" in body
    assert f"{probe.CONTROL_SEED_SPREAD:.3f}" in body


def test_both_arms_generate_hindi_and_differ_only_in_the_reference():
    """
    en_ref is the shipping en -> hi configuration. If it ever started
    generating English text the arm would stop being the shipping one and the
    comparison would be to something nobody ships.
    """
    probe = pytest.importorskip("colab.indicf5_tts_probe")

    assert set(probe.DEFAULT_ARMS) == {"hi_ref", "en_ref"}
    assert probe.ARMS["hi_ref"] == ("hindi_reference_short.wav", "hindi_short")
    assert probe.ARMS["en_ref"] == ("english_reference_short.wav", "english_short")
    # the generated text comes from the ladder, which is Hindi, for both arms
    _, sentences = probe.load_ladder()
    assert all(script_ratio(s["text"], "hi") == 1.0 for s in sentences)


def test_the_long_rows_are_marked_as_a_chunk_test_too():
    """
    fix_duration forces one chunk, so the top of the ladder is a chunk-length
    measurement as well as a duration one. A reader who takes a bad 21s row as
    evidence about duration alone has been misled by the report.
    """
    probe = pytest.importorskip("colab.indicf5_tts_probe")

    _, sentences = probe.load_ladder()
    past = [s for s in sentences if s["target_s"] > probe.LONG_CHUNK_S]
    assert past, "nothing on the ladder tests the chunk limit"
    assert probe.LONG_CHUNK_S >= 15.0


def test_a_failed_synthesis_still_produces_a_row():
    """
    The top of the ladder may not synthesize at all. A row that vanishes on
    exception leaves a report that looks complete and is missing exactly its
    most interesting case.
    """
    probe = pytest.importorskip("colab.indicf5_tts_probe")

    source = Path("colab/indicf5_tts_probe.py").read_text(encoding="utf-8")
    failure = source[source.index("except Exception as exc:"):]
    assert "rows.append" in failure[:900], "an exception drops the row entirely"
    assert "synthesis_error" in failure[:900]
    # and it has to survive being written out, or a container replacement
    # turns the run's most interesting row into a clip that merely went missing
    assert "synthesis_error" in probe.SAVED
