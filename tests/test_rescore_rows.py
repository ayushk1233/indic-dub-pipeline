"""
Rescoring must reconstruct the same rows synthesis wrote.

The first phase 0 run cost most of an hour of GPU and produced 28 sound clips
with void numbers: the ASR raised on every one (FINDINGS §13). Re-synthesizing
to fix a transcription bug would have re-made files that were never wrong, so
`rescore` re-reads the audio instead.

That is only safe if the reconstruction is exact. A row carries the label the
whole report groups by, the sentence id that selects the intended English, and
the slot every pace number divides by. Get the label wrong and the seed spread
is computed across mislabelled arms; get the id wrong and each clip is scored
against a different sentence. Both produce a complete, plausible, wrong report
— the same failure shape as the run this function exists to repair.

The directory name is the only place the arm and seed survive, and it is
lossy on purpose: `label.replace("/", "_")` turns `deva_hand/s0` into
`deva_hand_s0`, which contains two underscores and only one of them is a
separator.

No audio, no ASR, no GPU.
"""

import json

import pytest

probe = pytest.importorskip("colab.indicf5_xlit_probe")


@pytest.fixture
def clips(tmp_path, monkeypatch):
    """Lay out an OUT directory the way main() writes one."""
    soundfile = pytest.importorskip("soundfile")
    numpy = pytest.importorskip("numpy")

    def build(labels, indexes=(0, 5, 6), seconds=1.0):
        out = tmp_path / "indicf5_xlit_probe"
        for label in labels:
            directory = out / label.replace("/", "_")
            directory.mkdir(parents=True, exist_ok=True)
            for index in indexes:
                soundfile.write(
                    str(directory / f"{index:02d}.wav"),
                    numpy.zeros(int(seconds * probe.SAMPLE_RATE), dtype="float32"),
                    probe.SAMPLE_RATE, subtype="PCM_16")
        monkeypatch.setattr(probe, "OUT", out)
        monkeypatch.setattr(probe, "ROWS", out / "rows.json")
        return out

    return build


def test_the_arm_name_survives_its_own_underscore(clips):
    """
    `deva_hand_s0` splits as `deva_hand` + seed 0, not `deva` + `hand_s0`.
    Only a trailing _s<digits> is the separator.
    """
    clips(["deva_hand/s0", "deva_hand/s2"])
    rows = probe._rebuild_rows()

    assert {r["arm"] for r in rows} == {"deva_hand"}
    assert {r["seed"] for r in rows} == {0, 2}
    assert {r["label"] for r in rows} == {"deva_hand/s0", "deva_hand/s2"}


def test_a_single_seed_arm_keeps_its_bare_label(clips):
    """
    `latin` runs one seed, so main() writes the label without a suffix. Reading
    a seed out of it would put the control in its own phantom arm and leave the
    VERDICT unable to find the control at all.
    """
    clips(["latin"])
    rows = probe._rebuild_rows()

    assert {r["label"] for r in rows} == {"latin"}
    assert {r["arm"] for r in rows} == {"latin"}
    assert {r["seed"] for r in rows} == {0}


def test_each_clip_is_scored_against_its_own_sentence(clips):
    """The id comes from the filename and selects the intended English."""
    clips(["latin"], indexes=(0, 5))
    rows = {r["index"]: r for r in probe._rebuild_rows()}

    frozen = json.loads(probe.SENTENCES.read_text(encoding="utf-8"))["sentences"]
    english = {e["id"]: e["text"] for e in frozen}

    assert rows[0]["text"] == english[0]
    assert rows[5]["text"] == english[5]


def test_the_slot_comes_from_the_fixture_not_the_clip(clips):
    """
    `got/slot` is the pace number. The slot is what he actually took; the clip
    duration is what the model produced. Reading the slot off the clip would
    make every pace exactly 1.00 and hide the failure it exists to show.
    """
    clips(["latin"], indexes=(6,), seconds=9.0)
    row = probe._rebuild_rows()[0]

    slots = {s["id"]: s for s in
             json.loads(probe.SLOTS.read_text(encoding="utf-8"))["slots"]}

    assert row["slot_s"] == pytest.approx(slots[6]["duration_s"])
    assert row["actual_s"] == pytest.approx(9.0, abs=0.01)
    assert row["slot_s"] != pytest.approx(row["actual_s"])


def test_in_reference_is_carried_through(clips):
    """
    Sentences 0 and 1 sit inside the reference clip and the report flags them.
    Losing the flag on a rescore would silently promote them to evidence.
    """
    clips(["latin"], indexes=(0, 6))
    rows = {r["index"]: r for r in probe._rebuild_rows()}

    assert rows[0]["in_reference"] is True
    assert rows[6]["in_reference"] is False


def test_saved_rows_round_trip_to_the_same_paths(clips, tmp_path):
    """
    When main() did save rows, rescore uses them — and must rebuild each path
    from the label the same way main() wrote it.
    """
    out = clips(["deva_hand/s1"], indexes=(5,))
    probe.save_rows([{
        "arm": "deva_hand", "seed": 1, "label": "deva_hand/s1", "index": 5,
        "text": "Thirty-one fit perfectly.", "given": "थर्टी-वन फिट परफेक्टली।",
        "language": "en", "actual_s": 1.0, "slot_s": 1.86,
        "natural_s": 1.9, "requested_s": 1.86, "in_reference": False,
    }])

    saved = json.loads(probe.ROWS.read_text(encoding="utf-8"))
    path = out / saved[0]["label"].replace("/", "_") / f"{saved[0]['index']:02d}.wav"

    assert path.exists()


def test_rebuilding_ignores_a_directory_with_no_clips(clips):
    """A zip unpacked half way should not invent rows with no audio."""
    out = clips(["latin"])
    (out / "deva_hand_s0").mkdir()

    assert {r["arm"] for r in probe._rebuild_rows()} == {"latin"}
