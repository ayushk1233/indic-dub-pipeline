"""
The `deva_ref` arm must differ from `deva_hand` in exactly one thing.

The arm exists to separate two explanations for the accent overshoot the
speaker heard (FINDINGS §4b): the Devanagari spelling of the generated text, or
being handed a reference whose audio is English and whose transcript is Latin
while the generated text is Devanagari. The two have opposite consequences —
one is a dial that applies to everybody, the other transfers in context and
adapts per speaker for free — so the experiment is only worth running if it is
clean.

Clean means: same reference audio, same generated sentences, same seeds, and a
reference transcript that is a word-for-word transliteration of the Latin one
rather than a fresh paraphrase. Every test here pins one of those. A second
variable would not raise; it would produce a difference, and the difference
would be attributed to the reference pair.

No audio, no model, no GPU.
"""

import json
import unicodedata

import pytest

probe = pytest.importorskip("colab.indicf5_xlit_probe")

from colab.indicf5_english import devanagari_fraction, plain  # noqa: E402


@pytest.fixture
def fixture():
    return json.loads(probe.REFERENCE_DEVA.read_text(encoding="utf-8"))


@pytest.fixture
def latin_reference():
    """Exactly what main() passes as ref_text for the other arms."""
    transcripts = json.loads(
        (probe.FIXTURES / "reference_text.json").read_text(encoding="utf-8"))
    return plain(transcripts[probe.REFERENCE_KEY]["text"])


def test_the_latin_side_is_the_transcript_the_other_arms_are_given(
        fixture, latin_reference):
    """
    If it drifts from `plain(english_short)`, the two arms stop sharing a
    reference and the comparison silently acquires a second variable — the
    words themselves, not their script.
    """
    assert fixture["latin"] == latin_reference


def test_the_devanagari_is_word_for_word_against_the_latin(fixture):
    """
    A paraphrase would change how much text the model has to align to the same
    ten seconds of audio, which is a different experiment. The stored alignment
    is the reviewable artifact; this asserts it actually reconstructs both
    sides rather than being decoration.
    """
    latin_words = [pair[0] for pair in fixture["alignment"]]
    deva_words = [pair[1] for pair in fixture["alignment"]]

    assert " ".join(latin_words) == fixture["latin"]
    assert " ".join(deva_words) == fixture["devanagari"]
    assert len(latin_words) == fixture["words"]


def test_plain_is_a_no_op_on_the_devanagari(fixture):
    """
    main() runs both reference transcripts through `plain()` — FINDINGS §5c,
    the policy that stopped a clip opening with invented speech. On Latin it
    lowercases and strips punctuation; the fixture claims it does nothing here.
    Asserted rather than assumed, because if `plain` did strip something the
    two arms would be conditioned on texts of different lengths.
    """
    assert plain(fixture["devanagari"]) == fixture["devanagari"]


def test_the_reference_passes_the_same_text_gate_the_arms_do(fixture):
    """The reference is conditioning text; a Latin word left in it is §4."""
    assert probe.check_text(fixture["devanagari"]) == []
    assert devanagari_fraction(fixture["devanagari"]) == 1.0


def test_the_text_is_nfc_and_stores_nukta_decomposed(fixture):
    """
    U+0958..U+095F are Unicode composition exclusions, so NFC pulls ज़ apart
    into ज + U+093C rather than building it. A file that round-trips through
    an editor doing NFD, or one hand-typed with the precomposed form, is a
    different byte sequence and therefore a different token sequence.
    """
    text = fixture["devanagari"]
    assert unicodedata.normalize("NFC", text) == text
    assert fixture["precomposed_nukta"] == []
    assert [c for c in text if "क़" <= c <= "य़"] == []
    assert fixture["bytes"] == len(text.encode("utf-8"))
    assert fixture["chars"] == len(text)


def test_the_new_words_are_the_only_ones_not_already_reviewed(fixture):
    """
    The review burden is the claim. Every other word is lifted from
    deva_hand.json, which the speaker has already read and corrected, so
    `new_words` is what he is actually being asked to check. If a word drifts
    out of deva_hand this stops being true and the file needs re-reviewing.
    """
    hand = json.loads((probe.XLIT / "deva_hand.json").read_text(encoding="utf-8"))
    reviewed = set()
    for sentence in hand["sentences"]:
        for word in sentence["devanagari"].replace("—", " ").split():
            reviewed.add(word.strip("।,"))

    new = [w for w in dict.fromkeys(fixture["devanagari"].split())
           if w not in reviewed]
    assert new == fixture["new_words"]


def test_an_unreviewed_reference_is_refused(fixture):
    """
    Same rule as deva_hand, and for a stronger reason: a drafting error in the
    generated text spoils one sentence, while one in the reference conditions
    every clip in the arm.
    """
    if fixture.get("reviewed"):
        pytest.skip("reviewed; the refusal path is covered by the branch below")

    data, refusal = probe.load_reference_deva()
    assert data is None
    assert "not reviewed" in refusal
    assert str(fixture["new_words"]) in refusal


def test_deva_ref_generates_deva_hands_sentences(fixture):
    """
    The arm varies ref_text. If it also had its own generated text there would
    be nothing to attribute a difference to.
    """
    assert probe.ARM_TEXT["deva_ref"] == probe.ARM_TEXT["deva_hand"] == "deva_hand"


def test_rebuilding_rows_ignores_the_reference_file(tmp_path, monkeypatch):
    """
    reference_deva.json sits in fixtures/xlit/ beside the arm files and has a
    different shape. _rebuild_rows globs that directory, so an unguarded
    `data["sentences"]` there would raise on every rescore of a run that saved
    no rows — the exact path that exists to salvage an hour of GPU.
    """
    soundfile = pytest.importorskip("soundfile")
    numpy = pytest.importorskip("numpy")

    out = tmp_path / "indicf5_xlit_probe"
    (out / "deva_ref_s0").mkdir(parents=True)
    soundfile.write(str(out / "deva_ref_s0" / "05.wav"),
                    numpy.zeros(probe.SAMPLE_RATE, dtype="float32"),
                    probe.SAMPLE_RATE, subtype="PCM_16")
    monkeypatch.setattr(probe, "OUT", out)

    rows = probe._rebuild_rows()

    assert [r["arm"] for r in rows] == ["deva_ref"]
    # and it is scored against deva_hand's spelling, not against nothing
    hand = json.loads((probe.XLIT / "deva_hand.json").read_text(encoding="utf-8"))
    assert rows[0]["given"] == {s["id"]: s["devanagari"]
                                for s in hand["sentences"]}[5]


def test_the_vocabulary_check_covers_the_reference_by_default():
    """
    A missing token in the reference transcript is how §4's spillover happens:
    the model cannot align ref_text to the reference audio, and
    infer_batch_process slices off `ref_audio_len` frames regardless. The
    fixture's own note tells a reader this file is checked; this is that
    promise.
    """
    vocab_check = pytest.importorskip("colab.vocab_check")
    import inspect

    default = inspect.signature(vocab_check.check_arms).parameters["arms"].default
    assert "reference_deva" in default
