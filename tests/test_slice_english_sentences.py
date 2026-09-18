"""
The slicer decides how long the speaker takes to say each sentence, and that
number becomes `fix_duration` for the probe. A wrong span is not a loud
failure: it produces a plausible clip, a plausible duration, and a ruler that
is quietly wrong for every metric built on it.

Both tests below are failures that actually happened on the first run against
fixtures/english_speech.wav, not hypotheticals.

  - "Thirty-one fit perfectly." came back as `fit perfectly,` in 0.92s — 27
    characters per second, faster than anything this project has measured from
    a human or a model. Whisper writes "31" and the scripted text writes
    "Thirty-one", and on a letters-only projection those share nothing, so the
    subject was dropped from the span.

  - "Seven were impossible, and we had to rewrite them." came back opening with
    `needed stretching,` — the tail of "Nine needed stretching.", a sentence
    fixture7 deliberately excludes. The span was taken from the first to the
    last matching block over the whole remaining transcript, and incidental
    matches stretched it backwards across the boundary.

No audio and no ASR here: `locate` is given a word list directly.
"""

import pytest

slicer = pytest.importorskip("scripts.slice_english_sentences")


@pytest.fixture
def words():
    """
    A word list shaped like faster_whisper's, one word every half second.

    The timings are regular because these tests are about *which* words a span
    covers; the duration arithmetic on top of it is the caller's.
    """
    def build(text, step=0.5):
        return [{"text": word, "start": index * step, "end": (index + 1) * step}
                for index, word in enumerate(text.split())]

    return build


def test_a_number_spelled_out_matches_the_same_number_in_digits(words):
    """Sentence 5: the scripted 'Thirty-one' against Whisper's '31'."""
    heard = words("31 fit perfectly, 7 were impossible")
    found = slicer.locate(heard, "Thirty-one fit perfectly.")

    assert found["first"] == 0, "the number was dropped from the front of the span"
    assert found["last"] == 2
    assert found["ratio"] > 0.9


def test_percent_survives_the_change_of_notation(words):
    heard = words("about 20 % longer to say")
    found = slicer.locate(heard, "about twenty percent longer")

    assert found["ratio"] > 0.9


def test_the_span_stops_at_the_sentence_it_was_asked_for(words):
    """
    Sentence 6: the words before it belong to a sentence fixture7 excludes,
    and must not be pulled into the slice.
    """
    heard = words("Nine needed stretching, 7 were impossible and we had to "
                  "rewrite them")
    found = slicer.locate(heard, "Seven were impossible, and we had to rewrite them.")

    assert heard[found["first"]]["text"] == "7"
    assert found["ratio"] > 0.9


def test_search_from_keeps_a_repeated_phrase_from_matching_backwards(words):
    heard = words("fit perfectly and later things fit perfectly again")
    first = slicer.locate(heard, "fit perfectly")
    second = slicer.locate(heard, "fit perfectly", search_from=first["last"] + 1)

    assert second["first"] > first["last"]


def test_a_sentence_that_was_never_said_scores_below_the_floor(words):
    """
    The guard that matters. Slicing on a bad match would hand every downstream
    metric a plausible, wrong span, so main() raises instead of writing it.
    """
    heard = words("Hindi takes about 20 % longer to say than the English")
    found = slicer.locate(heard, "The quick brown fox jumped over the lazy dog.")

    assert found["ratio"] < slicer.MIN_MATCH


def test_the_real_sentences_clear_the_floor_by_a_margin(words):
    """
    The floor is loose because the speaker paraphrased and the ASR has its own
    error. It is not so loose that it would accept anything: the worst real
    sentence should still sit well above it.
    """
    heard = words("You get a video, lecture and interview anything with clear "
                  "speech and it gives you back the same video speaking in Hindi")
    found = slicer.locate(
        heard,
        "You give it a video — a lecture, an interview, anything with clear "
        "speech — and it gives you back the same video speaking Hindi."
    )

    assert found["ratio"] > slicer.MIN_MATCH + 0.2


def test_numbers_past_the_table_are_left_as_digits():
    """`_key` must not raise on something outside the small number table."""
    assert slicer._key("4096") == "4096"
    assert slicer._key("100") == slicer._key("one hundred")
