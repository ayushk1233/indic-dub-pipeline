"""
Tests for detecting output in the wrong language.

This check exists because of a measured failure: diverse beam search returned
Maithili for a Hindi request, and LaBSE scored that candidate 0.888 against
the English source while correct Hindi candidates scored around 0.86. The
semantic metric ranked the wrong-language output best, so something else has
to catch it.
"""

from src.stages.translation.language_check import (
    is_off_language,
    language_score,
    tokenize,
)
from src.stages.translation.length_control import (
    Candidate,
    score_candidates,
    select,
)


GOOD_HINDI = (
    "वैज्ञानिकों का मानना है कि ओसेलोट गंध से जानवरों "
    "का पता लगाते हैं और उन्हें खाते हैं ।"
)
MAITHILI = (
    "वैज्ञानिकसभक माननाइ छैक कि ओसेलोट गन्ध द्वारा "
    "जानवरसभकेँ ट्रैक करैत छैक आ खाइत छैक ।"
)


class FixedRateModel:
    def predict(self, text, language):
        return max(len(text.strip()) / 10.0, 0.05)


def test_tokenizer_strips_devanagari_danda_and_punctuation():
    assert tokenize("नमस्ते, शो में स्वागत है ।") == [
        "नमस्ते",
        "शो",
        "में",
        "स्वागत",
        "है",
    ]


def test_hindi_is_recognised_as_hindi():
    verdict = language_score(GOOD_HINDI, "hi")

    assert verdict.checked
    assert verdict.score == 1.0
    assert verdict.matched > 0
    assert not verdict.suspect


def test_maithili_is_flagged_for_a_hindi_request():
    verdict = language_score(MAITHILI, "hi")

    assert verdict.suspect
    assert verdict.competing > 0
    assert verdict.matched == 0


def test_a_language_with_no_script_neighbour_is_not_checked():
    # Tamil has its own script, so the script check already settles it and a
    # marker verdict would only add noise.
    verdict = language_score("சிலர் நம்புகிறார்கள்", "ta")

    assert not verdict.checked
    assert not verdict.suspect


def test_text_with_no_marker_evidence_is_not_suspect():
    # A short phrase may legitimately contain no function words. Guessing from
    # no evidence would reject good translations.
    verdict = language_score("ओसेलोट", "hi")

    assert not verdict.checked
    assert not verdict.suspect


def test_marathi_is_distinguished_from_hindi():
    assert not is_off_language("हे पुस्तक चांगले आहे आणि मला आवडते", "mr")
    assert is_off_language("यह किताब अच्छी है और मुझे पसंद है", "mr")


def test_an_off_language_candidate_is_never_acceptable():
    candidate = Candidate(
        text=MAITHILI,
        predicted_duration_s=1.0,
        budget_s=2.0,
        fit_ratio=0.5,
        fidelity=0.99,        # scores brilliantly on meaning
        off_language=True,
    )

    assert candidate.fits
    assert not candidate.acceptable


def test_selection_rejects_the_off_language_candidate_even_when_it_scores_best():
    # This reproduces the real measurement: the wrong-language candidate is
    # both shorter and higher-scoring than the correct one.
    candidates = score_candidates(
        [GOOD_HINDI, MAITHILI],
        budget_s=8.0,
        language="hi",
        duration_model=FixedRateModel(),
        fidelities=[0.86, 0.888],
    )

    assert candidates[1].off_language

    selection = select(candidates)

    assert selection.chosen.text == GOOD_HINDI


def test_when_everything_is_off_language_that_is_said_plainly():
    candidates = score_candidates(
        [MAITHILI],
        budget_s=8.0,
        language="hi",
        duration_model=FixedRateModel(),
        fidelities=[0.9],
    )

    selection = select(candidates)

    assert "off-language" in selection.reason
