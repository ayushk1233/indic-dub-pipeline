"""
Choose a translation that fits the time available.

The measured problem this project exists around: Hindi takes materially longer
to speak than the English it replaces, so a faithful translation routinely
overruns the slot it has to fit. Every downstream lever for fixing that —
time-stretching, letting segments drift — degrades the result. The cheapest
place to solve it is here, before a single GPU second is spent, by not
producing an unspeakable translation in the first place.

So instead of taking the one output beam search likes best, we ask the model
for several, and choose among them on two axes that pull against each other:

  fit       predicted spoken duration against the budget, using the duration
            model fitted to real recordings
  fidelity  how much of the source meaning survived, as cosine similarity
            between multilingual sentence embeddings of source and candidate

A shorter candidate is worthless if it dropped half the sentence, and a
perfect translation is worthless if nobody can say it in two seconds. The
selection policy below refuses to trade fidelity below a floor, and within
that constraint takes the best fit it can get.

A third axis turned out to be necessary and is not optional. Diverse beam
search sometimes returns a candidate in the wrong language — Maithili for a
Hindi request — and LaBSE ranks those *highest*, because it is trained to be
language-agnostic and the meaning genuinely did survive. Fidelity alone
therefore cannot guard against it, so candidates are additionally screened by
`language_check`, and an off-language candidate is disqualified outright
rather than scored down. There is no length worth shipping the wrong language
for.

Recording every candidate's two scores is deliberate: the distribution across
a corpus is the fidelity-versus-fit tradeoff curve, which is the honest way to
report what this technique costs.
"""

from dataclasses import dataclass, field

from src.stages.translation.language_check import is_off_language


# Cosine similarity to the source below which a candidate is not a translation
# of it any more, whatever its length.
FIDELITY_FLOOR = 0.80

# A candidate whose predicted duration exceeds its budget by more than this is
# not worth preferring over a slightly less faithful one that fits.
FIT_CEILING = 1.0

# Candidates whose fit is within this of each other count as equally good, so
# the tie is broken on fidelity rather than on a meaningless third decimal.
FIT_EPSILON = 0.02


@dataclass
class Candidate:
    """
    One translation option, scored on both axes.
    """

    text: str

    predicted_duration_s: float
    budget_s: float

    # Predicted duration divided by budget. At or below 1.0 it fits.
    fit_ratio: float

    # Cosine similarity to the source sentence. None when no scorer was
    # supplied, in which case selection falls back to length alone.
    fidelity: float | None = None

    # True when the text looks like a different language that shares the
    # target's script. Disqualifying, not merely penalised.
    off_language: bool = False

    @property
    def fits(self) -> bool:
        return self.fit_ratio <= FIT_CEILING

    @property
    def acceptable(self) -> bool:
        """
        Eligible to be chosen at all.

        Off-language candidates are excluded whatever they score, because the
        fidelity metric cannot see the problem and length cannot excuse it.
        With no fidelity score available, everything else is considered and
        length decides.
        """
        if self.off_language:
            return False

        return self.fidelity is None or self.fidelity >= FIDELITY_FLOOR


@dataclass
class Selection:
    """
    The chosen translation, with everything that was rejected to get there.
    """

    chosen: Candidate
    candidates: list[Candidate] = field(default_factory=list)

    # Which branch of the policy produced the choice. Worth persisting: when a
    # segment sounds wrong, this says whether length control compromised it.
    reason: str = ""

    @property
    def num_candidates(self) -> int:
        return len(self.candidates)

    @property
    def baseline(self) -> Candidate | None:
        """
        What plain beam search would have returned, which is the first
        candidate the model produced. The comparison against `chosen` is the
        measurable effect of length control.
        """
        return self.candidates[0] if self.candidates else None

    @property
    def fidelity_cost(self) -> float:
        """
        How much fidelity was given up against the baseline. Zero when length
        control changed nothing, positive when it traded meaning for fit.
        """
        base = self.baseline

        if base is None or base.fidelity is None or self.chosen.fidelity is None:
            return 0.0

        return max(base.fidelity - self.chosen.fidelity, 0.0)

    @property
    def fit_gain(self) -> float:
        """
        How much overrun was removed against the baseline.
        """
        base = self.baseline

        if base is None:
            return 0.0

        return max(base.fit_ratio - self.chosen.fit_ratio, 0.0)


def score_candidates(
    texts: list[str],
    budget_s: float,
    language: str,
    duration_model,
    fidelities: list[float] | None = None,
) -> list[Candidate]:
    """
    Attach a predicted duration and, when available, a fidelity score to each
    candidate translation.

    `duration_model` is anything exposing predict(text, language) -> seconds;
    `DurationModelSet` is the intended one.
    """
    budget = max(budget_s, 1e-6)
    candidates: list[Candidate] = []

    for index, text in enumerate(texts):
        predicted = duration_model.predict(text, language)

        candidates.append(
            Candidate(
                text=text,
                predicted_duration_s=predicted,
                budget_s=budget,
                fit_ratio=predicted / budget,
                fidelity=(
                    fidelities[index]
                    if fidelities is not None and index < len(fidelities)
                    else None
                ),
                off_language=is_off_language(text, language),
            )
        )

    return candidates


def select(candidates: list[Candidate]) -> Selection:
    """
    Pick the best candidate under the policy described at the top of this file.

    The order of preference:

      1. Faithful enough and fits. Among these take the most faithful, since
         every one of them already solves the timing problem.
      2. Faithful enough but none fit. Take the closest to fitting, because
         the cascade downstream can absorb a small overrun and cannot absorb
         a large one.
      3. Nothing is faithful enough. Take the most faithful available and say
         so, rather than shipping a short candidate that lost the meaning.
         A segment that overruns is a timing problem; a segment that says the
         wrong thing is a correctness problem, and the second is worse.
    """
    if not candidates:
        raise ValueError("select() needs at least one candidate")

    acceptable = [c for c in candidates if c.acceptable]

    if not acceptable:
        # Prefer something in the right language even if it scored poorly,
        # over something fluent in the wrong one.
        on_language = [c for c in candidates if not c.off_language]
        pool = on_language or candidates

        chosen = max(pool, key=lambda c: (c.fidelity or 0.0))

        return Selection(
            chosen=chosen,
            candidates=candidates,
            reason=(
                "every candidate was off-language; kept the most faithful"
                if not on_language
                else "no candidate met the fidelity floor; kept the most faithful"
            ),
        )

    fitting = [c for c in acceptable if c.fits]

    if fitting:
        best_fit = min(c.fit_ratio for c in fitting)
        contenders = [
            c for c in fitting if c.fit_ratio <= best_fit + FIT_EPSILON
        ]
        chosen = max(
            contenders,
            key=lambda c: (c.fidelity if c.fidelity is not None else 0.0),
        )
        return Selection(
            chosen=chosen,
            candidates=candidates,
            reason="fits the budget",
        )

    chosen = min(acceptable, key=lambda c: c.fit_ratio)

    return Selection(
        chosen=chosen,
        candidates=candidates,
        reason="nothing fit; kept the closest that stayed faithful",
    )


def choose_translation(
    texts: list[str],
    budget_s: float,
    language: str,
    duration_model,
    fidelities: list[float] | None = None,
) -> Selection:
    """
    Score and select in one call.
    """
    return select(
        score_candidates(
            texts,
            budget_s=budget_s,
            language=language,
            duration_model=duration_model,
            fidelities=fidelities,
        )
    )
