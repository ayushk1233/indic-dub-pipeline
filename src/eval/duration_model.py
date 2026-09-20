"""
Predict how long a piece of text takes to speak.

Every feasibility verdict in this project reduces to one question: will this
text fit in this many seconds? The original answer divided characters by a
constant typed in by hand. This module replaces that with a model fitted to
real recordings, and — more importantly — measures whether the model is
actually better than the constant it replaces. If it is not, the constant
wins and this module says so.

Two ideas drive the feature set.

First, a recording is not pure articulation. It carries setup and trailing
silence that barely varies with length, so duration is better described by an
intercept plus a per-unit cost than by a single rate. A constant-rate model
cannot express that and systematically misjudges short clips.

Second, character count is a poor unit for Indic scripts. Devanagari vowel
signs and the virama attach to a preceding consonant rather than adding a
syllable, so "क्या" is four codepoints but one spoken syllable. Counting base
characters separately from combining marks gives the model a far better proxy
for syllable count, which is what speech duration actually tracks.
"""

import json
import statistics
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


# Codepoint categories that modify a preceding character instead of standing
# on their own: nonspacing marks and spacing combining marks.
COMBINING_CATEGORIES = {"Mn", "Mc"}

PUNCTUATION_CATEGORIES = {"Po", "Pd", "Ps", "Pe", "Pi", "Pf", "Pc"}


def text_features(text: str) -> dict[str, float]:
    """
    Reduce text to the handful of counts that predict its spoken length.
    """
    stripped = text.strip()

    base_chars = 0
    combining = 0
    punctuation = 0

    for char in stripped:
        if char.isspace():
            continue

        category = unicodedata.category(char)

        if category in COMBINING_CATEGORIES:
            combining += 1
        elif category in PUNCTUATION_CATEGORIES:
            punctuation += 1
        else:
            base_chars += 1

    return {
        "chars": float(len(stripped)),
        "base_chars": float(base_chars),
        "combining": float(combining),
        "words": float(len(stripped.split())),
        "punctuation": float(punctuation),
    }


FEATURE_ORDER = ("base_chars", "combining", "words", "punctuation")


def feature_vector(text: str) -> list[float]:
    features = text_features(text)
    return [features[name] for name in FEATURE_ORDER]


@dataclass
class ModelScore:
    """
    Held-out error for one predictor, in seconds.
    """

    name: str
    mae_s: float
    median_ae_s: float
    p90_ae_s: float
    num_samples: int


@dataclass
class LanguageDurationModel:
    """
    Fitted per-language duration predictor plus the baseline it must beat.
    """

    language: str

    coefficients: list[float] = field(default_factory=list)
    intercept: float = 0.0

    # The constant-rate fallback, in characters per second. Used when no
    # fitted model exists, and as the baseline the fit is scored against.
    fallback_cps: float = 13.0

    num_train: int = 0
    num_test: int = 0

    model_score: ModelScore | None = None
    baseline_score: ModelScore | None = None

    @property
    def beats_baseline(self) -> bool:
        if self.model_score is None or self.baseline_score is None:
            return False
        return self.model_score.mae_s < self.baseline_score.mae_s

    def predict(self, text: str, include_overhead: bool = False) -> float:
        """
        Predicted seconds to speak `text`.

        By default this returns *articulation time only*, excluding the fitted
        intercept, and that default is the important part.

        The intercept is fitted on FLEURS, where every clip is a studio
        recording carrying leading and trailing silence that barely varies with
        length. For Hindi it comes out at 1.54s. That overhead is real in the
        training data and entirely absent from the deployment data: synthesized
        segments dropped onto a dubbing timeline have no recording silence, and
        what little the decoder appends is trimmed during assembly.

        Including it produces nonsense at the lengths this pipeline actually
        sees. FLEURS sentences run 40 to 270 characters and never shorter than
        3.6 seconds; dubbing segments are routinely 8 to 80 characters in one
        to three second slots. With the intercept, eight characters of Hindi
        predicts 2.61 seconds, which would make length control believe almost
        everything overruns and shorten translations that were already fine.

        This is a training-versus-deployment distribution mismatch, not a bad
        fit. The per-feature costs transfer; the constant does not. Pass
        include_overhead=True to reproduce the numbers the model was scored on.

        Falls back to the constant rate when there is no fitted model, or when
        the fit failed to beat that constant on held-out data. Never returns a
        non-positive duration.
        """
        if not self.coefficients or not self.beats_baseline:
            return self._baseline_predict(text)

        vector = feature_vector(text)
        predicted = sum(
            coefficient * value
            for coefficient, value in zip(self.coefficients, vector)
        )

        if include_overhead:
            predicted += self.intercept

        return max(predicted, 0.05)

    def _baseline_predict(self, text: str) -> float:
        chars = len(text.strip())
        rate = self.fallback_cps if self.fallback_cps > 0 else 13.0
        return max(chars / rate, 0.05)


@dataclass
class DurationModelSet:
    """
    One fitted model per language, with the provenance of the fit.
    """

    models: dict[str, LanguageDurationModel] = field(default_factory=dict)
    source: str = "unfitted"

    def predict(
        self,
        text: str,
        language: str,
        include_overhead: bool = False,
    ) -> float:
        model = self.models.get(language)

        if model is None:
            model = LanguageDurationModel(language=language)

        return model.predict(text, include_overhead=include_overhead)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "source": self.source,
            "feature_order": list(FEATURE_ORDER),
            "models": {
                language: {
                    "language": model.language,
                    "coefficients": model.coefficients,
                    "intercept": model.intercept,
                    "fallback_cps": model.fallback_cps,
                    "num_train": model.num_train,
                    "num_test": model.num_test,
                    "beats_baseline": model.beats_baseline,
                    "model_score": (
                        vars(model.model_score) if model.model_score else None
                    ),
                    "baseline_score": (
                        vars(model.baseline_score) if model.baseline_score else None
                    ),
                }
                for language, model in self.models.items()
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return path

    @classmethod
    def load(cls, path: Path) -> "DurationModelSet":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if list(payload.get("feature_order", [])) != list(FEATURE_ORDER):
            raise ValueError(
                "Saved duration model uses a different feature order than this "
                "code expects; refit it rather than loading stale coefficients."
            )

        models = {}

        for language, entry in payload.get("models", {}).items():
            model = LanguageDurationModel(
                language=entry["language"],
                coefficients=entry.get("coefficients", []),
                intercept=entry.get("intercept", 0.0),
                fallback_cps=entry.get("fallback_cps", 13.0),
                num_train=entry.get("num_train", 0),
                num_test=entry.get("num_test", 0),
            )

            for key, target in (
                ("model_score", "model_score"),
                ("baseline_score", "baseline_score"),
            ):
                raw = entry.get(key)
                if raw:
                    setattr(model, target, ModelScore(**raw))

            models[language] = model

        return cls(models=models, source=payload.get("source", "loaded"))


def _score(name: str, errors: list[float]) -> ModelScore:
    ordered = sorted(errors)
    p90_index = min(int(0.9 * (len(ordered) - 1)), len(ordered) - 1) if ordered else 0

    return ModelScore(
        name=name,
        mae_s=statistics.fmean(errors) if errors else 0.0,
        median_ae_s=statistics.median(errors) if errors else 0.0,
        p90_ae_s=ordered[p90_index] if ordered else 0.0,
        num_samples=len(errors),
    )


def fit_language(
    samples: list,
    language: str,
    test_fraction: float = 0.25,
    alpha: float = 1.0,
) -> LanguageDurationModel:
    """
    Fit a ridge regression from text features to spoken duration, and score it
    against the constant-rate baseline on a held-out split.

    `samples` are SpeechSample records. The split is deterministic — every
    fourth sentence by position — so refitting gives the same answer twice and
    the comparison against the baseline stays honest across runs.
    """
    from sklearn.linear_model import Ridge

    usable = [s for s in samples if s.duration_s > 0 and s.chars > 0]

    if len(usable) < 8:
        return LanguageDurationModel(language=language, num_train=len(usable))

    stride = max(int(1 / test_fraction), 2)

    train = [s for i, s in enumerate(usable) if i % stride != 0]
    test = [s for i, s in enumerate(usable) if i % stride == 0]

    if not train or not test:
        return LanguageDurationModel(language=language, num_train=len(usable))

    # The baseline this fit has to beat: one rate for the language, taken from
    # the training half only so the comparison is not contaminated.
    fallback_cps = statistics.median(s.cps for s in train)

    ridge = Ridge(alpha=alpha)
    ridge.fit(
        [feature_vector(s.text) for s in train],
        [s.duration_s for s in train],
    )

    predictions = ridge.predict([feature_vector(s.text) for s in test])

    model_errors = [
        abs(float(predicted) - s.duration_s)
        for predicted, s in zip(predictions, test)
    ]
    baseline_errors = [
        abs(s.chars / fallback_cps - s.duration_s)
        for s in test
    ]

    return LanguageDurationModel(
        language=language,
        coefficients=[float(c) for c in ridge.coef_],
        intercept=float(ridge.intercept_),
        fallback_cps=float(fallback_cps),
        num_train=len(train),
        num_test=len(test),
        model_score=_score("features+ridge", model_errors),
        baseline_score=_score("constant cps", baseline_errors),
    )


def render_scores(model_set: DurationModelSet) -> str:
    lines: list[str] = []

    lines.append("=" * 78)
    lines.append("DURATION MODEL   held-out error against the constant-rate baseline")
    lines.append("=" * 78)
    lines.append("")
    lines.append(
        f"  {'lang':>5} {'train':>6} {'test':>5} "
        f"{'baseline':>9} {'model':>8} {'change':>8}  verdict"
    )

    for language in sorted(model_set.models):
        model = model_set.models[language]

        if model.model_score is None or model.baseline_score is None:
            lines.append(f"  {language:>5}  not enough data to fit")
            continue

        baseline_mae = model.baseline_score.mae_s
        model_mae = model.model_score.mae_s
        change = (
            100.0 * (model_mae - baseline_mae) / baseline_mae
            if baseline_mae > 0
            else 0.0
        )

        lines.append(
            f"  {language:>5} {model.num_train:>6} {model.num_test:>5} "
            f"{baseline_mae:>8.3f}s {model_mae:>7.3f}s {change:>7.1f}%  "
            + ("model wins" if model.beats_baseline else "BASELINE WINS, keeping it")
        )

    lines.append("")
    lines.append("  Error is mean absolute error in seconds on held-out sentences.")
    lines.append("  A model that loses is not used: predict() falls back to the constant.")
    lines.append("")
    lines.append("  These scores include the fitted intercept, which is FLEURS recording")
    lines.append("  overhead. predict() excludes it by default, because synthesized speech")
    lines.append("  on a dubbing timeline does not carry it. See predict()'s docstring.")
    lines.append("")
    lines.append("=" * 78)

    return "\n".join(lines)


DEFAULT_MODEL_PATH = Path("artifacts") / "measurements" / "duration_model.json"


def fit_from_fleurs(
    languages: list[str],
    split: str = "validation",
    limit: int | None = None,
) -> DurationModelSet:
    from src.data.fleurs import load_samples

    models = {}

    for language in languages:
        print(f"Fitting {language}...", flush=True)
        samples = load_samples(language, split=split, limit=limit)
        models[language] = fit_language(samples, language)
        print(f"  {len(samples)} sentences", flush=True)

    return DurationModelSet(
        models=models,
        source=f"FLEURS {split}" + (f", capped at {limit}" if limit else ""),
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Fit and score the spoken-duration model against FLEURS."
    )
    parser.add_argument("--languages", nargs="+", default=["en", "hi"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    model_set = fit_from_fleurs(
        args.languages,
        split=args.split,
        limit=args.limit,
    )

    print()
    print(render_scores(model_set))

    path = model_set.save(Path(args.out))
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()
