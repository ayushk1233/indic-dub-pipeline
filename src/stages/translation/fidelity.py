"""
Score how much of a source sentence survives in a translation.

Length control only works if something stops it from preferring a short
candidate that dropped half the meaning. That guard is this module.

The check is cross-lingual sentence embedding similarity: encode the English
source and the Indic candidate into a shared vector space with LaBSE and take
the cosine between them. LaBSE is trained so that translations of the same
sentence land close together across 100+ languages, which is exactly the
property needed, and it is Apache-2.0.

What this deliberately does NOT catch, measured rather than assumed: output
in the wrong language. Diverse beam search on IndicTrans2 sometimes pushes a
beam group into a different language sharing the script, returning Maithili
for a Hindi request. Scored against the English source, that Maithili
candidate came back at 0.888 — higher than every correct Hindi candidate,
which sat near 0.86. LaBSE is behaving exactly as designed: it is trained to
be language-agnostic, so a faithful translation into the wrong language is
still a faithful translation by this measure.

Fidelity and target-language correctness are therefore orthogonal checks, and
this module only does the first. See `language_check.py` for the second.
"""

from dataclasses import dataclass

# Multilingual sentence encoder aligned across languages, Apache-2.0.
DEFAULT_MODEL_ID = "sentence-transformers/LaBSE"


@dataclass
class FidelityScorer:
    """
    Cosine similarity between a source sentence and its candidate translations.

    The model is loaded on first use, not on construction, so that importing
    the pipeline does not pull 1.8 GB off the network. Construct it once and
    reuse it; loading per segment would dominate runtime.
    """

    model_id: str = DEFAULT_MODEL_ID
    device: str = "cpu"

    _model: object = None

    def load(self) -> None:
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_id, device=self.device)

    def score(self, source: str, candidates: list[str]) -> list[float]:
        """
        Similarity of each candidate to the source, in the same order.

        Empty candidates score zero rather than raising, because an empty
        translation is a real thing the model produces and it should be
        ranked last rather than crashing the run.
        """
        if not candidates:
            return []

        self.load()

        from sentence_transformers import util

        embeddings = self._model.encode(
            [source] + candidates,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        similarities = util.cos_sim(embeddings[0], embeddings[1:])[0]

        return [
            0.0 if not candidate.strip() else float(value)
            for candidate, value in zip(candidates, similarities)
        ]


class NullFidelityScorer:
    """
    Stand-in used when embedding scoring is unavailable or switched off.

    Returns no scores at all rather than fake ones, which makes length control
    fall back to choosing on duration alone. That is a worse policy, but it is
    an honest one; inventing a fidelity number would be worse still.
    """

    def load(self) -> None:
        return

    def score(self, source: str, candidates: list[str]) -> list[float] | None:
        return None
