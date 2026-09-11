"""
FLEURS loader, used to turn guessed constants into measured ones.

FLEURS is the speech counterpart of FLORES-101: the same 2,000 sentences read
aloud by native speakers in 102 languages, with the `id` field identical across
languages. That n-way parallelism is what makes it the right instrument here,
because one sentence id gives both the English recording and the Hindi
recording of the same content, so the duration expansion of a dub can be
measured directly instead of assumed.

Audio is never decoded. Every measurement this module needs comes from the
`num_samples` field, which is why the heavy `torchcodec` dependency is avoided
and why a whole language can be scanned by streaming rather than downloading.
"""

import json
from dataclasses import dataclass
from pathlib import Path

# FLEURS records everything at 16 kHz.
FLEURS_SAMPLE_RATE = 16000

# Our two-letter pipeline codes mapped onto FLEURS config names. The right-hand
# side is verified against the dataset's own lang_id label set.
FLEURS_CONFIGS = {
    "en": "en_us",
    "as": "as_in",
    "bn": "bn_in",
    "gu": "gu_in",
    "hi": "hi_in",
    "kn": "kn_in",
    "ml": "ml_in",
    "mr": "mr_in",
    "ne": "ne_np",
    "or": "or_in",
    "pa": "pa_in",
    "sd": "sd_in",
    "ta": "ta_in",
    "te": "te_in",
    "ur": "ur_pk",
}


@dataclass(frozen=True)
class SpeechSample:
    """
    One recorded sentence, reduced to the fields that matter for timing.
    """

    sentence_id: int
    language: str
    text: str
    duration_s: float

    @property
    def chars(self) -> int:
        return len(self.text.strip())

    @property
    def words(self) -> int:
        return len(self.text.split())

    @property
    def cps(self) -> float:
        """
        Characters per second over the whole clip.

        This is a clip rate, not an articulation rate: FLEURS recordings carry
        some leading and trailing silence, so it understates how fast the
        speaker actually talks. That overhead is real and roughly constant,
        which is precisely what the duration model's intercept is for.
        """
        return self.chars / self.duration_s if self.duration_s > 0 else 0.0


def load_samples(
    language: str,
    split: str = "validation",
    limit: int | None = None,
    streaming: bool = True,
    use_cache: bool = True,
) -> list[SpeechSample]:
    """
    Load FLEURS sentences for one language.

    `language` is a two-letter pipeline code such as "hi", not a FLEURS config
    name. Pass limit=None to scan the whole split.

    Results are cached to disk as JSON, because streaming a language pulls the
    audio bytes over the network even though only the sample counts are read,
    which takes minutes and has been seen to time out. The cached form is a few
    hundred kilobytes and makes refitting instant.
    """
    from datasets import Audio, load_dataset

    cached = _read_cache(language, split, limit) if use_cache else None

    if cached is not None:
        return cached

    config = FLEURS_CONFIGS.get(language)

    if config is None:
        raise ValueError(
            f"No FLEURS config for language {language!r}. "
            f"Known: {sorted(FLEURS_CONFIGS)}"
        )

    dataset = load_dataset(
        "google/fleurs",
        config,
        split=split,
        streaming=streaming,
    )

    # Turn off audio decoding before iterating. Without this the loader tries
    # to decode every clip and requires torchcodec, for data we never read.
    dataset = dataset.cast_column("audio", Audio(decode=False))

    samples: list[SpeechSample] = []

    for record in dataset:
        text = (record.get("transcription") or "").strip()
        num_samples = record.get("num_samples") or 0

        if not text or num_samples <= 0:
            continue

        samples.append(
            SpeechSample(
                sentence_id=int(record["id"]),
                language=language,
                text=text,
                duration_s=num_samples / FLEURS_SAMPLE_RATE,
            )
        )

        if limit is not None and len(samples) >= limit:
            break

    if use_cache:
        # Only a run that scanned to the end of the split holds every sample,
        # so only that run may claim the cache is complete. Without this a
        # capped run would satisfy a later request for the whole split.
        _write_cache(language, split, samples, complete=limit is None)

    return samples


def _read_cache(
    language: str,
    split: str,
    limit: int | None,
) -> list[SpeechSample] | None:
    """
    Return cached samples, but only when the cache holds at least as many as
    were asked for. A cache built with a small limit must not silently satisfy
    a later request for the whole split.
    """
    path = cache_path(language, split)

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    records = payload.get("samples", [])
    complete = payload.get("complete", False)

    if limit is None and not complete:
        return None

    if limit is not None and len(records) < limit:
        return None

    samples = [
        SpeechSample(
            sentence_id=r["sentence_id"],
            language=r["language"],
            text=r["text"],
            duration_s=r["duration_s"],
        )
        for r in records
    ]

    return samples[:limit] if limit is not None else samples


def _write_cache(
    language: str,
    split: str,
    samples: list[SpeechSample],
    complete: bool,
) -> None:
    path = cache_path(language, split)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = 0
    existing_complete = False

    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
                existing = len(payload.get("samples", []))
                existing_complete = payload.get("complete", False)
        except (OSError, json.JSONDecodeError):
            existing = 0

    # Never shrink a cache, and never downgrade a complete one to partial: a
    # capped run must not overwrite a full scan.
    if len(samples) < existing or (existing_complete and not complete):
        return

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "language": language,
                "split": split,
                "complete": complete,
                "samples": [
                    {
                        "sentence_id": s.sentence_id,
                        "language": s.language,
                        "text": s.text,
                        "duration_s": s.duration_s,
                    }
                    for s in samples
                ],
            },
            f,
            ensure_ascii=False,
        )


def pair_by_sentence(
    source: list[SpeechSample],
    target: list[SpeechSample],
) -> list[tuple[SpeechSample, SpeechSample]]:
    """
    Join two languages on the shared sentence id.

    The result is the same content spoken twice, which is the only honest way
    to measure how much longer a dub runs than its source.
    """
    by_id = {sample.sentence_id: sample for sample in target}

    return [
        (sample, by_id[sample.sentence_id])
        for sample in source
        if sample.sentence_id in by_id
    ]


def cache_path(language: str, split: str) -> Path:
    return Path("artifacts") / "measurements" / f"fleurs_{language}_{split}.json"
