"""
The calibrated scale, extracted so more than one experiment can stand on it.

A raw speaker-embedding cosine means nothing on its own. It needs a floor —
what a stranger scores, which is not zero — and a ceiling measured at the same
clip length as the thing being judged, because a speaker embedding taken from
five seconds is a noisier estimate than one from twenty and regresses toward
the population mean. Getting either wrong has moved a headline number in this
project by more than ten points, three separate times.

This is the same construction colab/indicf5_check.py performs inline. It is
lifted here rather than imported from there because that module builds it
halfway through a long main() that also loads two TTS models. Once a run has
confirmed the two agree, indicf5_check should call this instead.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from colab.english_report import cosine

ANCHOR = {"gpt_cond_len": 60, "gpt_cond_chunk_len": 30,
          "max_ref_length": 60, "sound_norm_refs": False}

# Ceilings measured at roughly 18s, 10s, 6s and 4s.
SPLITS = (3, 6, 10, 14)


def ceiling_for(duration, points):
    """The ceiling measured at whichever clip length is closest to this one."""
    if not points:
        return float("nan")
    return min(points, key=lambda point: abs(point[0] - duration))[1]


def position(score, duration, points, floor):
    """Where a score sits between a stranger and the speaker himself, as a
    percentage, against the ceiling for this clip's own length."""
    ceiling = ceiling_for(duration, points)
    span = ceiling - floor
    if not np.isfinite(score) or not np.isfinite(span) or span <= 0:
        return float("nan")
    return 100.0 * (score - floor) / span


@dataclass
class Scale:
    floor: float
    anchors: dict = field(default_factory=dict)
    curves: dict = field(default_factory=dict)

    def position(self, score, duration, curve):
        return position(score, duration, self.curves[curve], self.floor)

    def ceiling(self, duration, curve):
        return ceiling_for(duration, self.curves[curve])


def build(xtts, fixtures, scratch):
    """
    Measure the floor and the two ceiling curves with the loaded XTTS encoder.

    No synthesis happens here. Every ceiling reading is the speaker against
    himself on real tape, and the floor is the speaker against 58 studio
    speakers who are not him.
    """
    fixtures, scratch = Path(fixtures), Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    def latents(path, **params):
        with torch.no_grad():
            return xtts.get_conditioning_latents(audio_path=[str(path)], **params)

    def embed(path):
        try:
            _, embedding = latents(path)
            return embedding
        except Exception:
            return None

    _, anchor_en = latents(fixtures / "english_speech.wav", **ANCHOR)
    _, anchor_hi = latents(fixtures / "hindi_speech.wav", **ANCHOR)

    def pieces(path, count, label, anchor):
        audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        edges = np.linspace(0, audio.size, count + 1).astype(int)
        scores, durations = [], []
        for index in range(count):
            piece = scratch / f"{label}_{count}_{index}.wav"
            sf.write(str(piece), audio[edges[index]:edges[index + 1]], rate,
                     subtype="PCM_16")
            embedding = embed(piece)
            if embedding is not None:
                scores.append(cosine(anchor, embedding))
                durations.append((edges[index + 1] - edges[index]) / rate)

        if not scores:
            return None
        return float(np.mean(durations)), float(np.mean(scores))

    def curve(path, label, anchor):
        points = [pt for pt in (pieces(path, n, label, anchor) for n in SPLITS)
                  if pt is not None]
        return sorted(points)

    floor_scores = []
    try:
        for _, entry in xtts.speaker_manager.speakers.items():
            floor_scores.append(cosine(anchor_en, entry["speaker_embedding"]))
    except Exception:
        pass

    return Scale(
        floor=float(np.median(floor_scores)) if floor_scores else 0.0,
        anchors={"en": anchor_en, "hi": anchor_hi},
        curves={
            "same": curve(fixtures / "english_speech.wav", "en_piece", anchor_en),
            "cross": curve(fixtures / "hindi_speech.wav", "hi_piece", anchor_en),
        },
    )
