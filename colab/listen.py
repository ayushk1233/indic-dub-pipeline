"""
Play the decoder sweep side by side.

Grouped by segment rather than by configuration, because the only comparison
that matters is the same sentence spoken five ways.
"""

import json
import wave
from pathlib import Path

import numpy as np
from IPython.display import Audio, HTML, display

BUNDLE = Path("/content/tts_bundle")
OUT_ROOT = Path("/content/experiments")

# Set to "/content/model_comparison" to play that sweep instead.
OUT_ROOT_OVERRIDE = None

CONFIG_ORDER = [
    "greedy", "sampled", "sampled_fast", "sampled_split", "greedy_fast",
    "xtts_short_cond", "xtts_long_cond", "indicf5",
]

NOTES = {
    "greedy": "what the pipeline ships today",
    "sampled": "XTTS defaults, the prosody reference point",
    "sampled_fast": "sampling plus speed=1.2",
    "sampled_split": "sampling plus sentence splitting",
    "greedy_fast": "greedy plus speed=1.2, isolates rate control",
    "xtts_short_cond": "XTTS, reference capped at 10s (what ships today)",
    "xtts_long_cond": "XTTS, full 21s reference",
    "indicf5": "IndicF5, trained on Indian languages",
}


def duration_of(path):
    with wave.open(str(path), "rb") as h:
        return h.getnframes() / h.getframerate()


def heading(text, size=18):
    display(HTML(f"<h3 style='margin:18px 0 4px'>{text}</h3>"))


def main():
    global OUT_ROOT
    if OUT_ROOT_OVERRIDE:
        OUT_ROOT = Path(OUT_ROOT_OVERRIDE)

    request = json.loads((BUNDLE / "request" / "synthesis_request.json").read_text())

    heading("Reference — the voice being cloned")
    display(Audio(filename=str(BUNDLE / "request" / "reference.wav")))

    for segment in request["segments"]:
        sid = segment["segment_id"]
        slot = segment["end_ts"] - segment["start_ts"]

        heading(f"Segment {sid} — {slot:.2f}s slot, {len(segment['text'])} chars")
        display(HTML(
            f"<div style='font-size:15px;line-height:1.6;margin:4px 0 10px'>"
            f"{segment['text']}</div>"
        ))

        for name in CONFIG_ORDER:
            path = OUT_ROOT / name / f"seg_{sid:05d}.wav"
            if not path.exists():
                path = Path("/content/model_comparison") / name / f"seg_{sid:05d}.wav"
            if not path.exists():
                continue
            d = duration_of(path)
            display(HTML(
                f"<div style='margin-top:8px'><b>{name}</b> "
                f"<span style='color:#666'>&mdash; {d:.2f}s, "
                f"{d / slot:.2f}x slot, {len(segment['text']) / d:.1f} cps "
                f"&mdash; {NOTES[name]}</span></div>"
            ))
            display(Audio(filename=str(path)))

    spread = sorted(OUT_ROOT.glob("spread/run_*.wav"))

    if spread:
        heading("Same input, four samples — the duration spread")
        display(HTML(
            "<div style='color:#666'>These four differ only by the random draw. "
            "Judge whether they differ in quality, not just in length.</div>"
        ))
        for path in spread:
            display(HTML(f"<div style='margin-top:8px'><b>{path.stem}</b> "
                         f"<span style='color:#666'>&mdash; "
                         f"{duration_of(path):.2f}s</span></div>"))
            display(Audio(filename=str(path)))


if __name__ == "__main__":
    main()
