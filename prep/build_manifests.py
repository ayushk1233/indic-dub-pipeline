"""
Gate and write the pre-flight manifests, once per route over identical clips (FINETUNE_PLAN §2d gates that
need no GPU; Whisper CER is step 2's; route decision of 2026-09-22).

  ./venv-train/bin/python -m prep.build_manifests finetune_data/preflight
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from train.data import write_manifest_dir
from train.model import load_vocab

CPS = (7.0, 20.0)
ROUTES = {"route_a": "deva", "route_b": "latin"}


def devanagari_fraction(text):
    letters = [c for c in text if not c.isspace()]
    return sum("ऀ" <= c <= "ॿ" for c in letters) / max(1, len(letters))


def latin_fraction(text):
    letters = [c for c in text if not c.isspace()]
    return sum("a" <= c <= "z" for c in letters) / max(1, len(letters))


def build(rows, xlit, vocab, out_root):
    out_root, counts, kept = Path(out_root), Counter(), {"train": [], "val": []}
    for r in rows:
        x = xlit[r["id"]]
        cps = len(x["latin"].replace(" ", "")) / r["duration"]
        if any(c not in vocab for c in x["deva"] + x["latin"]):
            counts["vocab"] += 1
        elif devanagari_fraction(x["deva"]) < 0.95:
            counts["devanagari"] += 1
        elif latin_fraction(x["latin"]) < 0.95:
            counts["latin"] += 1
        elif not CPS[0] <= cps <= CPS[1]:
            counts["cps"] += 1
        else:
            counts["kept"] += 1
            kept[r["split"]].append((r, x))
    for route, column in ROUTES.items():
        def rows_for(items):
            return [{"audio_path": r["audio_path"], "text": x[column], "duration": r["duration"]} for r, x in items]
        if kept["train"]:
            write_manifest_dir(rows_for(kept["train"]), out_root / route / "train")
            write_manifest_dir(rows_for(kept["train"][:16]), out_root / route / "overfit16")
        if kept["val"]:
            write_manifest_dir(rows_for(kept["val"]), out_root / route / "val_en", groups=["en"] * len(kept["val"]))
    return dict(counts)


def main(root):
    root = Path(root)
    rows = [json.loads(l) for l in (root / "rows.jsonl").read_text().splitlines()]
    counts = build(rows, json.loads((root / "xlit.json").read_text()), load_vocab(), root)
    (root / "gates.json").write_text(json.dumps(counts, indent=1))
    print(counts)


if __name__ == "__main__":
    main(sys.argv[1])
