"""
English transcripts in both forms, frozen (FINETUNE_PLAN §2d step 6, §3 step 5b): the normalised Latin
text (Route B) and its Devanagari transliteration (Route A). Runs in ./venv, which has src.text and g2p_en;
the training side only ever reads the JSON this writes.

  ./venv/bin/python -m prep.xlit_freeze finetune_data/preflight/rows.jsonl finetune_data/preflight/xlit.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from src.text.numbers import spell_numbers
from src.text.transliterate import transliterate_to_devanagari


def normalize_en(text):
    t = spell_numbers(text.lower(), "en")
    t = t.replace("'", "")
    t = re.sub(r"[^\w\s]|_", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def main(rows_jsonl, out_json):
    out = {}
    for line in Path(rows_jsonl).read_text().splitlines():
        r = json.loads(line)
        latin = normalize_en(r["text_latin"])
        out[r["id"]] = {"latin": latin, "deva": transliterate_to_devanagari(latin)}
    Path(out_json).write_text(json.dumps(out, ensure_ascii=False, indent=0))
    print(f"{len(out)} transcripts frozen")


if __name__ == "__main__":
    main(*sys.argv[1:3])
