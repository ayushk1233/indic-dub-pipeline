"""
Pre-flight slice of GLOBE's Indian-accent rows (FINETUNE_PLAN §8b): speaker-disjoint val, 3-20 s clips,
20-minute cap per speaker (§2f). Reads only the row groups it needs, over HTTP range requests, and
writes 24 kHz mono 16-bit FLAC (§2c).

  ./venv-train/bin/python -m prep.globe_slice --index finetune_data/globe_indian_index.json \
      --out finetune_data/preflight
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

REV = "579ae1077cac8fda90fbc880c4146f2e4dcf5f03"
LOW, HIGH = 3.0, 20.0


def _key(s, seed):
    return hashlib.sha256(f"{seed}:{s}".encode()).hexdigest()


def _order(r):
    return r["shard"], r["row_group"], r["row_in_group"]


def select(index_rows, *, train_hours=3.0, val_clips=150, cap_s=1200, seed=0):
    by_spk = defaultdict(list)
    for r in index_rows:
        if LOW <= r["duration"] <= HIGH:
            by_spk[r["speaker_id"]].append(r)
    speakers = sorted(by_spk, key=lambda s: _key(s, seed))
    val = []
    for s in speakers:                       # val: whole speakers until val_clips is reached
        if len(val) >= val_clips:
            break
        val.extend(sorted(by_spk[s], key=_order))
    val = val[:val_clips]
    val_spk = {r["speaker_id"] for r in val}
    train, used = [], 0.0
    for s in speakers:
        if s in val_spk:
            continue
        total = 0.0
        for r in sorted(by_spk[s], key=_order):
            if total + r["duration"] > cap_s or used >= train_hours * 3600:
                break
            train.append(r)
            total += r["duration"]
            used += r["duration"]
        if used >= train_hours * 3600:
            break
    return {"train": train, "val": val}


def fetch(rows, out_root):
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    import soxr
    from huggingface_hub import HfFileSystem

    fs, out_root, out = HfFileSystem(), Path(out_root), []
    groups = defaultdict(list)
    for r in rows:
        groups[(r["shard"], r["row_group"])].append(r)
    for n, ((shard, g), rs) in enumerate(sorted(groups.items())):
        f = pq.ParquetFile(fs.open(f"datasets/MushanW/GLOBE@{REV}/data/{shard}"))
        audio = f.read_row_group(g, columns=["audio"]).column("audio").to_pylist()
        for r in rs:
            wav, sr = sf.read(io.BytesIO(audio[r["row_in_group"]]["bytes"]), dtype="float32", always_2d=True)
            wav = wav.mean(axis=1)
            if sr != 24000:
                wav = soxr.resample(wav, sr, 24000, quality="HQ")
            rid = f"{r['speaker_id']}_{shard[6:11]}_{g:03d}_{r['row_in_group']:03d}"
            rel = Path("audio/globe") / r["speaker_id"] / f"{rid}.flac"
            (out_root / rel).parent.mkdir(parents=True, exist_ok=True)
            sf.write(out_root / rel, np.clip(wav, -1, 1), 24000, subtype="PCM_16")
            out.append({"id": rid, "audio_path": rel.as_posix(), "text_latin": r["transcript"],
                        "duration": len(wav) / 24000, "speaker_id": r["speaker_id"], "split": r["split"],
                        "lang": "en", "source": "globe", "licence": "CC0-1.0", "accent": r["accent"]})
        print(f"{n + 1}/{len(groups)} row groups", flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--train-hours", type=float, default=3.0)
    p.add_argument("--val-clips", type=int, default=150)
    a = p.parse_args(argv)
    sel = select(json.loads(Path(a.index).read_text()), train_hours=a.train_hours, val_clips=a.val_clips)
    rows = [dict(r, split=s) for s, rs in sel.items() for r in rs]
    out = fetch(rows, a.out)
    Path(a.out, "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print({s: (len(rs), len({r['speaker_id'] for r in rs}), round(sum(r["duration"] for r in rs) / 3600, 2))
           for s, rs in sel.items()})


if __name__ == "__main__":
    main()
