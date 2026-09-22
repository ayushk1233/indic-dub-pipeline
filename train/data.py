"""
Training datasets. ManifestDataset wraps the patched vendored CustomDataset over raw/ + duration.json
(FINETUNE_PLAN §2g). SyntheticDataset is the deterministic stand-in the CPU checks train on (§8a).
Both refuse a transcript longer than its mel: the DiT would silently cut it (§0).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from train.patches import import_vendor


class TextTooLongError(ValueError):
    pass


def check_fits(text: str, frames: int) -> None:
    if len(text) > frames:
        raise TextTooLongError(f"transcript has {len(text)} characters but the clip has {frames} mel "
                               "frames; the DiT would silently cut it (FINETUNE_PLAN §0)")


class SyntheticDataset(Dataset):
    def __init__(self, n, seed, vocab, min_frames=40, max_frames=160):
        g = torch.Generator().manual_seed(seed)
        self.seed = seed
        self.frames = torch.randint(min_frames, max_frames + 1, (n,), generator=g).tolist()
        letters = sorted(c for c in vocab if "अ" <= c <= "ह")
        assert letters, "vocab has no Devanagari letters"
        self.texts = []
        for f in self.frames:
            k = int(torch.randint(max(1, f // 8), max(2, f // 4), (1,), generator=g))
            self.texts.append("".join(letters[i] for i in torch.randint(0, len(letters), (k,), generator=g).tolist()))
        self.poison = False

    def __len__(self):
        return len(self.frames)

    def get_frame_len(self, i):
        return self.frames[i]

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + i)
        mel = torch.randn(100, self.frames[i], generator=g)
        if self.poison:
            mel = mel * float("nan")
        check_fits(self.texts[i], self.frames[i])
        return {"mel_spec": mel, "text": self.texts[i]}

    def manifest_sha256(self):
        rows = [{"frames": f, "text": t} for f, t in zip(self.frames, self.texts)]
        return hashlib.sha256(json.dumps([self.seed, rows], ensure_ascii=False).encode()).hexdigest()

    def val_items(self, groups=("hi", "en")):
        return [{"mel": self[i]["mel_spec"], "text": self.texts[i], "group": groups[i % len(groups)]}
                for i in range(len(self))]


def write_manifest_dir(rows, out_dir, groups=None):
    from datasets import Dataset as HFDataset

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    HFDataset.from_list(rows).save_to_disk(str(out / "raw"))
    (out / "duration.json").write_text(json.dumps({"duration": [r["duration"] for r in rows]}))
    if groups is not None:
        (out / "groups.json").write_text(json.dumps(groups))


class ManifestDataset(Dataset):
    def __init__(self, data_dir):
        from datasets import load_from_disk

        self.dir = Path(data_dir)
        rows = load_from_disk(str(self.dir / "raw"))
        durations = json.loads((self.dir / "duration.json").read_text())["duration"]
        if len(durations) != len(rows):
            raise ValueError(f"{self.dir}: {len(rows)} rows but {len(durations)} durations")
        self._inner = import_vendor().CustomDataset(rows, durations=durations)
        groups = self.dir / "groups.json"
        self.groups = json.loads(groups.read_text()) if groups.exists() else ["all"] * len(rows)

    def __len__(self):
        return len(self._inner)

    def get_frame_len(self, i):
        return self._inner.get_frame_len(i)

    def __getitem__(self, i):
        item = self._inner[i]
        check_fits(item["text"], item["mel_spec"].shape[-1])
        return item

    def manifest_sha256(self):
        h = hashlib.sha256()
        files = [self.dir / "duration.json", *sorted((self.dir / "raw").rglob("*"))]
        for p in files:
            if p.is_file():
                h.update(p.relative_to(self.dir).as_posix().encode())
                h.update(p.read_bytes())
        return h.hexdigest()

    def val_items(self):
        return [{"mel": self[i]["mel_spec"], "text": self[i]["text"], "group": self.groups[i]}
                for i in range(len(self))]
