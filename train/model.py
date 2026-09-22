"""Build the CFM/DiT from config and load the frozen base, hash-checked (FINETUNE_PLAN §7c)."""
from __future__ import annotations

import hashlib
from pathlib import Path

from safetensors.torch import load_file, save_file

from train.patches import VENDOR, import_vendor

VOCAB_PATH = VENDOR / "checkpoints" / "vocab.txt"


class BaseWeightMismatch(RuntimeError):
    pass


def load_vocab(path=VOCAB_PATH) -> dict[str, int]:
    vocab = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            vocab[line[:-1] if line.endswith("\n") else line] = i
    assert vocab[" "] == 0, "space must be index 0 (vendored convention)"
    return vocab


def build_cfm(cfg: dict, vocab: dict):
    v = import_vendor()
    m = cfg["model"]
    dit = v.DiT(dim=m["dim"], depth=m["depth"], heads=m["heads"], ff_mult=m["ff_mult"],
                text_dim=m["text_dim"], conv_layers=m["conv_layers"], text_num_embeds=len(vocab))
    c = cfg["cfm"]
    return v.CFM(transformer=dit, mel_spec_kwargs=dict(cfg["mel"]), vocab_char_map=vocab,
                 audio_drop_prob=c["audio_drop_prob"], cond_drop_prob=c["cond_drop_prob"],
                 frac_lengths_mask=tuple(c["frac_lengths_mask"]))


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_base(model, path) -> str:
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(path))
    return file_sha256(path)


def load_base(model, path, expected_sha: str | None = None) -> str:
    sha = file_sha256(path)
    if expected_sha is not None and sha != expected_sha:
        raise BaseWeightMismatch(f"{path}: sha256 {sha[:12]} != expected {expected_sha[:12]}")
    model.load_state_dict(load_file(str(Path(path))), strict=True)
    return sha
