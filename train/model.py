"""Build the CFM/DiT from config and load the frozen base, hash-checked (FINETUNE_PLAN §7c)."""
from __future__ import annotations

import hashlib
from pathlib import Path

from safetensors.torch import load_file, save_file

from train.patches import VENDOR, import_vendor

VOCAB_PATH = VENDOR / "checkpoints" / "vocab.txt"


class BaseWeightMismatch(RuntimeError):
    pass


RELEASE_PREFIX = "ema_model._orig_mod."       # the DiT/CFM in IndicF5's release model.safetensors
VOCODER_PREFIX = "vocoder._orig_mod."         # Vocos, shipped in the same file


def split_release(sd):
    """IndicF5's release model.safetensors -> (CFM state dict, Vocos state dict). Plain CFM dicts pass through."""
    if not any(k.startswith(RELEASE_PREFIX) for k in sd):
        return sd, {}
    cfm, voc, other = {}, {}, []
    for k, v in sd.items():
        if k.startswith(RELEASE_PREFIX):
            cfm[k[len(RELEASE_PREFIX):]] = v
        elif k.startswith(VOCODER_PREFIX):
            voc[k[len(VOCODER_PREFIX):]] = v
        else:
            other.append(k)
    if other:
        raise BaseWeightMismatch(f"release file has keys under no known prefix: {other[:3]}")
    return cfm, voc


def to_release(cfm_sd, vocoder_sd):
    return {**{RELEASE_PREFIX + k: v for k, v in cfm_sd.items()},
            **{VOCODER_PREFIX + k: v for k, v in vocoder_sd.items()}}


def expected_modules(cfg) -> list[str]:
    """FINETUNE_PLAN §0 block layout, for the configured depth and number of text blocks."""
    m, t = cfg["model"], "transformer."
    names = [t + "time_embed", t + "text_embed.text_embed", t + "input_embed.proj",
             t + "input_embed.conv_pos_embed", t + "norm_out", t + "proj_out"]
    names += [f"{t}text_embed.text_blocks.{i}" for i in range(m["conv_layers"])]
    for i in range(m["depth"]):
        b = f"{t}transformer_blocks.{i}."
        names += [b + s for s in ("attn_norm", "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
                                  "ff_norm", "ff.ff.0.0", "ff.ff.2")]
    return names


def layout_problems(model, cfg) -> list[str]:
    have = {n for n, _ in model.named_modules()}
    return [f"missing module {n}" for n in expected_modules(cfg) if n not in have]


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
    cfm_sd, _ = split_release(load_file(str(Path(path))))
    model.load_state_dict(cfm_sd, strict=True)
    return sha
