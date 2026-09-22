"""JSON-serialisable RNG state for Python, NumPy and Torch (FINETUNE_PLAN §7c: per rank, in state.json)."""
from __future__ import annotations

import base64
import random

import numpy as np
import torch


def _b64(t: torch.Tensor) -> str:
    return base64.b64encode(t.cpu().numpy().tobytes()).decode()


def _tensor(s: str) -> torch.Tensor:
    return torch.frombuffer(bytearray(base64.b64decode(s)), dtype=torch.uint8).clone()


def capture_rng() -> dict:
    version, internal, gauss = random.getstate()
    name, keys, pos, has_gauss, cached = np.random.get_state()
    out = {"python": [version, list(internal), gauss],
           "numpy": [name, keys.tolist(), int(pos), int(has_gauss), float(cached)],
           "torch": _b64(torch.get_rng_state())}
    if torch.cuda.is_available():
        out["cuda"] = [_b64(s) for s in torch.cuda.get_rng_state_all()]
    return out


def restore_rng(d: dict) -> None:
    version, internal, gauss = d["python"]
    random.setstate((version, tuple(internal), gauss))
    name, keys, pos, has_gauss, cached = d["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(_tensor(d["torch"]))
    if "cuda" in d and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_tensor(s) for s in d["cuda"]])
