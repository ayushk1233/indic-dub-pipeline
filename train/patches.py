"""
Load the vendored IndicF5 code unmodified and apply FINETUNE_PLAN §7a's two patches.

The vendored `f5_tts/model/__init__.py` imports its own trainer, which imports wandb. We fork that
trainer (§7) and never install wandb, so the package is registered bare and its submodules are
imported directly.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence

VENDOR = Path(__file__).resolve().parent / "vendor" / "indicf5"
DURATION_RANGE = (0.3, 30.0)
_NS: types.SimpleNamespace | None = None


class UnknownCharacterError(ValueError):
    """A transcript character is not in IndicF5's vocabulary (vendored code speaks it as a pause)."""


def strict_list_str_to_idx(text, vocab_char_map, padding_value=-1):
    rows = []
    for t in text:
        unknown = sorted({c for c in t if c not in vocab_char_map})
        if unknown:
            raise UnknownCharacterError(f"not in vocab: {unknown!r} in {''.join(t)[:60]!r}")
        rows.append(torch.tensor([vocab_char_map[c] for c in t], dtype=torch.long))
    return pad_sequence(rows, padding_value=padding_value, batch_first=True)


def strict_getitem(self, index):
    row = self.data[index]
    low, high = DURATION_RANGE
    if not low <= row["duration"] <= high:
        raise ValueError(f"row {index}: duration {row['duration']} s outside {low}-{high} s; "
                         "filter the manifest instead of skipping at load time")
    if self.preprocessed_mel:
        mel = torch.tensor(row["mel_spec"])
    else:
        samples, sr = sf.read(row["audio_path"], dtype="float32", always_2d=True)
        if sr != self.target_sample_rate:
            raise ValueError(f"{row['audio_path']}: {sr} Hz; pre-resample to "
                             f"{self.target_sample_rate} Hz (FINETUNE_PLAN §0)")
        audio = torch.from_numpy(samples.T).mean(dim=0, keepdim=True)
        mel = self.mel_spectrogram(audio).squeeze(0)
    return {"mel_spec": mel, "text": row["text"]}


def import_vendor() -> types.SimpleNamespace:
    global _NS
    if _NS is not None:
        return _NS
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    if "f5_tts.model" not in sys.modules:
        import f5_tts  # noqa: F401  (empty package __init__)
        package = types.ModuleType("f5_tts.model")
        package.__path__ = [str(VENDOR / "f5_tts" / "model")]
        sys.modules["f5_tts.model"] = package
    from f5_tts.model import cfm, dataset, modules, utils
    from f5_tts.model.backbones import dit

    utils.list_str_to_idx = strict_list_str_to_idx
    cfm.list_str_to_idx = strict_list_str_to_idx          # cfm imported the name directly
    dataset.CustomDataset.__getitem__ = strict_getitem
    _NS = types.SimpleNamespace(CFM=cfm.CFM, DiT=dit.DiT, MelSpec=modules.MelSpec,
                                CustomDataset=dataset.CustomDataset, collate_fn=dataset.collate_fn,
                                utils=utils, cfm=cfm, dataset=dataset)
    return _NS
