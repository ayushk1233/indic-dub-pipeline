"""
Merge LoRA into the base on CPU in fp32 (FINETUNE_PLAN §12 step 1). Keys come out in the vendored
CFM layout; mapping them to the IndicF5 release layout is G2's documented mapping, inverted, and
is done on GPU pre-flight, not here.
"""
from __future__ import annotations

import copy

import torch
from peft.tuners.lora import LoraLayer
from safetensors.torch import save_file

from train.model import file_sha256


def merged_state_dict(model) -> dict[str, torch.Tensor]:
    m = copy.deepcopy(model).float().cpu()
    for module in m.modules():
        if isinstance(module, LoraLayer):
            module.merge(safe_merge=True)
    out = {}
    for k, v in m.state_dict().items():
        if ".lora_" in k:
            continue
        out[k.replace(".base_layer", "")] = v.detach().clone().contiguous()
    return out


def export(model, path) -> str:
    save_file(merged_state_dict(model), str(path))
    return file_sha256(path)
