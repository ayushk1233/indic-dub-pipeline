"""Freezing, LoRA injection and named parameter groups (FINETUNE_PLAN §4)."""
from __future__ import annotations

import re

import torch
from peft import LoraConfig, inject_adapter_in_model
from torch import nn


def target_regex(targets) -> str:
    return r".*transformer_blocks\.\d+\.(" + "|".join(re.escape(t) for t in targets) + r")$"


def expected_lora_params(model, cfg) -> int:
    pattern, r = re.compile(target_regex(cfg["lora"]["targets"])), cfg["lora"]["r"]
    return sum(r * (m.in_features + m.out_features)
               for n, m in model.named_modules() if isinstance(m, nn.Linear) and pattern.fullmatch(n))


def attach(model, cfg):
    for p in model.parameters():
        p.requires_grad_(False)                          # freeze before anything else
    lc = cfg["lora"]
    model = inject_adapter_in_model(LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                               target_modules=target_regex(lc["targets"])), model)
    for p in model.parameters():                         # inject marks only lora_* trainable; be explicit
        p.requires_grad_(False)
    for n, p in model.named_parameters():
        if ".lora_" in n or any(n.startswith(m["prefix"] + ".") for m in cfg["full_modules"]):
            p.requires_grad_(True)
    return model


def param_groups(model, cfg) -> list[dict]:
    groups = {"lora": {"name": "lora", "params": [], "param_names": [], "lr": cfg["lora"]["lr"], "weight_decay": 0.0}}
    for m in cfg["full_modules"]:
        groups[m["prefix"]] = {"name": m["prefix"], "params": [], "param_names": [], "lr": m["lr"],
                               "weight_decay": cfg["optim"]["weight_decay_full"]}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = "lora" if ".lora_" in n else next(m["prefix"] for m in cfg["full_modules"] if n.startswith(m["prefix"] + "."))
        groups[key]["params"].append(p)
        groups[key]["param_names"].append(n)
    return [g for g in groups.values() if g["params"]]


def trainable_count(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def trainable_state_dict(model) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone().contiguous() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable(model, sd) -> None:
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(sd) != set(params):
        raise KeyError(f"trainable names differ: missing {sorted(set(params) - set(sd))[:3]}, "
                       f"unexpected {sorted(set(sd) - set(params))[:3]}")
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(sd[n])
