"""Training configuration: plain dicts from YAML (repo convention), with a hash for resume refusal."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

REQUIRED = ("route", "model", "mel", "cfm", "lora", "full_modules", "optim", "batch", "schedule",
            "ema", "precision", "seed", "checkpoint", "validation", "runtime")
HASH_EXCLUDED = ("runtime",)   # time guard and determinism flags may change between sessions


class ConfigError(ValueError):
    pass


def load_config(path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ConfigError(f"{path}: missing sections {missing}")
    if cfg["route"] not in ("A", "B"):
        raise ConfigError(f"{path}: route must be A or B, got {cfg['route']!r}")
    return cfg


def config_hash(cfg: dict) -> str:
    kept = {k: v for k, v in cfg.items() if k not in HASH_EXCLUDED}
    return hashlib.sha256(json.dumps(kept, sort_keys=True).encode()).hexdigest()
