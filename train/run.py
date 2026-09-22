"""
CLI for a training session: build the model from config, load the hash-checked frozen base, attach
LoRA, resume from the store's LATEST if there is one, train until done or the time guard.

  python -m train.run --config train/configs/route_a.yaml --work-dir /kaggle/working/run \
      --store hf:<private repo> --base <IndicF5 weights> --base-sha <sha> \
      --data manifest:<dir> --val manifest:<dir>
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch

from train import lora
from train.config import load_config
from train.data import ManifestDataset, SyntheticDataset
from train.model import build_cfm, load_base, load_vocab
from train.patches import import_vendor
from train.store import HfStore, LocalStore
from train.trainer import Trainer


def _dataset(spec, vocab):
    kind, _, rest = spec.partition(":")
    if kind == "synthetic":
        parts = [int(x) for x in rest.split(":")]
        n, seed = parts[:2]
        frames = parts[2:3]
        return SyntheticDataset(n, seed, vocab, *(frames * 2)) if frames else SyntheticDataset(n, seed, vocab)
    if kind == "manifest":
        return ManifestDataset(rest)
    raise SystemExit(f"unknown dataset spec {spec!r}")


def _store(spec):
    kind, _, rest = spec.partition(":")
    if kind == "local":
        return LocalStore(rest)
    if kind == "hf":
        return HfStore(rest, token=os.environ["HF_TOKEN"])
    raise SystemExit(f"unknown store spec {spec!r}")


def _die_hook(spec):
    if not spec:
        return None
    target = tuple(int(x) for x in spec.split(":"))

    def hook(update, micro):
        if (update, micro) == target:
            os._exit(137)                       # no cleanup, no flush: as kill -9
    return hook


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m train.run")
    for flag in ("--config", "--work-dir", "--store", "--base", "--data", "--val"):
        p.add_argument(flag, required=True)
    p.add_argument("--base-sha")
    p.add_argument("--guard-seconds", type=float)
    p.add_argument("--die-at")
    a = p.parse_args(argv)

    cfg = load_config(a.config)
    if a.guard_seconds is not None:
        cfg["runtime"]["guard_seconds"] = a.guard_seconds
    if cfg["runtime"].get("deterministic"):
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
    seed = cfg["seed"] + int(os.environ.get("RANK", 0))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    import_vendor()
    vocab = load_vocab()
    model = build_cfm(cfg, vocab)
    base_sha = load_base(model, a.base, expected_sha=a.base_sha)
    model = lora.attach(model, cfg)
    train_set, val_set = _dataset(a.data, vocab), _dataset(a.val, vocab)
    trainer = Trainer(cfg, model=model, train_set=train_set, val_items=val_set.val_items(), store=_store(a.store),
                      work_dir=a.work_dir, base_sha=base_sha, manifest_sha=train_set.manifest_sha256(),
                      on_micro=_die_hook(a.die_at))
    reason = trainer.run()
    print(f"stopped: {reason}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
