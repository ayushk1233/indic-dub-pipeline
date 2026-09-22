import copy
from pathlib import Path

import pytest
import torch

from train.config import ConfigError, config_hash, load_config
from train.model import BaseWeightMismatch, build_cfm, load_base, load_vocab, save_base

CONFIGS = Path("train/configs")


def test_vocab_has_space_at_zero_and_2545_entries(vocab):
    assert vocab[" "] == 0
    assert len(vocab) == 2545


def test_route_configs_carry_plan_values():
    a, b = load_config(CONFIGS / "route_a.yaml"), load_config(CONFIGS / "route_b.yaml")
    assert (a["lora"]["r"], a["lora"]["lr"], a["ema"]["decay"], a["ema"]["start_after"]) == (32, 1e-4, 0.999, 200)
    assert b["lora"]["r"] == 64 and [m["lr"] for m in b["full_modules"]] == [5e-5, 3e-5]
    for cfg, gpus in ((a, 2), (b, 2)):
        assert cfg["batch"]["frames_per_gpu"] * gpus * cfg["batch"]["grad_accum"] == 48000


def test_config_hash_ignores_runtime_but_not_training_values(tiny_cfg):
    h = config_hash(tiny_cfg)
    other = copy.deepcopy(tiny_cfg); other["runtime"]["guard_seconds"] = 3
    assert config_hash(other) == h
    other["lora"]["dropout"] = 0.1
    assert config_hash(other) != h


def test_config_missing_section_is_refused(tmp_path):
    (tmp_path / "bad.yaml").write_text("route: A\n")
    with pytest.raises(ConfigError, match="model"):
        load_config(tmp_path / "bad.yaml")


def test_tiny_model_has_plan_module_names(tiny_model):
    names = {n for n, _ in tiny_model.named_modules()}
    for n in ("transformer.transformer_blocks.1.attn.to_out.0", "transformer.transformer_blocks.0.ff.ff.0.0",
              "transformer.transformer_blocks.0.ff.ff.2", "transformer.text_embed.text_embed",
              "transformer.input_embed.proj", "transformer.norm_out", "transformer.proj_out"):
        assert n in names


def test_base_round_trip_and_hash_check(tiny_cfg, vocab, tiny_model, tmp_path):
    sha = save_base(tiny_model, tmp_path / "base.safetensors")
    torch.manual_seed(99)
    fresh = build_cfm(tiny_cfg, vocab)
    assert load_base(fresh, tmp_path / "base.safetensors", expected_sha=sha) == sha
    for (n, a), (_, b) in zip(tiny_model.state_dict().items(), fresh.state_dict().items()):
        assert torch.equal(a, b), n
    with pytest.raises(BaseWeightMismatch):
        load_base(fresh, tmp_path / "base.safetensors", expected_sha="0" * 64)
