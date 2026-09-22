import copy
import hashlib

import pytest
import torch

from train import lora
from train.config import load_config
from train.model import build_cfm


def _digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _loss(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    mel = torch.randn(2, 60, 100, generator=g)
    return model(mel, text=["कखग", "गघ"], lens=torch.tensor([60, 50]))[0]


@pytest.mark.c5
def test_trainable_count_matches_r_times_in_plus_out(tiny_cfg, tiny_model):
    expected = lora.expected_lora_params(tiny_model, tiny_cfg)
    model = lora.attach(tiny_model, tiny_cfg)
    assert expected == 2 * (4 * 4 * (128 + 128) + 4 * (128 + 256) + 4 * (256 + 128))
    assert lora.trainable_count(model) == expected


@pytest.mark.c5
@pytest.mark.slow
def test_full_size_route_a_trainable_count_matches_plan_rows(vocab):
    cfg = load_config("train/configs/route_a.yaml")
    model = lora.attach(build_cfm(cfg, vocab), cfg)
    plan_rows = 22 * 4 * 32 * 2048 + 22 * 2 * 32 * 3072          # §4 table rows (Decision 1)
    assert abs(lora.trainable_count(model) - plan_rows) / plan_rows < 0.01


@pytest.mark.c5
def test_step0_output_equals_base_output(tiny_cfg, tiny_model):
    base = copy.deepcopy(tiny_model).eval()
    model = lora.attach(tiny_model, tiny_cfg).eval()
    g = torch.Generator().manual_seed(3)
    x, cond = torch.randn(2, 50, 100, generator=g), torch.randn(2, 50, 100, generator=g)
    text, time = torch.randint(0, 50, (2, 20), generator=g), torch.rand(2, generator=g)
    kw = dict(x=x, cond=cond, text=text, time=time, drop_audio_cond=False, drop_text=False)
    with torch.no_grad():
        diff = (model.transformer(**kw) - base.transformer(**kw)).abs().max().item()
    assert diff < 1e-6


@pytest.mark.c5
def test_after_20_updates_frozen_unchanged_and_every_trainable_changed(tiny_cfg, tiny_model):
    model = lora.attach(tiny_model, tiny_cfg)
    before = {n: _digest(p) for n, p in model.named_parameters()}
    opt = torch.optim.AdamW(lora.param_groups(model, tiny_cfg))
    for step in range(20):
        opt.zero_grad()
        _loss(model, step).backward()
        opt.step()
    for n, p in model.named_parameters():
        changed = _digest(p) != before[n]
        assert changed == p.requires_grad, f"{n}: requires_grad={p.requires_grad}, changed={changed}"


def test_route_b_unfreezes_text_and_input_embed_with_their_lrs(tiny_cfg, tiny_model):
    cfg = copy.deepcopy(tiny_cfg)
    cfg["full_modules"] = [{"prefix": "transformer.text_embed", "lr": 5e-5},
                           {"prefix": "transformer.input_embed", "lr": 3e-5}]
    model = lora.attach(tiny_model, cfg)
    for n, p in model.named_parameters():
        if n.startswith(("transformer.text_embed.", "transformer.input_embed.")):
            assert p.requires_grad, n
    groups = {g["name"]: g for g in lora.param_groups(model, cfg)}
    assert groups["transformer.text_embed"]["lr"] == 5e-5 and groups["transformer.text_embed"]["weight_decay"] == 0.01
    assert groups["lora"]["weight_decay"] == 0.0


def test_param_groups_cover_each_trainable_param_exactly_once(tiny_cfg, tiny_model):
    model = lora.attach(tiny_model, tiny_cfg)
    names = [n for g in lora.param_groups(model, tiny_cfg) for n in g["param_names"]]
    assert sorted(names) == sorted(n for n, p in model.named_parameters() if p.requires_grad)


def test_trainable_state_round_trip_is_strict(tiny_cfg, tiny_model):
    model = lora.attach(tiny_model, tiny_cfg)
    sd = {k: torch.randn_like(v) for k, v in lora.trainable_state_dict(model).items()}
    lora.load_trainable(model, sd)
    assert all(torch.equal(v, lora.trainable_state_dict(model)[k]) for k, v in sd.items())
    with pytest.raises(KeyError):
        lora.load_trainable(model, {**sd, "transformer.extra": torch.zeros(1)})
