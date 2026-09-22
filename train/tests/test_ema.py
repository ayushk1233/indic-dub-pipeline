import pytest
import torch

from train import lora
from train.ema import TrainableEMA


def _bump(model, value):
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.fill_(value)


def test_tracks_trainable_parameters_only(tiny_cfg, tiny_model):
    model = lora.attach(tiny_model, tiny_cfg)
    ema = TrainableEMA(model, 0.9, start_after=2)
    assert set(ema.shadow_state()) == {n for n, p in model.named_parameters() if p.requires_grad}


def test_copies_before_start_then_decays(tiny_cfg, tiny_model):
    model = lora.attach(tiny_model, tiny_cfg)
    ema = TrainableEMA(model, 0.9, start_after=2)
    _bump(model, 1.0); ema.update(model, 1)
    assert all(torch.all(v == 1.0) for v in ema.shadow_state().values())
    _bump(model, 2.0); ema.update(model, 3)
    assert all(torch.allclose(v, torch.full_like(v, 0.9 * 1.0 + 0.1 * 2.0)) for v in ema.shadow_state().values())


def test_swapped_restores_originals_exactly(tiny_cfg, tiny_model):
    model = lora.attach(tiny_model, tiny_cfg)
    ema = TrainableEMA(model, 0.9, start_after=1)
    _bump(model, 5.0); ema.update(model, 0)
    _bump(model, 7.0)
    with ema.swapped(model):
        assert all(torch.all(p == 5.0) for p in model.parameters() if p.requires_grad)
    assert all(torch.all(p == 7.0) for p in model.parameters() if p.requires_grad)


def test_shadow_round_trip_is_strict(tiny_cfg, tiny_model):
    model = lora.attach(tiny_model, tiny_cfg)
    ema = TrainableEMA(model, 0.9, start_after=0)
    sd = {k: torch.randn_like(v) for k, v in ema.shadow_state().items()}
    ema.load_shadow(sd)
    assert all(torch.equal(sd[k], v) for k, v in ema.shadow_state().items())
    with pytest.raises(KeyError):
        ema.load_shadow({})
