import copy

import pytest
import torch

from train import lora
from train.export import export, merged_state_dict
from train.model import build_cfm, file_sha256


def _trained(tiny_cfg, vocab, full_modules=()):
    cfg = copy.deepcopy(tiny_cfg)
    cfg["full_modules"] = list(full_modules)
    torch.manual_seed(0)
    model = lora.attach(build_cfm(cfg, vocab), cfg)
    opt = torch.optim.AdamW(lora.param_groups(model, cfg), lr=1e-2)
    for step in range(5):
        g = torch.Generator().manual_seed(step)
        opt.zero_grad()
        model(torch.randn(2, 50, 100, generator=g), text=["कख", "ग"], lens=torch.tensor([50, 40]))[0].backward()
        opt.step()
    return cfg, model.eval()


def _out(model):
    g = torch.Generator().manual_seed(1)
    kw = dict(x=torch.randn(1, 40, 100, generator=g), cond=torch.randn(1, 40, 100, generator=g),
              text=torch.randint(0, 40, (1, 10), generator=g), time=torch.rand(1, generator=g),
              drop_audio_cond=False, drop_text=False)
    with torch.no_grad():
        return model.transformer(**kw)


@pytest.mark.c10
@pytest.mark.parametrize("full", [(), ({"prefix": "transformer.text_embed", "lr": 5e-5},)])
def test_merged_model_matches_unmerged(tiny_cfg, vocab, full):
    cfg, model = _trained(tiny_cfg, vocab, full)
    plain = build_cfm(cfg, vocab).eval()
    plain.load_state_dict(merged_state_dict(model), strict=True)
    assert (_out(plain) - _out(model)).abs().max().item() < 1e-5


def test_export_writes_fp32_and_leaves_the_trained_model_unmerged(tiny_cfg, vocab, tmp_path):
    cfg, model = _trained(tiny_cfg, vocab)
    before = _out(model)
    sha = export(model, tmp_path / "merged.safetensors")
    assert sha == file_sha256(tmp_path / "merged.safetensors")
    assert torch.equal(before, _out(model))
    from safetensors.torch import load_file
    assert all(t.dtype == torch.float32 for t in load_file(str(tmp_path / "merged.safetensors")).values()
               if t.is_floating_point())
