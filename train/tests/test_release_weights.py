from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from train.model import BaseWeightMismatch, build_cfm, layout_problems, load_base, split_release, to_release

REAL = Path("finetune_data/indicf5-ba85abedf18d/model.safetensors")
REAL_SHA = "ba7f3671180fb7784e24bd1dafc96e729a38ce02e7f6d3877cdef32525a1865c"


def _release_file(model, path, extra=None):
    sd = to_release({k: v.contiguous() for k, v in model.state_dict().items()},
                    {"backbone.norm.weight": torch.ones(4)})
    sd.update(extra or {})
    save_file(sd, str(path))


def test_release_file_loads_strictly(tiny_cfg, vocab, tiny_model, tmp_path):
    _release_file(tiny_model, tmp_path / "rel.safetensors")
    torch.manual_seed(1)
    fresh = build_cfm(tiny_cfg, vocab)
    load_base(fresh, tmp_path / "rel.safetensors")
    for (n, a), (_, b) in zip(tiny_model.state_dict().items(), fresh.state_dict().items()):
        assert torch.equal(a, b), n


def test_split_release_separates_vocoder(tiny_model):
    cfm, voc = split_release(to_release(tiny_model.state_dict(), {"head.out.bias": torch.zeros(2)}))
    assert set(cfm) == set(tiny_model.state_dict()) and set(voc) == {"head.out.bias"}


def test_plain_state_dict_passes_through(tiny_model):
    sd = tiny_model.state_dict()
    assert split_release(sd) == (sd, {})


def test_release_file_with_unknown_prefix_refuses(tiny_model, tmp_path):
    _release_file(tiny_model, tmp_path / "rel.safetensors", extra={"model._orig_mod.x": torch.zeros(1)})
    with pytest.raises(BaseWeightMismatch, match="model._orig_mod.x"):
        load_base(tiny_model, tmp_path / "rel.safetensors")


def test_layout_matches_plan_block_layout(tiny_cfg, tiny_model):
    assert layout_problems(tiny_model, tiny_cfg) == []


def test_layout_reports_a_missing_module(tiny_cfg, tiny_model):
    del tiny_model.transformer.transformer_blocks[0].attn.to_q
    assert any("transformer_blocks.0.attn.to_q" in p for p in layout_problems(tiny_model, tiny_cfg))


@pytest.mark.slow
@pytest.mark.skipif(not REAL.exists(), reason="IndicF5 release weights not downloaded")
def test_real_release_g2(vocab):
    from train.config import load_config

    cfg = load_config("train/configs/route_a.yaml")
    model = build_cfm(cfg, vocab)
    assert load_base(model, REAL, expected_sha=REAL_SHA) == REAL_SHA
    assert layout_problems(model, cfg) == []
    assert sum(p.numel() for p in model.parameters()) == 337_096_804
    assert not any(p.is_meta for p in model.parameters())
