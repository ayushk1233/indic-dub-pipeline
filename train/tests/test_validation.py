import pytest
import torch

from train.data import SyntheticDataset
from train.patches import strict_list_str_to_idx
from train.validation import FixedNoiseValidator, flow_loss


def test_flow_loss_equals_cfm_forward_under_the_same_draws(tiny_model, vocab):
    model = tiny_model.eval()
    model.audio_drop_prob, model.cond_drop_prob = 0.0, 0.0
    g = torch.Generator().manual_seed(0)
    mel, lens, text = torch.randn(2, 60, 100, generator=g), torch.tensor([60, 45]), ["कखग", "गघ"]
    torch.manual_seed(11)
    with torch.no_grad():
        reference = model(mel, text=text, lens=lens)[0]
    torch.manual_seed(11)                     # same draw order as CFM.forward: frac, start, x0, time
    frac = torch.zeros(2).float().uniform_(0.7, 1.0)
    start_rand = torch.rand_like(frac)
    x0 = torch.randn_like(mel)
    time = torch.rand(2)
    with torch.no_grad():
        ours = flow_loss(model, mel, strict_list_str_to_idx(text, vocab), lens, x0, time, frac, start_rand)
    assert torch.allclose(ours, reference, atol=1e-6)


@pytest.mark.c11
def test_validation_loss_is_identical_across_runs_and_leaves_global_rng_alone(tiny_model, vocab):
    items = SyntheticDataset(6, 5, vocab).val_items()
    v = FixedNoiseValidator(items, vocab, seed=4321)
    tiny_model.train()
    rng = torch.get_rng_state()
    first, second = v.evaluate(tiny_model), v.evaluate(tiny_model)
    assert first == second
    assert torch.equal(rng, torch.get_rng_state())
    assert tiny_model.training
    assert set(first) == {"hi", "en", "all"}


@pytest.mark.c11
def test_a_second_validator_with_the_same_seed_agrees(tiny_model, vocab):
    items = SyntheticDataset(4, 5, vocab).val_items()
    assert FixedNoiseValidator(items, vocab, 1).evaluate(tiny_model) == FixedNoiseValidator(items, vocab, 1).evaluate(tiny_model)
