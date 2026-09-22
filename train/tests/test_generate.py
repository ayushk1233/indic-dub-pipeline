import numpy as np
import pytest
import torch

from train.generate import generate


class MeanVocoder:                      # deterministic stand-in: one sample per frame
    def decode(self, mel):              # mel [b, d, n]
        return mel.mean(dim=1)


def _official(model, audio, ref_text, gen_text, seed, fix_duration):
    pytest.importorskip("matplotlib", reason="vendored utils_infer imports matplotlib (Decision 10)")
    from f5_tts.infer.utils_infer import infer_batch_process

    torch.manual_seed(seed)
    wave, sr, spec = infer_batch_process((audio, 24000), ref_text, [gen_text], model, MeanVocoder(),
                                         mel_spec_type="vocos", fix_duration=fix_duration, device="cpu", nfe_step=4)
    return wave, spec


@pytest.mark.parametrize("gain", [1.0, 0.01])          # 0.01: quieter than target_rms, scaled up then back
def test_generate_matches_infer_batch_process(tiny_model, gain):
    torch.manual_seed(0)
    audio = torch.randn(1, 24000) * 0.3 * gain
    ref, gen = "नमस्ते दोस्त", "आज मौसम अच्छा है"
    wave, spec = _official(tiny_model, audio, ref, gen, seed=7, fix_duration=2.5)
    ours = generate(tiny_model, audio, ref, gen, seed=7, fix_duration=2.5, vocoder=MeanVocoder(), nfe=4)
    np.testing.assert_array_equal(ours["mel"].numpy(), spec)
    np.testing.assert_array_equal(ours["wave"], wave)


def test_generate_matches_infer_batch_process_quiet_reference(tiny_model):
    audio = torch.full((1, 12000), 1e-3)
    wave, spec = _official(tiny_model, audio, "ab1", "cd", seed=3, fix_duration=None)   # ASCII tail -> trailing space
    ours = generate(tiny_model, audio, "ab1", "cd", seed=3, vocoder=MeanVocoder(), nfe=4)
    np.testing.assert_array_equal(ours["mel"].numpy(), spec)
    np.testing.assert_array_equal(ours["wave"], wave)


def test_vocoder_loads_release_tensors(tmp_path):
    from safetensors.torch import save_file

    from train.model import to_release
    from train.vocoder import build_vocos, load_vocoder

    v = build_vocos()
    save_file(to_release({}, {k: t.contiguous() for k, t in v.state_dict().items()}), str(tmp_path / "r.safetensors"))
    loaded = load_vocoder(tmp_path / "r.safetensors")
    assert all(torch.equal(a, b) for a, b in zip(v.state_dict().values(), loaded.state_dict().values()))
