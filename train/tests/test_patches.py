import numpy as np
import pytest
import soundfile as sf
import torch

from train.patches import UnknownCharacterError, strict_list_str_to_idx

VOCAB = {" ": 0, "न": 1, "म": 2, "स": 3, "त": 4, "े": 5}


def test_known_text_maps_like_the_vendored_lookup(vendor):
    text = ["नमस्ते"[:2], "स त"]
    ours = strict_list_str_to_idx(text, VOCAB)
    theirs = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor([VOCAB.get(c, 0) for c in t]) for t in text], padding_value=-1, batch_first=True)
    assert torch.equal(ours, theirs)


def test_unknown_character_raises_instead_of_becoming_space():
    with pytest.raises(UnknownCharacterError, match="☃"):
        strict_list_str_to_idx(["न☃म"], VOCAB)


@pytest.mark.c1
def test_cfm_forward_uses_the_strict_lookup(vendor):
    vocab = dict(VOCAB)
    tiny = vendor.DiT(dim=64, depth=1, heads=1, ff_mult=2, text_dim=32, conv_layers=1,
                      text_num_embeds=len(vocab))
    model = vendor.CFM(transformer=tiny, vocab_char_map=vocab, mel_spec_kwargs=dict(
        n_fft=1024, hop_length=256, win_length=1024, n_mel_channels=100,
        target_sample_rate=24000, mel_spec_type="vocos"))
    with pytest.raises(UnknownCharacterError):
        model(torch.randn(1, 40, 100), text=["न☃"], lens=torch.tensor([40]))


def _wav(path, seconds, sr=24000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, (0.1 * np.sin(np.arange(int(seconds * sr)) / 20)).astype("float32"), sr)
    return str(path)


def _dataset(vendor, rows):
    return vendor.CustomDataset(rows, durations=[r["duration"] for r in rows])


@pytest.mark.c2
def test_dataset_reads_the_path_it_is_given(vendor, tmp_path):
    path = _wav(tmp_path / "indictts" / "wavs-24k" / "a.wav", 1.0)   # the vendored code rewrites this
    item = _dataset(vendor, [{"audio_path": path, "text": "नम", "duration": 1.0}])[0]
    assert item["text"] == "नम"
    assert item["mel_spec"].shape[0] == 100
    assert abs(item["mel_spec"].shape[1] - 1.0 * 24000 / 256) <= 2


@pytest.mark.c2
def test_dataset_raises_on_out_of_range_duration_instead_of_skipping(vendor, tmp_path):
    rows = [{"audio_path": _wav(tmp_path / "long.wav", 1.0), "text": "न", "duration": 45.0},
            {"audio_path": _wav(tmp_path / "ok.wav", 1.0), "text": "म", "duration": 1.0}]
    with pytest.raises(ValueError, match="45"):
        _dataset(vendor, rows)[0]


def test_dataset_refuses_to_resample(vendor, tmp_path):
    path = _wav(tmp_path / "a.wav", 1.0, sr=16000)
    with pytest.raises(ValueError, match="16000"):
        _dataset(vendor, [{"audio_path": path, "text": "न", "duration": 1.0}])[0]


def test_import_vendor_is_idempotent(vendor):
    from train.patches import import_vendor
    assert import_vendor().CFM is vendor.CFM
