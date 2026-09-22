import numpy as np
import pytest
import soundfile as sf
import torch

from train.data import ManifestDataset, SyntheticDataset, TextTooLongError, check_fits, write_manifest_dir


def test_synthetic_items_are_deterministic_and_fit(vocab):
    a, b = SyntheticDataset(12, 7, vocab), SyntheticDataset(12, 7, vocab)
    for i in range(12):
        assert torch.equal(a[i]["mel_spec"], b[i]["mel_spec"]) and a[i]["text"] == b[i]["text"]
        assert a[i]["mel_spec"].shape == (100, a.get_frame_len(i))
        assert 0 < len(a[i]["text"]) <= a.get_frame_len(i)
        assert all(c in vocab for c in a[i]["text"])
    assert a.manifest_sha256() == b.manifest_sha256() != SyntheticDataset(12, 8, vocab).manifest_sha256()


def test_synthetic_poison_makes_nan(vocab):
    ds = SyntheticDataset(2, 1, vocab)
    ds.poison = True
    assert torch.isnan(ds[0]["mel_spec"]).all()


def test_text_longer_than_mel_raises():
    with pytest.raises(TextTooLongError, match="12 characters"):
        check_fits("क" * 12, 10)


def _manifest(tmp_path, texts):
    rows = []
    for i, t in enumerate(texts):
        p = tmp_path / "wav" / f"{i}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, (0.1 * np.sin(np.arange(24000) / 15)).astype("float32"), 24000)
        rows.append({"audio_path": str(p), "text": t, "duration": 1.0})
    write_manifest_dir(rows, tmp_path / "m", groups=["hi", "en"][: len(rows)])
    return tmp_path / "m"


def test_manifest_dataset_loads_and_hashes(tmp_path, vocab):
    d = _manifest(tmp_path, ["नम", "सत"])
    ds = ManifestDataset(d)
    assert len(ds) == 2 and ds.groups == ["hi", "en"]
    assert ds[1]["text"] == "सत" and ds[1]["mel_spec"].shape[0] == 100
    assert abs(ds.get_frame_len(0) - 24000 / 256) < 1e-6
    h = ds.manifest_sha256()
    assert h == ManifestDataset(d).manifest_sha256()
    d2 = _manifest(tmp_path / "other", ["नम", "सम"])       # one row's text differs
    assert ManifestDataset(d2).manifest_sha256() != h


def test_manifest_item_longer_text_than_mel_raises(tmp_path):
    ds = ManifestDataset(_manifest(tmp_path, ["न" * 200]))
    with pytest.raises(TextTooLongError):
        ds[0]
