import pickle

import numpy as np
import soundfile as sf

from prep.build_manifests import build
from train.data import ManifestDataset

GOOD = {"latin": "hello there my friend how are you", "deva": "हेलो देयर माय फ्रेंड हाउ आर यू"}


def _row(tmp, i, secs=3.0, split="train", spk="S1"):
    p = tmp / "audio" / f"{i}.flac"
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(p, (np.random.default_rng(i).standard_normal(int(secs * 24000)) * 0.1).astype("float32"), 24000, subtype="PCM_16")
    return {"id": str(i), "audio_path": f"audio/{i}.flac", "duration": secs, "speaker_id": spk, "split": split,
            "lang": "en", "source": "globe", "licence": "CC0-1.0", "accent": "India", "text_latin": "x"}


def test_gates_reject_unknown_chars_wrong_script_and_cps(tmp_path, vocab):
    rows = [_row(tmp_path, i) for i in range(5)]
    xlit = {"0": GOOD,
            "1": dict(GOOD, deva="हेलो ☃ देयर"),                   # not in vocab
            "2": dict(GOOD, deva="hello देयर"),                    # < 0.95 Devanagari
            "3": dict(GOOD, latin="hello देयर my friend"),        # < 0.95 Latin
            "4": {"latin": "hi", "deva": "हाय"}}                   # 0.5 cps < 7
    counts = build(rows, xlit, vocab, tmp_path / "out")
    assert counts == {"kept": 1, "vocab": 1, "devanagari": 1, "latin": 1, "cps": 1}


def test_both_routes_hold_the_same_clips(tmp_path, vocab):
    from datasets import load_from_disk

    rows = [_row(tmp_path, i) for i in range(3)]
    build(rows, {str(i): GOOD for i in range(3)}, vocab, tmp_path / "out")
    a = load_from_disk(str(tmp_path / "out/route_a/train/raw"))
    b = load_from_disk(str(tmp_path / "out/route_b/train/raw"))
    assert a["audio_path"] == b["audio_path"] and a["duration"] == b["duration"]
    assert a["text"][0] == GOOD["deva"] and b["text"][0] == GOOD["latin"]


def test_manifest_relative_paths_resolve_read_only(tmp_path, vocab):
    rows = [_row(tmp_path, i, split="val") for i in range(2)]
    build(rows, {str(i): GOOD for i in range(2)}, vocab, tmp_path)
    d = tmp_path / "route_b" / "val_en"
    for p in [d, *d.rglob("*")]:
        p.chmod(0o555 if p.is_dir() else 0o444)
    try:
        ds = ManifestDataset(d, root=tmp_path)
        assert ds[0]["mel_spec"].shape[0] == 100 and ds.groups == ["en", "en"]
        assert ManifestDataset(d)[1]["text"] == GOOD["latin"]                  # default root: two levels up
        again = pickle.loads(pickle.dumps(ds))                                  # DataLoader workers (Task 3)
        assert again[0]["text"] == ds[0]["text"] and again.manifest_sha256() == ds.manifest_sha256()
    finally:
        for p in [d, *d.rglob("*")]:
            p.chmod(0o755 if p.is_dir() else 0o644)
