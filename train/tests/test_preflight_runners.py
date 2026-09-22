import json

from train import preflight_runners as pr


def test_g2_runner_on_a_tiny_release_file(tmp_path, tiny_cfg, tiny_model):
    from safetensors.torch import save_file

    from train.model import to_release

    save_file(to_release({k: v.contiguous() for k, v in tiny_model.state_dict().items()}, {}), str(tmp_path / "b.safetensors"))
    res = pr.g2(base=tmp_path / "b.safetensors", cfg=tiny_cfg, expected_params=sum(p.numel() for p in tiny_model.parameters()))
    assert res["status"] == "pass", res


def test_g4_mel_parity_on_a_manifest(tmp_path, vocab):
    from prep.build_manifests import build
    from prep.tests.test_build_manifests import GOOD, _row

    build([_row(tmp_path, 0, split="val")], {"0": GOOD}, vocab, tmp_path)
    res = pr.g4(data=tmp_path)
    assert res["status"] == "pass" and "rms" in res["detail"], res


def test_every_check_has_a_runner():
    assert all(callable(getattr(pr, f"g{i}")) for i in range(1, 15))


def test_cli_writes_json(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "g1", lambda **kw: {"status": "pass", "detail": "stub"})
    pr.main(["g1", "--out", str(tmp_path)])
    assert json.loads((tmp_path / "g1.json").read_text())["status"] == "pass"


def test_per_route_checks_combine(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "g7", lambda route, **kw: {"status": "pass", "detail": route, "frames_per_gpu": {"a": 12000, "b": 9000}[route]})
    pr.main(["g7", "--route", "a", "--out", str(tmp_path)])
    assert json.loads((tmp_path / "g7.json").read_text())["status"] == "partial"      # route b not run yet
    pr.main(["g7", "--route", "b", "--out", str(tmp_path)])
    combined = json.loads((tmp_path / "g7.json").read_text())
    assert combined["status"] == "pass" and combined["frames_per_gpu"] == 9000


def test_request_names_a_val_clip_its_text_and_a_fixed_duration(tmp_path, vocab):
    from prep.build_manifests import build
    from prep.tests.test_build_manifests import GOOD, _row

    build([_row(tmp_path, 0, split="val")], {"0": GOOD}, vocab, tmp_path)
    pr.main(["request", "--data", str(tmp_path), "--out", str(tmp_path / "o")])
    req = json.loads((tmp_path / "o" / "request.json").read_text())
    assert req["ref"].endswith("audio/0.flac") and req["ref_text"] == GOOD["deva"]
    assert req["fix_duration"] == 3.0 + 3.0 and all(c in vocab for c in req["gen_text"])
