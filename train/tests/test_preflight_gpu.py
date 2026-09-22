import numpy as np

from train import preflight_gpu as g


def _log(n, loss=lambda u: 1.0, step=1.0, data=0.05, fps=1000.0, scale=65536.0, start=1):
    return [{"update": u, "outcome": "ok", "loss": loss(u), "step_s": step, "data_s": data, "fps": fps,
             "frames": fps * (step + data), "scale": scale, "grad_norm": 0.5, "mem_gb": 10.0} for u in range(start, start + n)]


def test_g2():
    assert g.g2_verdict([], 337_096_804, False)["status"] == "pass"
    assert g.g2_verdict(["missing module x"], 337_096_804, False)["status"] == "fail"
    assert g.g2_verdict([], 337_096_804, True)["status"] == "fail"


def test_g3_tolerance():
    a = np.zeros((100, 50), dtype=np.float32)
    assert g.g3_verdict(a, a + 5e-4)["status"] == "pass"
    assert g.g3_verdict(a, a + 2e-3)["status"] == "fail"
    assert g.g3_verdict(a, np.zeros((100, 49)))["status"] == "fail"          # shape mismatch is a failure, not a crash


def test_g5_partial_without_indic():
    assert g.g5_verdict({"en_a": 0.9, "en_b": 1.4})["status"] == "partial"
    assert g.g5_verdict({"en_a": 0.9, "en_b": 1.4, "hi": 0.5})["status"] == "pass"
    assert g.g5_verdict({"en_a": 0.5, "en_b": 1.4, "hi": 0.9})["status"] == "fail"   # val_hi must be below every val_en


def test_g6_needs_half_drop():
    assert g.g6_verdict(_log(300, loss=lambda u: 1.0 if u <= 10 else 0.4))["status"] == "pass"
    assert g.g6_verdict(_log(300, loss=lambda u: 1.0 if u <= 10 else 0.6))["status"] == "fail"


def test_g7_binary_search():
    r = g.g7_search(lambda frames: frames * 1e6, lo=4000, hi=40000, limit_bytes=0.85 * 16e9)
    assert r["frames_per_gpu"] == 13000 and r["status"] == "pass"
    assert g.g7_search(lambda f: 1e12, lo=4000, hi=40000, limit_bytes=1e9)["status"] == "fail"


def test_g8_wait_fraction():
    assert g.g8_verdict(_log(210, data=0.05))["status"] == "pass"
    assert g.g8_verdict(_log(210, data=0.5))["status"] == "fail"


def test_g9_scaling_and_loss_match():
    one = _log(100, fps=1000.0, loss=lambda u: 1.0 + 0.01 * (u % 3))
    assert g.g9_verdict(one, _log(100, fps=1800.0, loss=lambda u: 1.0 + 0.01 * (u % 3)))["status"] == "pass"
    assert g.g9_verdict(one, _log(100, fps=1500.0, loss=lambda u: 1.0 + 0.01 * (u % 3)))["status"] == "fail"
    assert g.g9_verdict(one, _log(100, fps=1800.0, loss=lambda u: 2.0))["status"] == "fail"


def test_g10():
    assert g.g10_verdict(_log(1000))["status"] == "pass"
    bad = _log(1000) + [{"event": "nonfinite", "update": 5, "loss": float("nan")}]
    assert g.g10_verdict(bad)["status"] == "fail"
    assert g.g10_verdict(_log(1000, scale=0.5))["status"] == "fail"


def test_g11():
    log = _log(100) + [{"event": "resumed", "update": 100, "epoch": 0, "batch": 200}] + _log(50, start=101)
    assert g.g11_verdict(log)["status"] == "pass"
    skip = _log(100) + [{"event": "resumed", "update": 100, "epoch": 0, "batch": 200}] + _log(50, start=103)
    assert g.g11_verdict(skip)["status"] == "fail"
    assert g.g11_verdict(_log(100))["status"] == "fail"


def test_g12():
    log = _log(500) + [{"event": "validation", "update": 500, "seconds": 10.0}]
    assert g.g12_verdict(log)["status"] == "pass"
    assert g.g12_verdict(log + [{"event": "validation", "update": 501, "seconds": 100.0}])["status"] == "fail"


def test_g14():
    log = _log(10) + [{"event": "time_guard", "update": 10}]
    assert g.g14_verdict(log, {"update": 10}, hub=True)["status"] == "pass"
    assert g.g14_verdict(log, {"update": 10}, hub=False)["status"] == "partial"
    assert g.g14_verdict(_log(10), None, hub=True)["status"] == "fail"


def test_report_lists_every_check(tmp_path):
    res = {f"G{i}": {"status": "not run", "detail": "-"} for i in range(1, 15)}
    text = g.render(res, "abc123")
    assert all(f"| G{i} |" in text for i in range(1, 15)) and "abc123" in text
