import json
from pathlib import Path

NB = Path("train/kaggle/preflight_gpu.ipynb")


def _src():
    return "\n".join("".join(c["source"]) for c in json.loads(NB.read_text())["cells"])


def test_clones_fine_tune_branch_and_pins_base():
    s = _src()
    assert "-b fine-tune" in s and "ba85abedf18dc479a447eaa0eccbd76ab78a47d5" in s
    assert "ba7f3671180fb7784e24bd1dafc96e729a38ce02e7f6d3877cdef32525a1865c" in s


def test_token_comes_from_kaggle_secrets_only():
    s = _src()
    assert "UserSecretsClient" in s and "hf_" not in s.lower().replace("hf_token", "")


def test_runs_every_check_and_the_report():
    s = _src()
    assert all(f"preflight_runners g{i} " in s for i in range(1, 15))
    assert all(f"preflight_runners g{i} --route {r}" in s for i in (6, 7) for r in "ab")
    assert "preflight_runners report" in s


def test_train_venv_uses_cu126_torch():
    s = _src() + Path("requirements-train-kaggle.txt").read_text()
    assert "torch==2.11.0" in s and "cu126" in s
