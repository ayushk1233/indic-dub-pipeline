"""
Every output path in this project was /content, which exists on Colab and
nowhere else. The work moved to Kaggle for its 30 weekly GPU hours and every
module wrote into a directory that does not belong to the host.
"""

import pytest

from colab import workspace


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(workspace.ENV_VAR, raising=False)


def test_an_explicit_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv(workspace.ENV_VAR, str(tmp_path))
    assert workspace.root() == tmp_path


def test_kaggle_is_preferred_over_colab(monkeypatch, tmp_path):
    kaggle, content = tmp_path / "kaggle", tmp_path / "content"
    kaggle.mkdir()
    content.mkdir()
    monkeypatch.setattr(workspace, "CANDIDATES", (str(kaggle), str(content)))
    assert workspace.root() == kaggle


def test_colab_is_used_when_kaggle_is_absent(monkeypatch, tmp_path):
    content = tmp_path / "content"
    content.mkdir()
    monkeypatch.setattr(workspace, "CANDIDATES",
                        (str(tmp_path / "absent"), str(content)))
    assert workspace.root() == content


def test_a_plain_box_falls_back_to_the_working_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(workspace, "CANDIDATES", (str(tmp_path / "nope"),))
    monkeypatch.chdir(tmp_path)
    assert workspace.root() == tmp_path


def test_importing_creates_nothing(monkeypatch, tmp_path):
    """A module-level path must not have a filesystem side effect."""
    monkeypatch.setenv(workspace.ENV_VAR, str(tmp_path / "unborn"))
    workspace.out("audio")
    workspace.report("run.txt")
    assert not (tmp_path / "unborn").exists()


def test_names_hang_off_the_root(monkeypatch, tmp_path):
    monkeypatch.setenv(workspace.ENV_VAR, str(tmp_path))
    assert workspace.out("indicf5_check") == tmp_path / "indicf5_check"
    assert workspace.report("run.txt") == tmp_path / "run.txt"
