import pytest

from train.tests.ddp_harness import run_two_ranks


@pytest.mark.slow
def test_guard_decision_is_shared_so_ranks_stop_together(tmp_path):
    r0, r1 = run_two_ranks("guard", tmp_path, {"runtime": {"guard_seconds": 60}})
    assert r0["reason"] == r1["reason"] == "time_guard"
    assert r0["state"]["update"] == r1["state"]["update"]


@pytest.mark.slow
def test_only_the_main_rank_downloads_on_resume(tmp_path):
    first = run_two_ranks("plain", tmp_path, {"schedule": {"epochs": 1}, "checkpoint": {"every_updates": 2}})
    # a second session with the same config resumes from LATEST
    second = run_two_ranks("plain", tmp_path, {"schedule": {"epochs": 1}, "checkpoint": {"every_updates": 2}})
    assert (tmp_path / "downloads.txt").read_text().split() == ["0"]
    assert first[0]["state"]["update"] == second[0]["state"]["update"] == second[1]["state"]["update"]


@pytest.mark.slow
def test_step_loss_is_reduced_so_ranks_agree(tmp_path):
    r0, r1 = run_two_ranks("poison", tmp_path)
    assert r0["state"]["recent_losses"] == r1["state"]["recent_losses"]
    assert r0["state"]["update"] == r1["state"]["update"]
