from train.stops import check_stop
from train.store import LocalStore, read_latest

RULES = {"target_group": "en", "plateau_rel": 0.005, "plateau_evals": 3, "forgetting_rel": 0.05,
         "forgetting_evals": 2, "baselines": {}}


def _h(key, values):
    return [{key: v} for v in values]


def test_plateau_stops_when_best_of_last_n_barely_beats_best_before():
    assert check_stop(_h("en", [1.0, 0.9, 0.899, 0.898, 0.8975]), RULES).startswith("plateau")


def test_improving_run_does_not_stop():
    assert check_stop(_h("en", [1.0, 0.9, 0.8, 0.7]), RULES) is None


def test_forgetting_needs_consecutive_evaluations_above_baseline():
    rules = dict(RULES, baselines={"mr": 1.0})
    assert check_stop([{"en": 1, "mr": 1.06}, {"en": 0.9, "mr": 1.07}], rules).startswith("forgetting: mr")
    assert check_stop([{"en": 1, "mr": 1.06}, {"en": 0.9, "mr": 1.02}], rules) is None


def test_trainer_stops_saves_and_uploads_on_a_stop_rule(tiny_cfg, vocab, tmp_path):
    from train.tests.test_trainer import _trainer
    stop = dict(tiny_cfg["validation"]["stop"], plateau_rel=1.0, plateau_evals=1)
    t, _ = _trainer(tiny_cfg, vocab, tmp_path, cfg_over={"validation": {"every_updates": 2, "stop": stop}})
    assert t.run() == "validation_stop"
    assert t.state["update"] == 4 and len(t.state["val_history"]) == 2
    assert read_latest(LocalStore(tmp_path / "hub"))["update"] == 4
