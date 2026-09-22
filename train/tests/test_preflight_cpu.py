import subprocess

from train.preflight_cpu import NOT_HERE, render, run_checks


def test_run_checks_maps_exit_codes_and_marks_data_checks_not_run():
    def fake(cmd, **kw):
        check = cmd[-1]                                   # the marker expression
        code = {"c5": 0, "c6": 1}.get(check, 5)          # 5 = pytest "no tests collected"
        return subprocess.CompletedProcess(cmd, code, stdout=f"{check} summary\n", stderr="")

    got = run_checks(["C3", "C5", "C6", "C7"], runner=fake)
    assert got["C5"]["status"] == "pass" and got["C6"]["status"] == "fail"
    assert got["C7"]["status"] == "fail"                 # a C5-C12 check with no tests is a failure
    assert got["C3"]["status"] == "not run" and "data" in got["C3"]["detail"]
    text = render(got, "abc123")
    assert "abc123" in text and "| C6 | fail |" in text
    assert set(NOT_HERE) == {"C1", "C2", "C3", "C4"}
