"""
Run FINETUNE_PLAN §8a's CPU checks that exist in this repo and write train/preflight_report.md.
C1-C4 check prepared data (§15 step 2); only their patch-level tests exist yet, so they are listed as
not run here rather than passed.
"""
from __future__ import annotations

import datetime
import subprocess
import sys
from pathlib import Path

CHECKS = [f"C{n}" for n in range(1, 13)]
NOT_HERE = {c: "needs prepared data (§15 step 2); patch-level tests only" for c in ("C1", "C2", "C3", "C4")}


def run_checks(checks, runner=subprocess.run):
    results = {}
    for c in checks:
        if c in NOT_HERE:
            results[c] = {"status": "not run", "detail": NOT_HERE[c]}
            continue
        cmd = [sys.executable, "-m", "pytest", "train/tests", "-q", "-m", c.lower()]
        r = runner(cmd, capture_output=True, text=True)
        last = (r.stdout.strip().splitlines() or [""])[-1]
        results[c] = {"status": "pass" if r.returncode == 0 else "fail",
                      "detail": last if r.returncode != 5 else "no tests collected"}
    return results


def render(results, commit) -> str:
    rows = "".join(f"| {c} | {r['status']} | {r['detail']} |\n" for c, r in results.items())
    return (f"# CPU pre-flight (FINETUNE_PLAN §8a)\n\ncommit `{commit}`, "
            f"{datetime.datetime.now().isoformat(timespec='seconds')}\n\n| check | status | detail |\n|---|---|---|\n{rows}")


def main() -> int:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    results = run_checks(CHECKS)
    Path("train/preflight_report.md").write_text(render(results, commit))
    print(render(results, commit))
    return 1 if any(r["status"] == "fail" for r in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
