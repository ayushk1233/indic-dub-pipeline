"""FINETUNE_PLAN §10b stop rules, judged on EMA validation losses in evaluation order."""
from __future__ import annotations


def check_stop(history, rules):
    k = rules["forgetting_evals"]
    for group, base in rules["baselines"].items():
        recent = [h.get(group) for h in history[-k:]]
        if len(recent) == k and all(v is not None and v > base * (1 + rules["forgetting_rel"]) for v in recent):
            return f"forgetting: {group} more than {rules['forgetting_rel']:.0%} above its baseline for {k} evaluations"
    n, g = rules["plateau_evals"], rules["target_group"]
    values = [h[g] for h in history if g in h]
    if len(values) > n and min(values[-n:]) > min(values[:-n]) * (1 - rules["plateau_rel"]):
        return f"plateau: {g} improved less than {rules['plateau_rel']:.1%} over {n} evaluations"
    return None
