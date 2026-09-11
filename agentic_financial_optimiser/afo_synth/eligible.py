"""Eligible-only population sampling."""

from __future__ import annotations

import numpy as np

from . import config, eligibility, generator

CAP_HEADROOM = 100.0  # eligible if average headroom is above this
CAP_LIQUID = 3000.0  # or total liquid balances are above this


def is_eligible(w) -> bool:
    """Financial capacity and not in persistent difficulty."""
    h = eligibility.headroom_series(w)  # their 12 monthly headroom figures
    has_capacity = (float(np.mean(h)) > CAP_HEADROOM) or (float(w.opening_pca + w.opening_savings) > CAP_LIQUID)
    return has_capacity and eligibility.assess(w)["status"] != "out_of_scope"


def generate_eligible_population(
    seed: int | None = None, size: int = 1000, target_mix: dict | None = None, max_rounds: int = 80
):
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    cohorts = list(config.COHORTS)
    kept = {c: [] for c in cohorts}  # eligible customers found so far, per cohort

    if target_mix is None:
        targets = None  # no target: take the neutral eligible mix
    else:
        s = sum(target_mix.values())
        targets = {c: int(round(size * target_mix.get(c, 0) / s)) for c in cohorts}
        targets[cohorts[0]] += size - sum(targets.values())  # tidy rounding onto the first cohort

    cid = 0
    for _ in range(max_rounds):
        if targets is None:  # natural mode: stop when we have enough total
            have = sum(len(v) for v in kept.values())
            if have >= size:
                break
            need = size - have
            attempt = {c: max(20, int(round(need * config.COHORTS[c]["share"] * 2.5))) for c in cohorts}
        else:  # target mode: stop when every cohort is full
            if all(len(kept[c]) >= targets[c] for c in cohorts):
                break
            attempt = {
                c: 0 if len(kept[c]) >= targets[c] else max(50, (targets[c] - len(kept[c])) * 12) for c in cohorts
            }

        for c in cohorts:
            if attempt[c] <= 0:
                continue
            for w in generator.make_cohort_customers(c, attempt[c], rng, cid):
                if is_eligible(w):
                    kept[c].append(w)  # keep only the eligible ones
            cid += attempt[c]

    if targets is None:
        customers = [w for c in cohorts for w in kept[c]]
        rng.shuffle(customers)
        customers = customers[:size]  # natural mix: trim to exactly 'size'
    else:
        customers = [w for c in cohorts for w in kept[c][: targets[c]]]  # take each cohort's target count

    for i, w in enumerate(customers):
        w.customer_id = f"C{i:05d}"  # renumber ids 0 size-1

    return generator.assemble(customers)
