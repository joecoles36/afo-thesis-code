"""Income-drop and one-off-cost shocks."""

from __future__ import annotations

import numpy as np

from . import config


def draw_shocks(cohort: str, rng: np.random.Generator):
    n = config.N_MONTHS
    sh = config.COHORTS[cohort]["shocks"]
    idr, ecost = sh["income_drop"], sh["essential_cost"]
    inc_mult = np.ones(n)  # income multiplier per month: 1.0 = normal, <1 = a drop
    ess_add = np.zeros(n)  # extra essential cost per month: 0 = normal, >0 = a one-off bill

    t = 0
    while t < n:
        if rng.random() < idr["p_monthly"]:  # does an income drop start this month?
            dur, frac = idr["duration_months"], idr["drop_fraction_mean"]
            for k in range(t, min(n, t + dur)):
                inc_mult[k] = min(inc_mult[k], 1.0 - frac)  # cut income by 'frac' for 'dur' months
            t += dur
        else:
            t += 1

    for m in range(n):
        if rng.random() < ecost["p_monthly"]:  # does a one-off big cost hit this month?
            ess_add[m] += rng.lognormal(np.log(ecost["median_gbp"]), ecost["sigma"])

    return inc_mult, ess_add
