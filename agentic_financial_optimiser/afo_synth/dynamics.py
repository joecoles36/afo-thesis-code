"""Monthly income and spending paths."""

from __future__ import annotations

import numpy as np

from . import config

STICKY_INFLATION = 0.002  # spending creeps up roughly 0.2% a month and never falls with income


def seasonal_bump(cohort: str, t: int) -> float:
    cal = ((config.START_MONTH - 1 + t) % 12) + 1  # which calendar month (1-12) is simulation month t
    s = config.COHORTS[cohort]["seasonal"]
    bump = 0.0
    if cal == 12:  # December: everyone spends more
        bump += s["december_bump"]
    if s["family_summer"] and cal in (8, 9):  # Aug/Sep bump, family cohorts only
        bump += s["late_summer_bump"]
    return s["amplitude"] * bump


def simulate_streams(trait: dict, cohort: str, rng: np.random.Generator) -> dict:
    n = config.N_MONTHS
    dyn = config.COHORTS[cohort]["dynamics"]
    rho, delta = dyn["rho"], dyn["delta"]
    eta_sd, tsd = dyn["level_innovation_sd"], dyn["transitory_sd"]
    slope = float(trait["slope"])

    inc_lvl = float(trait["income"])  # starting income level
    ess_lvl = float(trait["income"]) * float(trait["essential_ratio"])  # starting essential spend
    dis_lvl = float(trait["income"]) * float(trait["discretionary_propensity"])  # starting discretionary spend

    inc = np.empty(n)
    ess = np.empty(n)
    dis = np.empty(n)  # arrays to fill, one slot per month
    inc_obs_prev, ess_obs_prev, dis_obs_prev = inc_lvl, ess_lvl, dis_lvl  # last month's observed values
    inc_lvl_prev, ess_lvl_prev, dis_lvl_prev = inc_lvl, ess_lvl, dis_lvl  # last month's underlying levels

    for t in range(n):
        # 1) update the hidden level of each stream
        inc_lvl_t = inc_lvl_prev * (1.0 + delta * slope) + rng.normal(
            0, eta_sd * inc_lvl_prev
        )  # income drifts by the slope
        ess_lvl_t = ess_lvl_prev * (1.0 + STICKY_INFLATION) + rng.normal(
            0, eta_sd * ess_lvl_prev
        )  # essentials only creep up
        dis_lvl_t = dis_lvl_prev * (1.0 + STICKY_INFLATION) + rng.normal(
            0, eta_sd * dis_lvl_prev
        )  # discretionary only creeps up
        inc_lvl_t = max(200.0, inc_lvl_t)
        ess_lvl_t = max(0.0, ess_lvl_t)
        dis_lvl_t = max(0.0, dis_lvl_t)

        # 2) observed = level + a fading echo of last month's wobble + seasonal (disc only) + fresh noise
        season = seasonal_bump(cohort, t)
        inc_t = inc_lvl_t + rho * (inc_obs_prev - inc_lvl_prev) + rng.normal(0, tsd["income"] * inc_lvl_t)
        ess_t = ess_lvl_t + rho * (ess_obs_prev - ess_lvl_prev) + rng.normal(0, tsd["essential"] * ess_lvl_t)
        dis_t = (
            dis_lvl_t
            + dis_lvl_t * season
            + rho * (dis_obs_prev - dis_lvl_prev)
            + rng.normal(0, tsd["discretionary"] * dis_lvl_t)
        )

        # 3) store this month, then roll the "previous" trackers forward
        inc[t], ess[t], dis[t] = max(0.0, inc_t), max(0.0, ess_t), max(0.0, dis_t)
        inc_obs_prev, ess_obs_prev, dis_obs_prev = inc[t], ess[t], dis[t]
        inc_lvl_prev, ess_lvl_prev, dis_lvl_prev = inc_lvl_t, ess_lvl_t, dis_lvl_t

    return {"income": inc, "essential": ess, "discretionary": dis}
