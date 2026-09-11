"""Marginal fidelity checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .. import config, copula


def evaluate(
    traits: pd.DataFrame, n_ref: int = 20000, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for cohort in config.COHORTS:
        sub = traits[traits.cohort == cohort]
        marg = config.COHORTS[cohort]["marginals"]
        u = np.sort(rng.random(n_ref))
        for v in config.COPULA_VARS:
            gen = sub[v].values
            ref = copula._invert(u, marg[v])
            scale = float(np.mean(np.abs(ref))) + 1e-9
            out.append(
                dict(
                    cohort=cohort,
                    trait=v,
                    KS=float(stats.ks_2samp(gen, ref).statistic),
                    W1_norm=float(
                        stats.wasserstein_distance(gen, ref) / scale
                    ),
                )
            )
    return pd.DataFrame(out)
