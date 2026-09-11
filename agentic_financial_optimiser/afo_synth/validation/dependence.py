"""Dependence fidelity check."""

from __future__ import annotations

import numpy as np

from .. import config

CONTINUOUS = ["income", "essential_ratio", "discretionary_propensity", "slope"]


def evaluate(traits) -> dict:
    S = traits[list(config.COPULA_VARS)].corr("spearman").values
    T = np.array(config.SPEARMAN_TARGET)
    idx = [list(config.COPULA_VARS).index(c) for c in CONTINUOUS]
    Sc, Tc = S[np.ix_(idx, idx)], T[np.ix_(idx, idx)]
    return dict(
        frobenius_all=float(np.linalg.norm(S - T)),
        frobenius_continuous=float(np.linalg.norm(Sc - Tc)),
        max_abs_continuous=float(np.abs(Sc - Tc).max()),
    )
