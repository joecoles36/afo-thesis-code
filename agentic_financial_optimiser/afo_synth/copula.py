"""Correlated trait sampling via a Gaussian Copula."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config


def spearman_to_pearson(spearman) -> np.ndarray:
    # convert the intuitive rank correlations into the form the Gaussian draw needs
    P = 2.0 * np.sin(np.pi / 6.0 * np.asarray(spearman, dtype=float))
    np.fill_diagonal(P, 1.0)
    return nearest_psd(P)


def nearest_psd(matrix) -> np.ndarray:
    # nudge a hand-written correlation matrix to the nearest valid one
    M = (np.asarray(matrix, float) + np.asarray(matrix, float).T) / 2.0
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 1e-8, None)
    A = (V * w) @ V.T
    d = np.sqrt(np.diag(A))
    A = A / np.outer(d, d)
    np.fill_diagonal(A, 1.0)
    return A


def _invert(u: np.ndarray, spec: dict) -> np.ndarray:
    # turn percentiles (u, 0 to 1) into actual trait values using the trait's own distribution
    dist = spec["dist"]
    if dist == "lognormal":
        cv, mean = spec["cv"], spec["mean"]
        sigma = np.sqrt(np.log(1.0 + cv**2))
        scale = mean / np.sqrt(1.0 + cv**2)
        x = stats.lognorm.ppf(u, s=sigma, scale=scale)
        return np.maximum(spec.get("floor", 0.0), x)
    if dist == "beta":
        m, k = spec["mean"], spec["concentration"]
        return stats.beta.ppf(u, m * k, (1.0 - m) * k)
    if dist == "gamma":
        mean, shape = spec["mean"], spec["shape"]
        return stats.gamma.ppf(u, a=shape, scale=mean / shape)
    if dist == "lognormal_zi":  # zero-inflated: some customers get exactly 0
        p0, med, sig = spec["p_zero"], spec["median"], spec["sigma"]
        x = np.zeros_like(u)
        nz = u > p0
        x[nz] = stats.lognorm.ppf((u[nz] - p0) / (1.0 - p0), s=sig, scale=med)
        return x
    if dist == "mixture_normal":  # blend of a declining group and a stable group
        pdec, d, s = spec["p_declining"], spec["declining"], spec["stable"]
        lo = min(d["mean"] - 6 * d["sd"], s["mean"] - 6 * s["sd"])
        hi = max(d["mean"] + 6 * d["sd"], s["mean"] + 6 * s["sd"])
        xs = np.linspace(lo, hi, 4000)
        cdf = pdec * stats.norm.cdf(xs, d["mean"], d["sd"]) + (1 - pdec) * stats.norm.cdf(xs, s["mean"], s["sd"])
        return np.interp(u, cdf, xs)
    raise ValueError(f"unknown marginal: {dist}")


def sample_traits(cohort: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    Sigma = spearman_to_pearson(config.SPEARMAN_TARGET)  # the valid correlation matrix
    L = np.linalg.cholesky(Sigma)  # its "square root", used to correlate the draws
    Z = rng.standard_normal((n, len(config.COPULA_VARS))) @ L.T  # n customers of correlated random numbers
    U = stats.norm.cdf(Z)  # turn those into percentiles (0 to 1)
    marg = config.COHORTS[cohort]["marginals"]
    cols = {name: _invert(U[:, j], marg[name]) for j, name in enumerate(config.COPULA_VARS)}
    df = pd.DataFrame(cols)
    df.insert(0, "cohort", cohort)
    return df
