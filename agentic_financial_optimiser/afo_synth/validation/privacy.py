"""Distance-to-closest-record check."""

from __future__ import annotations

import numpy as np

from .. import config, generator


def evaluate(size: int = 1500, seed_a: int = 42, seed_b: int = 7) -> dict:
    feats = list(config.COPULA_VARS)
    A = generator.generate_population(seed_a, size).traits[feats].values
    B = generator.generate_population(seed_b, size).traits[feats].values
    mu, sd = A.mean(0), A.std(0) + 1e-9
    A, B = (A - mu) / sd, (B - mu) / sd
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    dcr = d.min(1)
    return dict(min_dcr=float(dcr.min()), median_dcr=float(np.median(dcr)))
