"""Two-sample discriminator check."""

from __future__ import annotations

import numpy as np

from .. import config, generator


def evaluate(size: int = 1500, seed_a: int = 42, seed_b: int = 7) -> dict:
    a = generator.generate_population(seed_a, size).traits
    b = generator.generate_population(seed_b, size).traits
    feats = list(config.COPULA_VARS)
    X = np.vstack([a[feats].values, b[feats].values])
    y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        auc = float(np.mean(cross_val_score(clf, X, y, cv=5, scoring="roc_auc")))
    except Exception:
        auc = float("nan")
    return dict(auc=auc, note="~0.50 -> two draws indistinguishable (good)")
