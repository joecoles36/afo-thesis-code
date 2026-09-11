"""Eligibility and deterioration checks."""

from __future__ import annotations

import numpy as np

from . import config


def headroom_series(weather, cfg: dict | None = None) -> np.ndarray:
    """Monthly headroom after essentials and discretionary spend (savings-only scope)."""
    return np.array(
        [
            weather.income[month] - weather.essential[month] - weather.discretionary[month]
            for month in range(config.N_MONTHS)
        ]
    )


def assess(weather, cfg: dict | None = None) -> dict:
    """Classify a customer's headroom quality: improving, stable, deteriorating, or out of scope."""
    headroom = headroom_series(weather, cfg)
    months = np.arange(len(headroom))
    slope = float(np.polyfit(months, headroom, 1)[0])  # £ per month trend (a line through the 12 points)
    trend_norm = slope / max(1.0, float(np.mean(weather.income)))  # trend as a fraction of income
    consistency = float(np.mean(headroom > 0))  # share of months with positive headroom
    mean_headroom = float(np.mean(headroom))

    if consistency < config.CONSISTENCY_MIN and mean_headroom < 0:
        status = "out_of_scope"  # rarely positive and negative on average -> specialist support
    elif trend_norm < config.DETERIORATION_TREND:
        status = "deteriorating"  # heading down -> eligible, but fix the leak first
    elif trend_norm > config.IMPROVEMENT_TREND:
        status = "improving"
    else:
        status = "stable"

    return dict(
        status=status,
        trend_gbp_per_month=slope,
        trend_norm=trend_norm,
        consistency=consistency,
        mean_headroom=mean_headroom,
    )
