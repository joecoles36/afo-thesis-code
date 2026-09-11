"""Generic utilities for analysis and aggregation."""

from __future__ import annotations

import pandas as pd


def inscope_customers(pop, cohort: str, limit: int | None = None) -> list:
    """Return in-scope customers for a cohort, optionally capped at limit."""
    from . import eligibility

    customers = [
        w for w in pop.customers if w.cohort == cohort and eligibility.assess(w)["status"] != "out_of_scope"
    ]
    if not customers:
        raise ValueError(f"No in-scope customers found for cohort '{cohort}'")
    return customers[:limit] if limit is not None else customers


def mean_monthly_scores(customers: list, method, score_panel_fn, score_cols: list[str]) -> pd.DataFrame:
    """Average scores across all customers for one method.

    Parameters
    ----------
    customers : list
        List of Weather objects to score
    method : object
        Allocation method to apply
    score_panel_fn : callable
        Function that takes (weather, panel) and returns scores DataFrame
    score_cols : list[str]
        Columns to extract from the scores

    Returns
    -------
    pd.DataFrame
        Indexed by month (1-12) with requested columns, values are averages
    """
    from . import engine

    frames = [score_panel_fn(w, engine.run(w, method))[score_cols] for w in customers]
    return pd.concat(frames).groupby("month").mean().round(3)
