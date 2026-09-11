"""Adapters to convert engine output to score inputs.

Wired to the savings-only engine's real per-product columns (emergency =
instant access, isa, term_deposit). Two synthetic-side estimates live here:

    isa_utilisation resets at the ISA tax-year boundary (6 April), placed in the
    synthetic year from START_MONTH. With START_MONTH = 7 that is month index 9,
    matching the July-to-June sample. The whole April month is taken as the reset
    point (the agreed month-boundary approximation).

    the overdraft "used this month" signal is a per-month proxy, since the monthly
    engine has no intra-month balance. It is a seeded Bernoulli draw whose
    probability is 1 when the month ends negative and decays as the end-of-month
    balance rises, so the synthetic reproduces the intra-month signal the real
    data has. Seeded per customer so the population stays reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from . import scoring_function


_ISA_ALLOWANCE = config.RUN["isa_allowance"]
_APRIL_IDX = (4 - config.START_MONTH) % 12  # index of April in the synthetic year
_OVERDRAFT_PROXY_SCALE = 500.0  # GBP; placeholder, a calibration target


def isa_utilisation_series(panel: pd.DataFrame) -> np.ndarray:
    """Monthly ISA utilisation: cumulative ISA inflow within the current tax year / allowance."""
    to_isa = panel["to_isa"].to_numpy()
    cum = np.zeros(len(to_isa))
    running = 0.0
    for t in range(len(to_isa)):
        if t == _APRIL_IDX:
            running = 0.0  # tax-year reset at 6 April
        running += to_isa[t]
        cum[t] = running
    return np.minimum(cum / _ISA_ALLOWANCE, 1.0)


def overdraft_used_flags(panel: pd.DataFrame, customer_id) -> np.ndarray:
    """Per-month 'used overdraft at any point this month' proxy, 0/1, seeded per customer."""
    bal = panel["current_account"].to_numpy()
    p = np.where(bal <= 0.0, 1.0, np.exp(-bal / _OVERDRAFT_PROXY_SCALE))
    digits = "".join(ch for ch in str(customer_id) if ch.isdigit()) or "0"
    rng = np.random.default_rng(config.SEED + int(digits))
    return (rng.random(len(bal)) < p).astype(float)


def health_score_inputs(
    weather, panel: pd.DataFrame
) -> scoring_function.ScoreInputs:
    """Derive ScoreInputs from a Weather object and its engine panel (year-end grain)."""
    end = panel.iloc[-1]
    regular_outgoings = panel["essential"] + panel["discretionary"]
    total_savings = panel["emergency"] + panel["isa"] + panel["term_deposit"]

    emergency_attainment = (
        panel["emergency"] >= (3.0 * regular_outgoings)
    ).astype(float)
    positive_savings_attainment = (total_savings > 0.0).astype(float)
    isa_util = isa_utilisation_series(panel)
    overdraft_used = overdraft_used_flags(panel, weather.customer_id)

    return scoring_function.ScoreInputs(
        days_in_overdraft=float(overdraft_used.sum()),
        overdraft_fees_total=float(panel["distress_charge"].sum()),
        fully_funded_emergency_flag=float(emergency_attainment.sum()) / config.N_MONTHS,
        savings_consistency_flag=float(positive_savings_attainment.sum()) / config.N_MONTHS,
        isa_utilisation=float(isa_util[-1]),
        non_ia_savings_balance=float(end["isa"] + end["term_deposit"]),
        total_savings_interest=float(panel["interest_earned"].mean()),
        average_savings_balance=float(total_savings.mean()),
        total_savings_balance=float(
            end["emergency"] + end["isa"] + end["term_deposit"]
        ),
    )


def score_panel_monthly(
    weather,
    panel: pd.DataFrame,
    scorer: scoring_function.Scorer | None = None,
) -> pd.DataFrame:
    """Score each month of an engine panel individually. One row per month (1-12)."""
    if scorer is None:
        scorer = scoring_function.Scorer()

    regular_outgoings = panel["essential"] + panel["discretionary"]
    total_savings = panel["emergency"] + panel["isa"] + panel["term_deposit"]

    isa_util = isa_utilisation_series(panel)
    overdraft_used = overdraft_used_flags(panel, weather.customer_id)
    emergency_hits = (
        panel["emergency"] >= (3.0 * regular_outgoings)
    ).astype(float)
    cumulative_emergency_hits = emergency_hits.cumsum()
    positive_savings_hits = (total_savings > 0.0).astype(float)
    cumulative_positive_savings_hits = positive_savings_hits.cumsum()

    rows = []
    for t in range(len(panel)):
        row = panel.iloc[t]
        total_bal = float(
            row["emergency"] + row["isa"] + row["term_deposit"]
        )
        inputs = scoring_function.ScoreInputs(
            days_in_overdraft=float(overdraft_used[t]) * 30.0,
            overdraft_fees_total=float(row["distress_charge"]),
            fully_funded_emergency_flag=float(
                cumulative_emergency_hits.iloc[t]
            ) / float(t + 1),
            savings_consistency_flag=float(
                cumulative_positive_savings_hits.iloc[t]
            ) / float(t + 1),
            isa_utilisation=float(isa_util[t]),
            non_ia_savings_balance=float(
                row["isa"] + row["term_deposit"]
            ),
            total_savings_interest=float(row["interest_earned"]),
            average_savings_balance=total_bal,
            total_savings_balance=total_bal,
        )
        scoring_function.assert_valid_health_score_inputs(
            inputs, context=f"Month {t + 1}"
        )
        r = scorer.score(inputs)
        rows.append(
            dict(
                month=t + 1,
                spending=r.spending_score,
                emergency=r.emergency_score,
                savings=r.savings_score,
                overall=r.overall_score,
            )
        )

    return pd.DataFrame(rows).set_index("month")
