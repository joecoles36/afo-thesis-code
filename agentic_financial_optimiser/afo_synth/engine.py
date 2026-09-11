"""Month-by-month accounting engine."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import config


@dataclass
class EngineState:
    current_account: float
    emergency: float  # instant-access product: the liquid safety buffer
    isa: float
    term_deposit: float
    regular_outgoings: float = 0.0  # essential + discretionary; drives the emergency target
    essential: float = 0.0
    income: float = 0.0


def distress_response(state: "EngineState", cfg: dict) -> float:
    """Overdraft cost keyed off the realised balance. Method-agnostic."""
    if state.current_account < 0:
        if state.current_account < cfg["arranged_overdraft_limit"]:
            rate, fee = cfg["unauthorised_ear"], cfg["unauthorised_monthly_fee"]
        else:
            rate, fee = cfg["overdraft_ear"], 0.0
        charge = state.current_account * (rate / 12.0) + fee
        state.current_account -= charge
        return charge
    return 0.0


def run(weather, method, cfg: dict | None = None) -> pd.DataFrame:
    cfg = config.ENGINE if cfg is None else cfg
    state = EngineState(
        weather.opening_pca,
        weather.opening_ia,
        weather.opening_isa,
        weather.opening_td,
    )
    base_apr = weather.savings_apr
    spread = config.PRODUCT_APR_SPREAD

    # Monthly growth factors
    emergency_growth = 1 + (base_apr + spread["instant_access"]) / 12.0
    isa_growth = 1 + (base_apr + spread["isa"]) / 12.0
    term_deposit_growth = 1 + (base_apr + spread["term_deposit"]) / 12.0

    rows = []
    for month in range(config.N_MONTHS):
        income = float(weather.income[month])
        essential = float(weather.essential[month])
        discretionary = float(weather.discretionary[month])

        surplus = income - essential - discretionary  # savings-only: no debt servicing in the cashflow
        state.current_account += surplus  # cash hits the current account
        allocatable = max(0.0, min(max(0.0, surplus), state.current_account))  # shared overdraft gate

        state.essential, state.income, state.regular_outgoings = (
            essential,
            income,
            essential + discretionary,
        )
        action = method.decide(state, allocatable)
        total = action.to_emergency + action.to_isa + action.to_term_deposit
        if total > allocatable + 1e-9 and total > 0:  # never allocate more than is free
            scale_factor = allocatable / total
            action.to_emergency *= scale_factor
            action.to_isa *= scale_factor
            action.to_term_deposit *= scale_factor

        state.current_account -= (
            action.to_emergency + action.to_isa + action.to_term_deposit
        )
        state.emergency += action.to_emergency
        state.isa += action.to_isa
        state.term_deposit += action.to_term_deposit

        # per-product interest: instant access at the base rate, ISA and term deposit at a premium
        pre_liquid = state.emergency + state.isa + state.term_deposit
        state.emergency *= emergency_growth
        state.isa *= isa_growth
        state.term_deposit *= term_deposit_growth
        interest_earned = (
            state.emergency + state.isa + state.term_deposit
        ) - pre_liquid

        # the instant-access buffer covers a shortfall before any overdraft fee.
        # ISA and term deposit are left untouched (term deposit is locked in real life).
        emergency_drawn = 0.0
        if state.current_account < 0 and state.emergency > 0:
            emergency_drawn = min(-state.current_account, state.emergency)
            state.emergency -= emergency_drawn
            state.current_account += emergency_drawn
        charge = distress_response(state, cfg)

        rows.append(
            dict(
                month=month,
                income=income,
                essential=essential,
                discretionary=discretionary,
                surplus=surplus,
                current_account=state.current_account,
                emergency=state.emergency,
                isa=state.isa,
                term_deposit=state.term_deposit,
                total_savings=state.emergency + state.isa + state.term_deposit,
                overdraft=int(state.current_account < 0),
                distress_charge=charge,
                interest_earned=interest_earned,
                emergency_drawn=emergency_drawn,
                to_emergency=action.to_emergency,
                to_isa=action.to_isa,
                to_term_deposit=action.to_term_deposit,
            )
        )

    return pd.DataFrame(rows)
