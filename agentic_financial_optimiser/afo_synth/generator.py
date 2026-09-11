"""Builds the synthetic customer population."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, copula, dynamics, shocks


@dataclass
class Weather:
    customer_id: str
    cohort: str
    traits: dict
    income: np.ndarray
    essential: np.ndarray
    discretionary: np.ndarray
    savings_apr: float
    opening_pca: float
    opening_savings: float
    opening_emergency: float = 0.0
    # Opening savings split across the three real products. These sum to
    # opening_savings. The split is by the cohort's savings_split shares.
    opening_ia: float = 0.0
    opening_isa: float = 0.0
    opening_td: float = 0.0


@dataclass
class Population:
    customers: list
    traits: pd.DataFrame
    exog_panel: pd.DataFrame


def _cohort_counts(total: int) -> dict:
    names = list(config.COHORTS)
    shares = np.array([config.COHORTS[c]["share"] for c in names], float)
    counts = np.floor(shares / shares.sum() * total).astype(int)
    counts[-1] += total - counts.sum()
    return dict(zip(names, counts))


def _split_savings(total: float, cohort: str) -> dict:
    """Split an opening savings balance across the three products by cohort shares.

    A deterministic mean split: every customer in a cohort holds the same
    proportions. Per-customer variation is a later calibration refinement. The
    split consumes no random draws, so adding it leaves every existing draw (and
    therefore the whole seeded population) unchanged, and only adds columns.
    """
    split = config.COHORTS[cohort]["savings_split"]
    shares = np.array([split[p] for p in config.PRODUCTS], float)
    shares = shares / shares.sum()  # guard: normalise in case shares don't sum to 1
    ia, isa, td = (float(total) * shares).tolist()
    return {"opening_ia": ia, "opening_isa": isa, "opening_td": td}


def make_cohort_customers(
    cohort: str,
    n: int,
    rng: np.random.Generator,
    cid_start: int = 0,
) -> list:
    """Create n Weather customers of one cohort. Shared by the full and eligible-only generators.

    The rng draw order (batch copula sample, then per customer: streams, shocks, rate, opening
    balance) is fixed so a given seed reproduces the same population.
    """
    if n <= 0:
        return []

    tdf = copula.sample_traits(cohort, n, rng)
    rmean = config.COHORTS[cohort]["rates"]
    op = config.COHORTS[cohort]["opening_pca"]
    out = []

    for i in range(n):
        trait = tdf.iloc[i].to_dict()
        streams = dynamics.simulate_streams(trait, cohort, rng)
        inc_mult, ess_add = shocks.draw_shocks(cohort, rng)
        income = streams["income"] * inc_mult
        essential = streams["essential"] + ess_add
        discretionary = streams["discretionary"]
        sav_apr = float(
            np.clip(
                rng.normal(
                    rmean["savings_apr"]["mean"],
                    rmean["savings_apr"]["sd"],
                ),
                0.0,
                0.15,
            )
        )
        opening_pca = float(
            rng.normal(
                op["income_mult_mean"] * trait["income"],
                op["sd_fraction"] * op["income_mult_mean"] * trait["income"],
            )
        )
        opening_savings = float(trait["savings_buffer"])
        split = _split_savings(opening_savings, cohort)

        out.append(
            Weather(
                f"C{cid_start + i:05d}",
                cohort,
                trait,
                income,
                essential,
                discretionary,
                sav_apr,
                opening_pca,
                opening_savings,
                **split,
            )
        )

    return out


def assemble(customers: list) -> Population:
    """Build a Population (trait and exogenous-panel DataFrames) from a customer list."""
    trait_rows, panel_rows = [], []

    for w in customers:
        trait_rows.append(
            dict(
                customer_id=w.customer_id,
                cohort=w.cohort,
                **{k: w.traits[k] for k in config.COPULA_VARS},
                savings_apr=w.savings_apr,
                opening_pca=w.opening_pca,
                opening_ia=w.opening_ia,
                opening_isa=w.opening_isa,
                opening_td=w.opening_td,
            )
        )

        for t in range(config.N_MONTHS):
            panel_rows.append(
                dict(
                    customer_id=w.customer_id,
                    cohort=w.cohort,
                    month=t,
                    income=w.income[t],
                    essential=w.essential[t],
                    discretionary=w.discretionary[t],
                )
            )

    return Population(
        customers,
        pd.DataFrame(trait_rows),
        pd.DataFrame(panel_rows),
    )


def generate_population(
    seed: int | None = None,
    size: int | None = None,
) -> Population:
    """Generate the full exogenous population by natural cohort shares. Reproducible from seed."""
    seed = config.SEED if seed is None else seed
    size = config.POPULATION_SIZE if size is None else size
    rng = np.random.default_rng(seed)

    customers, cid = [], 0
    for cohort, nc in _cohort_counts(size).items():
        customers += make_cohort_customers(cohort, nc, rng, cid)
        cid += nc

    return assemble(customers)
