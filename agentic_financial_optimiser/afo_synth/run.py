"""Demo run."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import eligibility, engine, generator
from .methods.baseline import NullMethod
from .methods.hybrid import RankedMethod
from .methods.piping import PipingMethod
from .methods.rules import RulesMethod

logger = logging.getLogger(__name__)

METHODS = {
    "do_nothing": NullMethod(),
    "rules": RulesMethod(),
    "ranked": RankedMethod(),
    "allocation_equal": PipingMethod("equal"),
}


def outcome_metrics(panel: pd.DataFrame) -> dict:
    end = panel.iloc[-1]
    return dict(
        resilience_months=end.emergency / max(1.0, end.essential),  # months the liquid buffer covers
        overdraft_rate=float(panel.overdraft.mean()),
        emergency=float(end.emergency),
        isa=float(end.isa),
        term_deposit=float(end.term_deposit),
        total_savings=float(end.emergency + end.isa + end.term_deposit),
        distress_paid=float(panel.distress_charge.sum()),
    )


def main(seed: int = 42, size: int = 1500):
    logger.info("Generating population (seed=%s, size=%s)", seed, size)
    population = generator.generate_population(seed=seed, size=size)
    logger.info("Generated population of %s customers", len(population.customers))

    logger.info("Calculating eligibility for %s customers", len(population.customers))
    assessments = [
        dict(cohort=customer.cohort, **eligibility.assess(customer))
        for customer in population.customers
    ]
    logger.info("Eligibility calculated for %s customers", len(assessments))
    eligibility_df = pd.DataFrame(assessments)
    crosstab = pd.crosstab(
        eligibility_df.cohort, eligibility_df.status, normalize="index"
    )
    status_order = [
        c
        for c in ["out_of_scope", "deteriorating", "stable", "improving"]
        if c in crosstab.columns
    ]
    logger.info(
        "=== Deterioration screen (share of each cohort by headroom trend) ===\n%s",
        crosstab[status_order].round(2).to_string(),
    )

    logger.info("Applying %s methods to in-scope customers", list(METHODS.keys()))
    rows = []
    snapshot = None
    for customer, assessment in zip(population.customers, assessments):
        if assessment["status"] == "out_of_scope":
            continue
        if snapshot is None:
            snapshot = customer.income.copy()
        for method_name, method in METHODS.items():
            rows.append(
                dict(
                    cohort=customer.cohort,
                    method=method_name,
                    **outcome_metrics(engine.run(customer, method)),
                )
            )
    logger.info("Applied methods, produced %s outcome rows", len(rows))
    outcomes_df = pd.DataFrame(rows)
    metric_cols = [
        "resilience_months",
        "overdraft_rate",
        "emergency",
        "isa",
        "term_deposit",
        "total_savings",
        "distress_paid",
    ]

    pooled_means = outcomes_df.groupby("method")[metric_cols].mean().reindex(METHODS)
    n_in_scope = outcomes_df.cohort.count() // len(METHODS)
    logger.info(
        "=== Outcome metrics, mean over in-scope customers (n=%s) ===\n%s",
        n_in_scope,
        pooled_means.round(2).to_string(),
    )
    logger.info(
        "=== Delta vs do-nothing (pooled) ===\n%s",
        (pooled_means - pooled_means.loc["do_nothing"]).round(2).to_string(),
    )

    logger.info("=== Where the methods diverge: by cohort ===")
    for metric in ["resilience_months", "total_savings", "emergency"]:
        pivot = outcomes_df.pivot_table(
            index="cohort", columns="method", values=metric, aggfunc="mean"
        ).reindex(columns=METHODS)
        logger.info("[%s]\n%s", metric, pivot.round(2).to_string())

    first_in_scope_customer = next(
        customer
        for customer, assessment in zip(population.customers, assessments)
        if assessment["status"] != "out_of_scope"
    )
    logger.info(
        "Invariant: weather unchanged after all runs: %s",
        bool(np.array_equal(snapshot, first_in_scope_customer.income)),
    )

    return outcomes_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
