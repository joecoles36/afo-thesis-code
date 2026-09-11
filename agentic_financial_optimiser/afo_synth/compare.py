"""Approach and policy comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import eligible, engine
from .methods.baseline import NullMethod
from .methods.hybrid import RankedMethod
from .methods.piping import PipingMethod
from .methods.rules import RulesMethod

METHODS = {
    "do_nothing": NullMethod(),
    "rules": RulesMethod(),
    "ranked": RankedMethod(),
    "alloc_equal": PipingMethod("equal"),
    "alloc_resilience": PipingMethod("resilience_first"),
    "alloc_progress": PipingMethod("progress_savings"),
}


def _metrics(panel: pd.DataFrame, w) -> dict:
    end = panel.iloc[-1]
    start_nw = w.opening_pca + w.opening_ia + w.opening_isa + w.opening_td
    end_nw = end.current_account + end.emergency + end.isa + end.term_deposit
    predicted = (
        start_nw
        + (panel.income - panel.essential - panel.discretionary).sum()
        + panel.interest_earned.sum()
        - panel.distress_charge.sum()
    )
    return dict(
        resilience_months=end.emergency / max(1.0, end.essential),  # months the liquid buffer covers
        overdraft_rate=float(panel.overdraft.mean()),
        emergency=float(end.emergency),
        isa=float(end.isa),
        term_deposit=float(end.term_deposit),
        total_savings=float(end.emergency + end.isa + end.term_deposit),
        distress=float(panel.distress_charge.sum()),
        interest=float(panel.interest_earned.sum()),
        net_worth=float(end_nw),
        conservation_err=float(abs(end_nw - predicted)),
    )


def main(seed: int = 42, size: int = 1200, target_mix: dict | None = None):
    pop = eligible.generate_eligible_population(seed=seed, size=size, target_mix=target_mix)
    rows, weather_ok = [], []
    for w in pop.customers:
        snap = (w.income.copy(), w.essential.copy(), w.discretionary.copy())
        for mname, m in METHODS.items():
            rows.append(dict(cohort=w.cohort, method=mname, **_metrics(engine.run(w, m), w)))
        weather_ok.append(
            np.array_equal(snap[0], w.income)
            and np.array_equal(snap[1], w.essential)
            and np.array_equal(snap[2], w.discretionary)
        )

    df = pd.DataFrame(rows)
    cols = [
        "resilience_months",
        "overdraft_rate",
        "emergency",
        "isa",
        "term_deposit",
        "total_savings",
        "distress",
        "interest",
        "net_worth",
    ]
    pooled = df.groupby("method")[cols].mean().reindex(METHODS)

    mix = (pop.traits.cohort.value_counts(normalize=True) * 100).round(1).to_dict()
    print(f"Eligible-only population: {len(pop.customers)} customers, mix {mix}")
    print("\n=== Outcomes (mean over eligible customers) ===")
    print(pooled.round(2).to_string())
    print("\n=== Delta vs do-nothing ===")
    print((pooled - pooled.loc["do_nothing"]).round(2).to_string())
    print("\n=== Net-worth uplift vs do-nothing, by cohort ===")
    piv = df.pivot_table(
        index="cohort", columns="method", values="net_worth", aggfunc="mean"
    ).reindex(columns=METHODS)
    print(piv.sub(piv["do_nothing"], axis=0).round(2).to_string())

    print("\n=== Allocation policy contrast (product balances; mean) ===")
    print(
        pooled.loc[
            ["alloc_equal", "alloc_resilience", "alloc_progress"],
            ["emergency", "isa", "term_deposit", "resilience_months"],
        ]
        .round(2)
        .to_string()
    )

    print("\n=== Verification ===")
    active = [m for m in METHODS if m != "do_nothing"]
    checks = [
        ("weather byte-for-byte identical across methods", all(weather_ok)),
        (
            f"engine accounting conserves (max error {df.conservation_err.max():.2e})",
            df.conservation_err.max() < 1e-4,
        ),
        (
            "no negative product balances",
            bool(
                (df.emergency >= -1e-9).all()
                and (df.isa >= -1e-9).all()
                and (df.term_deposit >= -1e-9).all()
            ),
        ),
        (
            "active arms beat do-nothing on total savings",
            bool(
                (
                    pooled.loc[active, "total_savings"]
                    >= pooled.loc["do_nothing", "total_savings"] - 1e-6
                ).all()
            ),
        ),
        (
            "active arms beat do-nothing on net worth",
            bool(
                (
                    pooled.loc[active, "net_worth"]
                    >= pooled.loc["do_nothing", "net_worth"] - 1e-6
                ).all()
            ),
        ),
        ("arms genuinely diverge", float(pooled["net_worth"].std()) > 1e-6),
    ]

    ok = True
    for label, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {label}")
        ok &= passed
    print("\nALL VERIFIED" if ok else "\nSOME CHECKS FAILED")
    return df


if __name__ == "__main__":
    main()
