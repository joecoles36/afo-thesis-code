"""Eligibility funnel over the population."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import eligibility, generator

CAP_HEADROOM = 100.0  # avg monthly headroom threshold (GBP)
CAP_LIQUID = 3000.0  # total liquid balances threshold (GBP)


def assess_customer(w) -> dict:
    h = eligibility.headroom_series(w)
    avg_h = float(np.mean(h))
    liquid = float(w.opening_pca + w.opening_savings)
    status = eligibility.assess(w)["status"]
    return dict(
        cohort=w.cohort,
        avg_headroom=avg_h,
        liquid=liquid,
        has_capacity=(avg_h > CAP_HEADROOM) or (liquid > CAP_LIQUID),
        in_difficulty=(status == "out_of_scope"),
        status=status,
    )


def main(seed: int = 42, size: int = 1500, plot_path: str = "sample_funnel.png"):
    pop = generator.generate_population(seed, size)
    df = pd.DataFrame([assess_customer(w) for w in pop.customers])
    N = len(df)

    cap = df[df.has_capacity]
    elig = cap[~cap.in_difficulty]
    funnel = pd.DataFrame(
        [
            ("Full synthetic population", N),
            ("Complete data + main banked + low external footprint (assumed)", N),
            ("Financial capacity (avg headroom > £100 OR liquid > £3,000)", len(cap)),
            ("Not in persistent difficulty (drop out-of-scope)", len(elig)),
        ],
        columns=["rule", "customers"],
    )
    funnel["pct_of_start"] = (funnel.customers / N * 100).round(1)
    print("=== Sample-definition funnel ===")
    print(funnel.to_string(index=False))
    print(f"\nEligible for the MVP: {len(elig)} of {N} ({len(elig) / N * 100:.1f}%)")

    print("\n=== Eligible pool, cohort mix ===")
    print((elig.cohort.value_counts(normalize=True) * 100).round(1).astype(str).add(" %").to_string())
    print("\n=== Eligible pool, deterioration status ===")
    print((elig.status.value_counts(normalize=True) * 100).round(1).astype(str).add(" %").to_string())

    total_det = int((df.status == "deteriorating").sum())
    elig_det = int((elig.status == "deteriorating").sum())
    print(
        f"\nSliders (deteriorating but not yet in difficulty): {total_det} in the population, "
        f"{elig_det} survive into the eligible pool."
    )
    print("These are AFO's catch-early group. If this number collapses, the sample is too strict.")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = ["Full\npopulation", "Structural\n(assumed)", "Financial\ncapacity", "Not in\ndifficulty"]
        vals = list(funnel.customers)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(labels, vals, color=["#8d99ae", "#8d99ae", "#edae49", "#2a9d8f"])
        for i, v in enumerate(vals):
            ax.text(i, v, str(v), ha="center", va="bottom")
        ax.set_title("MVP sample-definition funnel (synthetic population)")
        ax.set_ylabel("customers")
        plt.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print("\nSaved funnel plot:", plot_path)
    except Exception as e:
        print("plot skipped:", e)

    return elig


if __name__ == "__main__":
    main()
