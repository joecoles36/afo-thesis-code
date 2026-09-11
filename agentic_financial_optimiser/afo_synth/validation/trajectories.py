"""Cohort trajectory plots."""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .. import config, eligibility, engine, generator
from ..methods.baseline import NullMethod
from ..methods.piping import PipingMethod

COL = {
    "Squeezed": "#d1495b",
    "Squeezed Middle": "#edae49",
    "Mass Affluent": "#66a182",
}


def plot(
    pop=None,
    seed: int = 42,
    size: int = 1500,
    path: str = "validation_trajectories.png",
):
    if pop is None:
        pop = generator.generate_population(seed, size)

    n = config.N_MONTHS
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for cohort in config.COHORTS:
        for declining, ls in [(True, "--"), (False, "-")]:
            hs = [
                eligibility.headroom_series(w)
                for w in pop.customers
                if w.cohort == cohort and (w.traits["slope"] < 0) == declining
            ]
            if hs:
                ax1.plot(
                    range(n),
                    np.mean(hs, 0),
                    ls,
                    color=COL[cohort],
                    lw=2,
                    label=f"{cohort} ({'declining' if declining else 'stable'})",
                )

    ax1.axhline(0, color="#333", lw=1, ls=":")
    ax1.set_title("Monthly headroom: declining (dashed) vs stable (solid)")
    ax1.set_xlabel("month")
    ax1.set_ylabel("headroom (GBP)")
    ax1.legend(fontsize=7, frameon=False)

    inscope = [
        w
        for w in pop.customers
        if eligibility.assess(w)["status"] != "out_of_scope"
    ]

    for mname, m, c in [
        ("do nothing", NullMethod(), "#8d99ae"),
        ("allocation (equal)", PipingMethod("equal"), "#2a9d8f"),
    ]:
        liq = np.mean(
            [engine.run(w, m)["total_savings"].values for w in inscope],
            0,
        )
        ax2.plot(
            range(n),
            liq,
            "-o",
            ms=3,
            color=c,
            lw=2,
            label=mname,
        )

    ax2.set_title("Mean total savings (in-scope)")
    ax2.set_xlabel("month")
    ax2.set_ylabel("GBP")
    ax2.legend(fontsize=8, frameon=False)

    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
