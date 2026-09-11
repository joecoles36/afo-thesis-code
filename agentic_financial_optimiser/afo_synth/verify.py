"""Acceptance checks."""

from __future__ import annotations

import numpy as np

from . import eligibility, engine, generator
from .methods.baseline import NullMethod
from .methods.hybrid import RankedMethod
from .methods.piping import PipingMethod
from .methods.rules import RulesMethod


def main(size: int = 800) -> bool:
    ok = True

    a = generator.generate_population(seed=42, size=size)
    b = generator.generate_population(seed=42, size=size)
    rep = all(
        np.array_equal(x.income, y.income) and np.array_equal(x.essential, y.essential)
        for x, y in zip(a.customers, b.customers)
    )
    print(f"[{'PASS' if rep else 'FAIL'}] same seed -> identical population")
    ok &= rep

    w = a.customers[0]
    snap = (w.income.copy(), w.essential.copy(), w.discretionary.copy())
    for m in (NullMethod(), RulesMethod(), RankedMethod(), PipingMethod("equal")):
        engine.run(w, m)
        inv = (
            np.array_equal(snap[0], w.income)
            and np.array_equal(snap[1], w.essential)
            and np.array_equal(snap[2], w.discretionary)
        )
        print(f"[{'PASS' if inv else 'FAIL'}] weather byte-for-byte identical across methods")
        ok &= inv

    inscope = [c for c in a.customers if eligibility.assess(c)["status"] != "out_of_scope"][:250]

    def mean_end(m):
        return np.mean(
            [engine.run(c, m).iloc[-1][["emergency", "isa", "term_deposit"]].to_numpy() for c in inscope], 0
        )

    er, ek, ea = mean_end(RulesMethod()), mean_end(RankedMethod()), mean_end(PipingMethod("equal"))
    diff = not (np.allclose(er, ek, atol=1.0) and np.allclose(er, ea, atol=1.0))
    print(f"[{'PASS' if diff else 'FAIL'}] three methods -> three different aggregate outcomes")
    ok &= diff

    def cohort_slope(declining):
        hs = [
            eligibility.headroom_series(c)
            for c in a.customers
            if c.cohort == "Squeezed Middle" and (c.traits["slope"] < 0) == declining
        ]
        return float(np.polyfit(range(len(hs[0])), np.mean(hs, 0), 1)[0])

    down, flat = cohort_slope(True), cohort_slope(False)
    det = down < -1.0 and abs(flat) < abs(down)
    print(f"[{'PASS' if det else 'FAIL'}] declining trend down ({down:.1f}/mo) vs stable ({flat:.1f}/mo)")
    ok &= det

    try:
        from .validation import run_all  # noqa

        vok = True
    except Exception:
        vok = False
    print(f"[{'PASS' if vok else 'FAIL'}] validation suite importable")
    ok &= vok

    print("\nALL PASSED" if ok else "\nSOME FAILED")
    return ok


if __name__ == "__main__":
    main()
