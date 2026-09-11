"""Runs the validation suite."""

from __future__ import annotations

from .. import generator
from . import dependence, discriminator, marginals, privacy, trajectories, tstr


def main(seed: int = 42, size: int = 1500, plot_path: str = "validation_trajectories.png"):
    pop = generator.generate_population(seed, size)
    m = marginals.evaluate(pop.traits)
    print("=== Marginal fidelity (worst 6 by KS) ===")
    print(m.sort_values("KS", ascending=False).head(6).round(3).to_string(index=False))
    print(
        "  mean KS:",
        round(float(m.KS.mean()), 3),
        "| mean W1_norm:",
        round(float(m.W1_norm.mean()), 3),
    )
    print("\n=== Dependence fidelity ===")
    print({k: round(v, 3) for k, v in dependence.evaluate(pop.traits).items()})
    print("\n=== Discriminator (two-sample AUC) ===")
    print(
        {
            k: (round(v, 3) if isinstance(v, float) else v)
            for k, v in discriminator.evaluate(size=size).items()
        }
    )
    print("\n=== Privacy (distance to closest record) ===")
    print({k: round(v, 3) for k, v in privacy.evaluate(size=size).items()})
    print("\n=== TSTR ===")
    print(tstr.evaluate())
    p = trajectories.plot(pop=pop, path=plot_path)
    print("\nSaved trajectory plot:", p)


if __name__ == "__main__":
    main()
