"""Demo run."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config, eligibility, engine, generator, utils
from .methods.baseline import NullMethod
from .methods.hybrid import RankedMethod
from .methods.piping import PipingMethod
from .methods.rules import RulesMethod
from .validation.scoring_adapters import health_score_inputs, score_panel_monthly
from .validation.scoring_function import Scorer, assert_valid_health_score_inputs

METHODS = {
    "do_nothing": NullMethod(),
    "rules": RulesMethod(),
    "ranked": RankedMethod(),
    "allocation_equal": PipingMethod("equal"),
}

_SCORER = Scorer()
_ISA_ALLOWANCE = config.RUN["isa_allowance"]
_ISA_FRACTION = config.RUN["isa_fraction"]

# Cohort colours for trajectory plots
_COLOUR = config.RUN["cohort_colours"]

_METHOD_STYLE = config.RUN["method_styles"]

# Cap customers per cohort for monthly analysis
_MAX_PER_COHORT = config.RUN["max_per_cohort"]
_ENABLE_PLOTS = config.RUN.get("enable_plots", True)
_OUTPUT_DIR = Path(config.RUN.get("output_dir", "data/output"))
_RUNS_ROOT = Path(config.RUN.get("runs_root", "data/output/runs"))


def _param_hash(seed: int, size: int) -> str:
    """Stable short hash of run-driving parameters."""
    payload = {
        "seed": seed,
        "size": size,
        "run_savings_only": config.RUN,
        "scoring_function": config.SCORING,
        "n_months": config.N_MONTHS,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:8]


def _build_run_paths(seed: int, size: int) -> dict[str, Path | str]:
    """Create deterministic run identifier and output paths."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    p_hash = _param_hash(seed, size)
    run_name = str(config.RUN.get("run_name", "run_savings_only"))
    run_id = f"{ts}_{run_name}_{p_hash}"

    run_dir = _RUNS_ROOT / run_id
    plots_dir = run_dir / "plots"
    tables_dir = run_dir / "tables"
    logs_dir = run_dir / "logs"

    return {
        "run_id": run_id,
        "param_hash": p_hash,
        "run_dir": run_dir,
        "plots_dir": plots_dir,
        "tables_dir": tables_dir,
        "logs_dir": logs_dir,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _write_run_metadata(
    paths: dict[str, Path | str], seed: int, size: int
) -> None:
    """Persist manifest and parameter snapshot for reproducibility."""
    run_dir = paths["run_dir"]
    assert isinstance(run_dir, Path)

    manifest = {
        "run_id": paths["run_id"],
        "timestamp_utc": paths["timestamp_utc"],
        "param_hash": paths["param_hash"],
        "module": "afo_synth.run_savings_only",
        "command": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
        "python_version": platform.python_version(),
    }

    params = {
        "seed": seed,
        "size": size,
        "n_months": config.N_MONTHS,
        "run_savings_only": config.RUN,
        "scoring_function": config.SCORING,
        "engine": config.ENGINE,
    }

    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (run_dir / "params.json").write_text(
        json.dumps(params, indent=2), encoding="utf-8"
    )


def _append_runs_index(
    paths: dict[str, Path | str],
    seed: int,
    size: int,
    n_in_scope: int,
) -> None:
    """Append a one-line run summary to runs_index.csv."""
    if not bool(config.RUN.get("write_runs_index", True)):
        return

    row = pd.DataFrame(
        [
            {
                "run_id": paths["run_id"],
                "timestamp_utc": paths["timestamp_utc"],
                "param_hash": paths["param_hash"],
                "seed": seed,
                "size": size,
                "n_in_scope_customers": n_in_scope,
                "run_dir": str(paths["run_dir"]),
            }
        ]
    )

    index_path = _RUNS_ROOT / "runs_index.csv"
    if index_path.exists():
        existing = pd.read_csv(index_path)
        out = pd.concat([existing, row], ignore_index=True)
    else:
        out = row

    out.to_csv(index_path, index=False)


def outcome_metrics(panel: pd.DataFrame) -> dict:
    end = panel.iloc[-1]
    return dict(
        resilience_months=end.total_savings / max(1.0, end.essential),
        overdraft_rate=float(panel.overdraft.mean()),
        emergency=float(end.emergency),
        isa=float(end.isa),
        term_deposit=float(end.term_deposit),
        total_savings=float(end.emergency + end.isa + end.term_deposit),
        distress_paid=float(panel.distress_charge.sum()),
    )


def _heatmap(
    data_df: pd.DataFrame,
    title: str,
    filename: str,
    cmap: str = "RdYlGn",
) -> None:
    """Plot a heatmap with annotations and save to PNG."""
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        data_df.values,
        cmap=cmap,
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    ax.set_xticks(range(len(data_df.columns)))
    ax.set_yticks(range(len(data_df.index)))
    ax.set_xticklabels(
        data_df.columns,
        rotation=45,
        ha="right",
        fontsize=9,
    )
    ax.set_yticklabels(data_df.index, fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(
        "Score (0-1)",
        rotation=270,
        labelpad=20,
        fontsize=10,
    )

    plt.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"  -> {filename}")


def _export_health_scores(
    hdf: pd.DataFrame,
    metric_cols: list[str],
    category_cols: list[str],
    tables_dir: Path,
) -> None:
    """Export health score analysis to 4 CSV files.

    Exports:
    - scores_detailed.csv: every customer's scores
    - scores_summary_by_method.csv: pooled by method
    - scores_by_cohort_method.csv: by cohort x method
    - scores_categories_only.csv: categories + overall
    """
    print("\n" + "=" * 80)
    print("Exporting results to CSV...")
    print("=" * 80)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Detailed: every customer's scores
    detailed = hdf.copy()
    detailed_path = tables_dir / "scores_detailed.csv"
    detailed.to_csv(detailed_path, index=False)
    print(f"  -> {detailed_path} (every customer's scores)")

    # Summary by method: pooled metrics + categories + overall
    summary = (
        hdf.groupby("method")[metric_cols + category_cols + ["overall"]]
        .mean()
        .reindex(METHODS)
    )
    summary_path = tables_dir / "scores_summary_by_method.csv"
    summary.to_csv(summary_path)
    print(f"  -> {summary_path} (pooled by method)")

    # Per-cohort summary
    cohort_summary = (
        hdf.groupby(["cohort", "method"])[
            metric_cols + category_cols + ["overall"]
        ]
        .mean()
        .reset_index()
    )
    cohort_path = tables_dir / "scores_by_cohort_method.csv"
    cohort_summary.to_csv(cohort_path, index=False)
    print(f"  -> {cohort_path} (cohort x method)")

    # Category scores only (for quick comparison)
    cat_only = (
        hdf.groupby("method")[category_cols + ["overall"]]
        .mean()
        .reindex(METHODS)
    )
    category_path = tables_dir / "scores_categories_only.csv"
    cat_only.to_csv(category_path)
    print(f"  -> {category_path} (categories + overall)")


def main(seed: int = 42, size: int = 1500):
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    run_paths = _build_run_paths(seed=seed, size=size)
    run_dir = run_paths["run_dir"]
    plots_dir = run_paths["plots_dir"]
    tables_dir = run_paths["tables_dir"]
    logs_dir = run_paths["logs_dir"]

    assert isinstance(run_dir, Path)
    assert isinstance(plots_dir, Path)
    assert isinstance(tables_dir, Path)
    assert isinstance(logs_dir, Path)

    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    _write_run_metadata(run_paths, seed=seed, size=size)

    print("\n" + "=" * 80)
    print(f"Run ID: {run_paths['run_id']}")
    print(f"Run outputs: {run_dir}")
    print("=" * 80)

    pop = generator.generate_population(seed=seed, size=size)
    assessed = [
        dict(cohort=w.cohort, **eligibility.assess(w))
        for w in pop.customers
    ]
    edf = pd.DataFrame(assessed)

    print(
        "=== Deterioration screen "
        "(share of each cohort by headroom trend) ==="
    )
    ct = pd.crosstab(
        edf.cohort,
        edf.status,
        normalize="index",
    )
    order = [
        c
        for c in [
            "out_of_scope",
            "deteriorating",
            "stable",
            "improving",
        ]
        if c in ct.columns
    ]
    print(ct[order].round(2).to_string())

    rows = []
    health_rows = []
    snapshot = None

    for w, a in zip(pop.customers, assessed):
        if a["status"] == "out_of_scope":
            continue

        if snapshot is None:
            snapshot = w.income.copy()

        for mname, m in METHODS.items():
            panel = engine.run(w, m)
            rows.append(
                dict(
                    cohort=w.cohort,
                    method=mname,
                    **outcome_metrics(panel),
                )
            )

            inputs = health_score_inputs(w, panel)
            warnings = assert_valid_health_score_inputs(
                inputs,
                context=f"{w.cohort}/{mname}",
            )

            for warning in warnings:
                print(f"WARNING: {warning}")

            r = _SCORER.score(inputs)

            # Capture all 7 metric scores + 3 category scores + 1 overall
            health_rows.append(
                dict(
                    cohort=w.cohort,
                    method=mname,

                    # Metric scores (7)
                    days_in_overdraft=r.days_in_overdraft_score,
                    overdraft_fees=r.overdraft_fees_score,
                    fully_funded_emergency=r.fully_funded_emergency_score,
                    savings_consistency=r.savings_consistency_score,
                    isa_utilisation=r.isa_utilisation_score,
                    non_ia_balance=r.non_ia_savings_balance_score,
                    savings_yield=r.savings_yield_score,

                    # Category scores (3)
                    spending=r.spending_score,
                    emergency=r.emergency_score,
                    savings=r.savings_score,

                    # Overall (1)
                    overall=r.overall_score,
                )
            )

    df = pd.DataFrame(rows)

    cols = [
        "resilience_months",
        "overdraft_rate",
        "emergency",
        "isa",
        "term_deposit",
        "total_savings",
        "distress_paid",
    ]

    pooled = df.groupby("method")[cols].mean().reindex(METHODS)

    print(
        f"\n=== Outcome metrics, mean over in-scope customers "
        f"(n={df.cohort.count() // len(METHODS)}) ==="
    )
    print(pooled.round(2).to_string())

    print("\n=== Delta vs do-nothing (pooled) ===")
    print(
        (pooled - pooled.loc["do_nothing"])
        .round(2)
        .to_string()
    )

    print("\n=== Where the methods diverge: by cohort ===")
    for metric in [
        "resilience_months",
        "total_savings",
        "emergency",
    ]:
        piv = df.pivot_table(
            index="cohort",
            columns="method",
            values=metric,
            aggfunc="mean",
        ).reindex(columns=METHODS)

        print(f"\n[{metric}]")
        print(piv.round(2).to_string())

    # ============================================================
    # HEALTH SCORE ANALYSIS
    # ============================================================

    hdf = pd.DataFrame(health_rows)

    metric_cols = [
        "days_in_overdraft",
        "overdraft_fees",
        "fully_funded_emergency",
        "savings_consistency",
        "isa_utilisation",
        "non_ia_balance",
        "savings_yield",
    ]

    category_cols = [
        "spending",
        "emergency",
        "savings",
    ]

    print("\n" + "=" * 80)
    print("HEALTH SCORE ANALYSIS - Detailed breakdown by cohort and method")
    print("=" * 80)

    # Per-cohort summary
    for cohort in sorted(hdf.cohort.unique()):
        cohort_data = (
            hdf[hdf.cohort == cohort]
            .groupby("method")[
                metric_cols + category_cols + ["overall"]
            ]
            .mean()
            .reindex(METHODS)
        )

        print(f"\n=== {cohort} - All scores by method ===")
        print(cohort_data.round(3).to_string())

    # Overall pooled summary
    print("\n\n=== POOLED SUMMARY (all in-scope customers) ===")
    print("\n[Metric scores (0-1, higher is better)]")
    metric_pooled = hdf.groupby("method")[metric_cols].mean().reindex(METHODS)
    print(metric_pooled.round(3).to_string())

    print("\n[Category scores]")
    category_pooled = hdf.groupby("method")[category_cols].mean().reindex(METHODS)
    print(category_pooled.round(3).to_string())

    print("\n[Overall health score]")
    overall_pooled = hdf.groupby("method")[["overall"]].mean().reindex(METHODS)
    print(overall_pooled.round(3).to_string())
    print("\nDelta vs do-nothing:")
    print((overall_pooled - overall_pooled.loc["do_nothing"]).round(3).to_string())

    print("\n[Overall score by cohort x method]")
    cohort_overall = hdf.pivot_table(
        index="cohort",
        columns="method",
        values="overall",
        aggfunc="mean",
    ).reindex(columns=METHODS)
    print(cohort_overall.round(3).to_string())

    # ============================================================
    # VISUALIZATIONS - HEATMAPS
    # ============================================================

    if _ENABLE_PLOTS:
        print("\n\n" + "=" * 80)
        print("Generating visualizations (heatmaps)...")
        print("=" * 80)

        # Heatmap 1: Category scores by method
        cat_pivot = hdf.groupby("method")[category_cols].mean().reindex(METHODS).T
        _heatmap(
            cat_pivot,
            "Category Scores by Method (all customers)",
            str(plots_dir / "scores_categories.png"),
            cmap="RdYlGn",
        )

        # Heatmap 2: Overall scores by cohort x method
        overall_pivot = hdf.pivot_table(
            index="cohort",
            columns="method",
            values="overall",
            aggfunc="mean",
        ).reindex(columns=METHODS)
        _heatmap(
            overall_pivot,
            "Overall Health Score by Cohort and Method",
            str(plots_dir / "scores_overall_cohort.png"),
            cmap="RdYlGn",
        )

        # Heatmap 3: All metric scores by method
        metric_pivot = hdf.groupby("method")[metric_cols].mean().reindex(METHODS).T
        _heatmap(
            metric_pivot,
            "All 7 Metric Scores by Method (all customers)",
            str(plots_dir / "scores_metrics.png"),
            cmap="RdYlGn",
        )
    else:
        print("\n\n" + "=" * 80)
        print("Plot generation disabled (RUN['enable_plots']=False).")
        print("=" * 80)

    # ============================================================
    # CSV EXPORTS
    # ============================================================
    _export_health_scores(hdf, metric_cols, category_cols, tables_dir=tables_dir)

    # ============================================================
    # MONTHLY HEALTH SCORE TRAJECTORIES
    # ============================================================

    print("\n\n" + "=" * 80)
    print("MONTHLY HEALTH SCORE TRAJECTORIES (12-month evolution)")
    print("=" * 80)

    # Collect monthly scores per cohort and method
    monthly_all = []
    cohort_list = sorted(
        set(
            w.cohort
            for w, a in zip(pop.customers, assessed)
            if a["status"] != "out_of_scope"
        )
    )
    trajectory_data = {}  # For plotting: {cohort: {method: monthly_df}}

    for cohort in cohort_list:
        print(f"\n=== {cohort} - Overall score by month and method ===")
        trajectory_data[cohort] = {}
        cohort_customers = utils.inscope_customers(
            pop,
            cohort,
            limit=_MAX_PER_COHORT,
        )

        # Collect monthly data for this cohort
        cohort_monthly = []
        for mname in METHODS.keys():
            monthly_df = utils.mean_monthly_scores(
                cohort_customers,
                METHODS[mname],
                score_panel_monthly,
                ["overall"],
            )
            trajectory_data[cohort][mname] = monthly_df

            for month, row in monthly_df.iterrows():
                monthly_all.append(
                    dict(
                        cohort=cohort,
                        method=mname,
                        month=int(month),
                        overall=float(row["overall"]),
                    )
                )

            cohort_monthly.append(monthly_df)

        # Print table: months x methods
        cohort_pivot = pd.concat(
            cohort_monthly,
            axis=1,
            keys=METHODS.keys(),
        ).round(3)
        cohort_pivot.columns = METHODS.keys()
        print(cohort_pivot.to_string())

    # Export monthly data to CSV
    monthly_df_export = pd.DataFrame(monthly_all)
    monthly_traj_path = tables_dir / "scores_monthly_trajectory.csv"
    monthly_df_export.to_csv(monthly_traj_path, index=False)
    print("\n" + "=" * 80)
    print("Exporting monthly trajectories to CSV...")
    print("=" * 80)
    print(f"  -> {monthly_traj_path}")

    # Plot trajectories: one plot per cohort
    if _ENABLE_PLOTS:
        print("\nGenerating trajectory plots...")
        for cohort in cohort_list:
            fig, ax = plt.subplots(figsize=(10, 6))
            for mname in METHODS.keys():
                monthly_df = trajectory_data[cohort][mname]
                style = _METHOD_STYLE[mname]
                ax.plot(
                    monthly_df.index,
                    monthly_df["overall"],
                    label=mname,
                    color=_COLOUR[cohort],
                    **style,
                )

            ax.set_xlabel("Month", fontsize=11)
            ax.set_ylabel("Overall Health Score (0-1)", fontsize=11)
            ax.set_title(
                f"Health Score Trajectory - {cohort}",
                fontsize=12,
                fontweight="bold",
            )
            ax.set_xticks(range(1, 13))
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=10)
            fig.tight_layout()

            fname = (
                f"monthly_trajectory_"
                f"{cohort.lower().replace(' ', '_')}.png"
            )
            out_path = plots_dir / fname
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
            print(f"  -> {out_path}")

    first = next(
        w
        for w, a in zip(pop.customers, assessed)
        if a["status"] != "out_of_scope"
    )
    print(
        "\nInvariant: weather unchanged after all runs:",
        bool(np.array_equal(snapshot, first.income)),
    )

    n_in_scope = int(df.cohort.count() // len(METHODS)) if not df.empty else 0
    _append_runs_index(
        run_paths,
        seed=seed,
        size=size,
        n_in_scope=n_in_scope,
    )
    return df


if __name__ == "__main__":
    main()
