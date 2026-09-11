"""Cohort definitions and model parameters."""

from __future__ import annotations

SEED: int = 42
N_MONTHS: int = 12
START_MONTH: int = 7
POPULATION_SIZE: int = 1500

# Target size of the emergency fund, expressed in months of regular outgoings.
EMERGENCY_TARGET_MONTHS: float = 3.0

# Liquidity buffer: current-account floor for the safety stage's low-balance top-up.
LIQUIDITY_BUFFER: float = 200.0

# Minimum surplus (GBP) required before an allocation method acts
SURPLUS_THRESHOLD: float = 1e-9

# Rules waterfall: after the emergency buffer, share of the remainder to the ISA
# (rest goes to the fixed-term deposit).
ISA_SHARE_OF_REMAINDER: float = 0.6

# Eligibility trend thresholds, expressed as a fraction of monthly income.
DETERIORATION_TREND: float = -0.01  # headroom falling faster than 1% of monthly income a month = deteriorating
IMPROVEMENT_TREND: float = 0.01  # rising faster than that = improving
CONSISTENCY_MIN: float = 0.5  # positive headroom in at least half the months to count as consistent

# Allocation policies for the piping method: weights across goals per policy name.
POLICIES: dict = {
    "equal": {"emergency": 1 / 3, "isa": 1 / 3, "term_deposit": 1 / 3},  # neutral control
    "resilience_first": {"emergency": 0.50, "isa": 0.30, "term_deposit": 0.20},  # protect the buffer first
    "progress_savings": {"emergency": 0.20, "isa": 0.50, "term_deposit": 0.30},  # lean into savings growth
}

# The three real savings products a customer can hold. Order is fixed: the
# opening savings balance is split across these, in this order, by each cohort's
# savings_split below.
PRODUCTS: tuple[str, ...] = ("instant_access", "isa", "term_deposit")

# Annual rate uplift per product, added to a cohort's base savings_apr. Instant
# access is the base (most liquid, lowest rate). ISA and term deposit pay more in
# exchange for less flexibility. Placeholder spreads and a calibration target.
PRODUCT_APR_SPREAD: dict = {
    "instant_access": 0.000,
    "isa": 0.005,
    "term_deposit": 0.010,
}

COPULA_VARS: tuple[str, ...] = (
    "income",  # persistent net monthly income level
    "essential_ratio",  # share of income that is non-negotiable spend
    "discretionary_propensity",  # tendency to discretionary spend, as share of income
    "savings_buffer",  # OPENING savings balance
    "slope",  # circumstance drift (negative = deteriorating)
)

SPEARMAN_TARGET: list[list[float]] = [
    [1.00, -0.50, 0.20, 0.55, 0.30],  # income
    [-0.50, 1.00, -0.30, -0.40, -0.30],  # essential_ratio
    [0.20, -0.30, 1.00, -0.10, -0.10],  # discretionary_propensity
    [0.55, -0.40, -0.10, 1.00, 0.30],  # savings_buffer
    [0.30, -0.30, -0.10, 0.30, 1.00],  # slope
]

ENGINE: dict = {
    "arranged_overdraft_limit": -1500.0,  # floor of the arranged overdraft (GBP)
    "overdraft_ear": 0.39,  # effective annual rate on arranged overdraft
    "unauthorised_ear": 0.79,  # punitive rate once below the arranged limit
    "unauthorised_monthly_fee": 15.0,  # flat monthly fee while below the limit
}

SCORING: dict = {
    "overdraft_fees_cap": 100.0,  # GBP 100 fees -> score of 0.0
    "non_ia_savings_cap": 10_000.0,  # GBP 10,000 -> score of 1.0
    "savings_yield_cap": 0.05 / 12,  # 5% APR monthly equivalent -> score of 1.0
    "category_weights": {
        "spending": 0.3,
        "emergency": 0.3,
        "savings": 0.4,
    },
}

RUN: dict = {
    "isa_allowance": 20_000.0,  # Annual ISA allowance (GBP)
    "isa_fraction": {  # Fraction of cohort income allocated to ISA
        "Squeezed": 0.10,
        "Squeezed Middle": 0.35,
        "Mass Affluent": 0.70,
    },
    "cohort_colours": {  # Visualization colors
        "Squeezed": "#d1495b",
        "Squeezed Middle": "#edae49",
        "Mass Affluent": "#66a182",
    },
    "method_styles": {  # Line styles for trajectory plots
        "do_nothing": {"ls": ":", "lw": 1.5},
        "rules": {"ls": "--", "lw": 1.5},
        "ranked": {"ls": "-", "lw": 2.0},
        "allocation_equal": {"ls": "-.", "lw": 1.5},
    },
    "max_per_cohort": 50,  # Cap customers per cohort for monthly analysis
    "enable_plots": False,  # Toggle heatmap + trajectory PNG generation
    "output_dir": "data/output",  # Base output directory for generated artifacts
    "runs_root": "data/output/runs",  # Root for versioned run folders
    "run_name": "run_savings_only",  # Prefix used in run identifiers
    "write_runs_index": True,  # Append one-line summary per run to runs_index.csv
}

# Mean split of the opening savings balance across the three products, per cohort.
# These are placeholder priors (lower-income cohorts hold more in liquid instant
# access and less locked away, while affluent cohorts hold more in tax-efficient ISA and
# fixed-term deposits) and are calibration targets: replace with real-data shares
# once the team's high-level summary lands. Each split must sum to 1.0.
COHORTS: dict = {
    "Squeezed": {
        "share": 0.40,
        "marginals": {
            "income": {"dist": "lognormal", "mean": 1600.0, "cv": 0.25, "floor": 400.0},
            "essential_ratio": {"dist": "beta", "mean": 0.85, "concentration": 30.0},
            "discretionary_propensity": {"dist": "gamma", "mean": 0.18, "shape": 4.0},
            "savings_buffer": {"dist": "lognormal_zi", "median": 300.0, "sigma": 0.90, "p_zero": 0.20},
            "slope": {
                "dist": "mixture_normal",
                "p_declining": 0.45,
                "declining": {"mean": -0.012, "sd": 0.004},
                "stable": {"mean": 0.002, "sd": 0.003},
            },
        },
        "rates": {"savings_apr": {"mean": 0.020, "sd": 0.003}},
        "savings_split": {"instant_access": 0.80, "isa": 0.15, "term_deposit": 0.05},
        "dynamics": {
            "rho": 0.50,  # persistence of transitory deviations around the level
            "delta": 1.0,  # cohort multiplier on the per-customer slope
            "level_innovation_sd": 0.010,  # eta, as a fraction of the level
            "transitory_sd": {"income": 0.05, "essential": 0.03, "discretionary": 0.10},  # eps per stream
        },
        "seasonal": {"amplitude": 0.6, "december_bump": 0.15, "late_summer_bump": 0.08, "family_summer": False},
        "shocks": {
            "income_drop": {"p_monthly": 0.030, "drop_fraction_mean": 0.35, "duration_months": 3},
            "essential_cost": {"p_monthly": 0.040, "median_gbp": 900.0, "sigma": 0.50},
        },
        "opening_pca": {"income_mult_mean": 0.30, "sd_fraction": 0.40},
    },
    "Squeezed Middle": {
        "share": 0.35,
        "marginals": {
            "income": {"dist": "lognormal", "mean": 2600.0, "cv": 0.25, "floor": 700.0},
            "essential_ratio": {"dist": "beta", "mean": 0.70, "concentration": 30.0},
            "discretionary_propensity": {"dist": "gamma", "mean": 0.18, "shape": 4.0},
            "savings_buffer": {"dist": "lognormal_zi", "median": 2500.0, "sigma": 0.90, "p_zero": 0.08},
            "slope": {
                "dist": "mixture_normal",
                "p_declining": 0.25,
                "declining": {"mean": -0.010, "sd": 0.004},
                "stable": {"mean": 0.003, "sd": 0.003},
            },
        },
        "rates": {"savings_apr": {"mean": 0.025, "sd": 0.003}},
        "savings_split": {"instant_access": 0.55, "isa": 0.30, "term_deposit": 0.15},
        "dynamics": {
            "rho": 0.50,
            "delta": 1.0,
            "level_innovation_sd": 0.010,
            "transitory_sd": {"income": 0.05, "essential": 0.03, "discretionary": 0.10},
        },
        "seasonal": {"amplitude": 1.0, "december_bump": 0.15, "late_summer_bump": 0.10, "family_summer": True},
        "shocks": {
            "income_drop": {"p_monthly": 0.020, "drop_fraction_mean": 0.35, "duration_months": 3},
            "essential_cost": {"p_monthly": 0.030, "median_gbp": 900.0, "sigma": 0.50},
        },
        "opening_pca": {"income_mult_mean": 0.40, "sd_fraction": 0.40},
    },
    "Mass Affluent": {
        "share": 0.25,
        "marginals": {
            "income": {"dist": "lognormal", "mean": 4500.0, "cv": 0.30, "floor": 1500.0},
            "essential_ratio": {"dist": "beta", "mean": 0.55, "concentration": 25.0},
            "discretionary_propensity": {"dist": "gamma", "mean": 0.22, "shape": 4.0},
            "savings_buffer": {"dist": "lognormal_zi", "median": 12000.0, "sigma": 1.00, "p_zero": 0.02},
            "slope": {
                "dist": "mixture_normal",
                "p_declining": 0.05,
                "declining": {"mean": -0.010, "sd": 0.004},
                "stable": {"mean": 0.004, "sd": 0.003},
            },
        },
        "rates": {"savings_apr": {"mean": 0.030, "sd": 0.003}},
        "savings_split": {"instant_access": 0.40, "isa": 0.40, "term_deposit": 0.20},
        "dynamics": {
            "rho": 0.40,
            "delta": 1.0,
            "level_innovation_sd": 0.010,
            "transitory_sd": {"income": 0.05, "essential": 0.03, "discretionary": 0.10},
        },
        "seasonal": {"amplitude": 1.2, "december_bump": 0.18, "late_summer_bump": 0.08, "family_summer": False},
        "shocks": {
            "income_drop": {"p_monthly": 0.012, "drop_fraction_mean": 0.35, "duration_months": 3},
            "essential_cost": {"p_monthly": 0.025, "median_gbp": 900.0, "sigma": 0.50},
        },
        "opening_pca": {"income_mult_mean": 0.60, "sd_fraction": 0.40},
    },
}
