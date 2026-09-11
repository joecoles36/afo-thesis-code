"""Scoring function version 2."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .. import config


# ---------------------------------------------------------------------------
# Normalisation constants
# ---------------------------------------------------------------------------

_DAYS_IN_MONTH_MAX: float = 31.0  # maximum calendar days in a month

# Tuneable parameters (defined in config.py)
_OVERDRAFT_FEES_CAP: float = config.SCORING["overdraft_fees_cap"]
_NON_IA_SAVINGS_CAP: float = config.SCORING["non_ia_savings_cap"]
_SAVINGS_YIELD_CAP: float = config.SCORING["savings_yield_cap"]
_CATEGORY_WEIGHTS: dict[str, float] = config.SCORING["category_weights"]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ScoreInputs:
    """Raw metrics for a single customer in a single period."""

    # Spending (grain: account_monthly)
    days_in_overdraft: float
    """Number of days in overdraft during the month."""
    overdraft_fees_total: float
    """Total overdraft charges (fees + interest) over the period, GBP."""

    # Emergency fund (grain: account_monthly)
    fully_funded_emergency_flag: float
    """1.0 if emergency balance >= 3 x regular monthly outgoings, else 0.0."""
    savings_consistency_flag: float
    """1.0 if net savings position is positive at month end, else 0.0."""

    # Savings + optimisation
    isa_utilisation: float
    """Cumulative ISA allowance used / annual allowance, in [0, 1].
    Pre-computed by the adapter as min(isa_inflow / allowance, 1).
    (grain: account_monthly_cumulative)"""
    non_ia_savings_balance: float
    """Total savings balance held outside an ISA wrapper, GBP.
    (grain: customer_monthly)"""
    total_savings_interest: float
    """Savings interest earned in the period across all products, GBP.
    For monthly scoring: this month's interest.
    For year-end scoring: mean monthly interest (total / n_months).
    (grain: customer_monthly or customer_monthly_mean)"""
    average_savings_balance: float
    """Average total savings balance over the period, GBP.
    For monthly scoring: same as total_savings_balance (point-in-time).
    For year-end scoring: mean of monthly total balances across the period.
    Used as the denominator for savings_yield_score.
    (grain: customer_monthly or customer_monthly_mean)"""
    total_savings_balance: float
    """Total savings balance across all open products at period end, GBP.
    Used for non_ia_savings_balance calculation in adapters.
    (grain: customer_monthly)"""


@dataclass
class ScoreResult:
    """Normalised scores for a single customer in a single period.

    All values are in [0, 1] where 1.0 represents the best possible outcome.
    """

    # --- Individual metric scores ---
    days_in_overdraft_score: float
    overdraft_fees_score: float
    fully_funded_emergency_score: float
    savings_consistency_score: float
    isa_utilisation_score: float
    non_ia_savings_balance_score: float
    savings_yield_score: float

    # --- Category scores ---
    spending_score: float
    emergency_score: float
    savings_score: float

    # --- Overall ---
    overall_score: float


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_health_score_inputs(
    inputs: ScoreInputs,
    raw_payload: dict | Sequence[tuple[str, object]] | None = None,
) -> tuple[bool, list, list]:
    """Validate ScoreInputs schema: required fields, types, and value ranges.

    Returns:
        (is_valid, errors, warnings) where is_valid indicates schema passed.
        errors and warnings are collected for the timing being.
        we can then check errors list and change to raise if needed.
    """

    errors, warnings = [], []

    # Duplicate check only for raw list/tuple payloads where duplicate keys are representable.
    if isinstance(raw_payload, Sequence) and not isinstance(
        raw_payload, (str, bytes, dict)
    ):
        seen: set[str] = set()
        duplicates: set[str] = set()
        for item in raw_payload:
            if (
                isinstance(item, tuple)
                and len(item) >= 1
                and isinstance(item[0], str)
            ):
                key = item[0]
                if key in seen:
                    duplicates.add(key)
                seen.add(key)
        if duplicates:
            errors.append(f"DUPLICATE FIELDS: {', '.join(sorted(duplicates))}")

    # Required fields check - all fields must be present and not None
    required_fields = [
        "days_in_overdraft",
        "overdraft_fees_total",
        "fully_funded_emergency_flag",
        "savings_consistency_flag",
        "isa_utilisation",
        "non_ia_savings_balance",
        "total_savings_interest",
        "average_savings_balance",
        "total_savings_balance",
    ]
    for field in required_fields:
        if not hasattr(inputs, field):
            errors.append(f"MISSING FIELD: {field}")
        elif getattr(inputs, field) is None:
            errors.append(f"NULL FIELD: {field}")

    # Type check - all fields must be float or int
    for field in required_fields:
        if hasattr(inputs, field):
            val = getattr(inputs, field)
            if not isinstance(val, (float, int)):
                errors.append(
                    f"WRONG TYPE for {field}: expected float/int, "
                    f"got {type(val).__name__}"
                )

    # Value range checks: warning-only (non-blocking for synthetic calibration stage)
    # Fields in [0, 1]: fully_funded_emergency_flag, savings_consistency_flag, isa_utilisation
    for field in [
        "fully_funded_emergency_flag",
        "savings_consistency_flag",
        "isa_utilisation",
    ]:
        if hasattr(inputs, field):
            val = getattr(inputs, field)
            if isinstance(val, (float, int)) and not (0.0 <= val <= 1.0):
                warnings.append(f"OUT OF RANGE [0, 1] for {field}: {val}")

    # Non-negative fields with soft warnings (print warning if negative)
    non_negative_fields = [
        "overdraft_fees_total",
        "non_ia_savings_balance",
        "total_savings_interest",
        "average_savings_balance",
        "total_savings_balance",
    ]
    for field in non_negative_fields:
        if hasattr(inputs, field):
            val = getattr(inputs, field)
            if isinstance(val, (float, int)) and val < 0.0:
                warnings.append(
                    f"NEGATIVE VALUE for {field}: {val} (expected >= 0)"
                )

    # days_in_overdraft soft warning if > 31 or < 0
    if hasattr(inputs, "days_in_overdraft"):
        val = getattr(inputs, "days_in_overdraft")
        if isinstance(val, (float, int)):
            if val < 0:
                warnings.append(f"NEGATIVE days_in_overdraft: {val}")
            elif val > 31:
                warnings.append(f"days_in_overdraft exceeds 31 days: {val}")

    return len(errors) == 0, errors, warnings


def assert_valid_health_score_inputs(
    inputs: ScoreInputs,
    context: str = "",
    raw_payload: dict | Sequence[tuple[str, object]] | None = None,
) -> list[str]:
    """Validate inputs and raise ValueError on hard schema failures.

    Returns warnings for optional caller-side handling.
    """

    is_valid, errors, warnings = validate_health_score_inputs(
        inputs, raw_payload=raw_payload
    )
    if not is_valid:
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}Schema validation failed: {'; '.join(errors)}"
        )
    return warnings


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class Scorer:
    """Calculates individual metric scores, category scores and an overall
    customer financial health score using the input field set.

    All scores are normalised to [0, 1] where 1.0 is the best outcome.
    Higher ``overall_score`` always indicates better customer financial health.
    """

    # -----------------------------------------------------------------------
    # Individual metric scorers
    # -----------------------------------------------------------------------

    @staticmethod
    def score_days_in_overdraft(days_in_overdraft: float) -> float:
        """Lower days in overdraft -> higher score.

        Normalised against 31 (the maximum possible days in a calendar month).

        score = max(0, 1 - days_in_overdraft / 31)
        """
        return max(
            0.0,
            1.0 - (days_in_overdraft / _DAYS_IN_MONTH_MAX),
        )

    @staticmethod
    def score_overdraft_fees(overdraft_fees_total: float) -> float:
        """Lower fees -> higher score. Fees of 0 -> 1.0, fees >= 100 -> 0.0.

        score = max(0, 1 - min(overdraft_fees_total / 100, 1))
        """
        return max(
            0.0,
            1.0 - min(overdraft_fees_total / _OVERDRAFT_FEES_CAP, 1.0),
        )

    @staticmethod
    def score_fully_funded_emergency(
        fully_funded_emergency_flag: float,
    ) -> float:
        """Emergency fund >= 3 x outgoings -> 1.0, otherwise 0.0."""
        return float(fully_funded_emergency_flag)

    @staticmethod
    def score_savings_consistency(
        savings_consistency_flag: float,
    ) -> float:
        """Net positive savings at month end -> 1.0, otherwise 0.0."""
        return float(savings_consistency_flag)

    @staticmethod
    def score_isa_utilisation(isa_utilisation: float) -> float:
        """Full ISA utilisation -> 1.0, no utilisation -> 0.0."""
        return float(isa_utilisation)

    @staticmethod
    def score_non_ia_savings_balance(
        non_ia_savings_balance: float,
    ) -> float:
        """Higher non-ISA savings balance -> higher score. Capped at 10,000.

        score = min(non_ia_savings_balance / 10_000, 1)
        """
        return min(
            non_ia_savings_balance / _NON_IA_SAVINGS_CAP,
            1.0,
        )

    @staticmethod
    def score_savings_yield(
        total_savings_interest: float,
        average_savings_balance: float,
    ) -> float:
        """Effective savings yield -> higher rate -> higher score.

        yield = total_savings_interest / average_savings_balance

        Normalised against a 5% APR monthly equivalent (0.004167).
        A customer earning 5% APR on their total portfolio scores 1.0.
        Returns 0.0 when average_savings_balance is zero.

        score = min((interest / avg_balance) / (0.05 / 12), 1)
        """
        if average_savings_balance <= 0.0:
            return 0.0
        return min(
            (total_savings_interest / average_savings_balance)
            / _SAVINGS_YIELD_CAP,
            1.0,
        )

    # -----------------------------------------------------------------------
    # Category scores
    # -----------------------------------------------------------------------

    @staticmethod
    def _spending_score(
        days_in_overdraft_score: float,
        overdraft_fees_score: float,
    ) -> float:
        """Simple average of the two overdraft-related metric scores."""
        return (days_in_overdraft_score + overdraft_fees_score) / 2.0

    @staticmethod
    def _emergency_score(
        fully_funded_emergency_score: float,
        savings_consistency_score: float,
    ) -> float:
        """Simple average of the two emergency-fund metric scores."""
        return (
            fully_funded_emergency_score + savings_consistency_score
        ) / 2.0

    @staticmethod
    def _savings_score(
        isa_utilisation_score: float,
        non_ia_savings_balance_score: float,
        savings_yield_score: float,
    ) -> float:
        """Simple average of the three savings-optimisation metric scores.

        Metrics:
        - isa_utilisation_score:      Are savings being placed in tax-efficient savings product?
        - non_ia_savings_balance_score: Does the customer have sufficient savings outside emergency funds?
        - savings_yield_score:        Is the customer earning a good rate on their portfolio?
        """
        return (
            isa_utilisation_score
            + non_ia_savings_balance_score
            + savings_yield_score
        ) / 3.0

    # -----------------------------------------------------------------------
    # Overall score
    # -----------------------------------------------------------------------

    @staticmethod
    def _overall_score(
        spending: float,
        emergency: float,
        savings: float,
    ) -> float:
        """Weighted average across the three category scores.

        overall = 0.3 * spending + 0.3 * emergency + 0.4 * savings
        """
        w = _CATEGORY_WEIGHTS
        return (
            w["spending"] * spending
            + w["emergency"] * emergency
            + w["savings"] * savings
        )

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    def score(self, inputs: ScoreInputs) -> ScoreResult:
        """Compute all metric, category and overall scores for one customer.

        Parameters
        ----------
        inputs:
            Pre-computed raw metrics for the customer / period.

        Returns
        -------
                ScoreResult
            Fully populated result with metric scores, category scores and
            the overall health score, all in [0, 1].
        """
        # Individual metric scores
        days_od_score = self.score_days_in_overdraft(inputs.days_in_overdraft)
        fees_score = self.score_overdraft_fees(inputs.overdraft_fees_total)
        fully_funded_score = self.score_fully_funded_emergency(
            inputs.fully_funded_emergency_flag
        )
        consistency_score = self.score_savings_consistency(
            inputs.savings_consistency_flag
        )
        isa_score = self.score_isa_utilisation(inputs.isa_utilisation)
        non_ia_score = self.score_non_ia_savings_balance(
            inputs.non_ia_savings_balance
        )
        yield_score = self.score_savings_yield(
            inputs.total_savings_interest,
            inputs.average_savings_balance,
        )

        # Category scores
        spending = self._spending_score(days_od_score, fees_score)
        emergency = self._emergency_score(
            fully_funded_score,
            consistency_score,
        )
        savings = self._savings_score(
            isa_score,
            non_ia_score,
            yield_score,
        )

        # Overall score
        overall = self._overall_score(spending, emergency, savings)

        return ScoreResult(
            days_in_overdraft_score=days_od_score,
            overdraft_fees_score=fees_score,
            fully_funded_emergency_score=fully_funded_score,
            savings_consistency_score=consistency_score,
            isa_utilisation_score=isa_score,
            non_ia_savings_balance_score=non_ia_score,
            savings_yield_score=yield_score,
            spending_score=spending,
            emergency_score=emergency,
            savings_score=savings,
            overall_score=overall,
        )
