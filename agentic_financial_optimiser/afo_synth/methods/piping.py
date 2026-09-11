"""Allocation optimisation and policies."""

from __future__ import annotations

from ..config import EMERGENCY_TARGET_MONTHS, POLICIES, SURPLUS_THRESHOLD
from .base import Action


def allocate_with_caps(amount: float, weights: dict, caps: dict):
    """Split amount across the goals by weight. When a goal hits its cap, drop it and
    re-pour its share across the rest. Returns (allocations, leftover)."""
    allocations = {goal: 0.0 for goal in weights}
    caps_remaining = dict(caps)
    active = {goal for goal in weights if weights[goal] > 0 and caps_remaining[goal] > 1e-9}
    remaining = amount

    while remaining > 1e-9 and active:
        total_weight = sum(weights[goal] for goal in active)
        if total_weight <= 0:
            break

        capped = []
        for goal in list(active):
            share = remaining * weights[goal] / total_weight
            if share >= caps_remaining[goal] - 1e-9:  # this pipe fills up
                allocations[goal] += caps_remaining[goal]
                remaining -= caps_remaining[goal]
                caps_remaining[goal] = 0.0
                capped.append(goal)

        if capped:
            for goal in capped:
                active.discard(goal)
            continue  # re-pour across the rest

        for goal in active:  # nobody capped: finish the split
            give = remaining * weights[goal] / total_weight
            allocations[goal] += give
            caps_remaining[goal] -= give
        remaining = 0.0

    return allocations, remaining


class PipingMethod:
    def __init__(self, policy: str = "equal"):
        self.policy = policy
        self.name = f"allocation[{policy}]"
        self.weights = POLICIES[policy]

    def decide(self, state, surplus: float) -> Action:
        if surplus <= SURPLUS_THRESHOLD:
            return Action()

        emergency_target = EMERGENCY_TARGET_MONTHS * state.regular_outgoings
        caps = {
            "emergency": max(0.0, emergency_target - state.emergency),
            "isa": float("inf"),  # ISA annual-allowance cap deferred (the tax-year layer)
            "term_deposit": float("inf"),
        }

        allocations, _ = allocate_with_caps(surplus, self.weights, caps)

        return Action(
            to_emergency=allocations["emergency"],
            to_isa=allocations["isa"],
            to_term_deposit=allocations["term_deposit"],
        )
