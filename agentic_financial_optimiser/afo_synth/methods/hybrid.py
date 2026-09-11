"""Ranked optimisation."""

from __future__ import annotations

from ..config import EMERGENCY_TARGET_MONTHS, SURPLUS_THRESHOLD
from .base import Action


class RankedMethod:
    name = "ranked"

    def decide(self, state, surplus: float) -> Action:
        if surplus <= SURPLUS_THRESHOLD:
            return Action()
        emergency_target = EMERGENCY_TARGET_MONTHS * state.regular_outgoings
        emergency_gap = max(0.0, emergency_target - state.emergency)
        scores = {
            "emergency": emergency_gap / max(1.0, emergency_target),  # further from target = more urgent
            "isa": 0.20,  # tax-efficient and flexible: modest baseline pull
            "term_deposit": 0.15,  # higher rate but locked: slightly lower pull
        }
        top_goal = max(scores, key=scores.get)
        action = Action()
        if top_goal == "emergency":
            action.to_emergency = min(surplus, emergency_gap)
        elif top_goal == "isa":
            action.to_isa = surplus
        else:
            action.to_term_deposit = surplus
        return action
