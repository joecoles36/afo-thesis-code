"""Rules waterfall."""

from __future__ import annotations

from ..config import EMERGENCY_TARGET_MONTHS, ISA_SHARE_OF_REMAINDER, SURPLUS_THRESHOLD
from .base import Action


class RulesMethod:
    name = "rules"

    def decide(self, state, surplus: float) -> Action:
        if surplus <= SURPLUS_THRESHOLD:
            return Action()

        action = Action()
        remaining = surplus

        # 1. top up the emergency buffer toward its target
        emergency_gap = max(
            0.0,
            EMERGENCY_TARGET_MONTHS * state.regular_outgoings - state.emergency,
        )
        to_emergency = min(remaining, emergency_gap)
        action.to_emergency = to_emergency
        remaining -= to_emergency

        # 2. split what is left between the flexible ISA and the higher-rate fixed term
        action.to_isa = remaining * ISA_SHARE_OF_REMAINDER
        action.to_term_deposit = remaining * (1.0 - ISA_SHARE_OF_REMAINDER)

        return action
