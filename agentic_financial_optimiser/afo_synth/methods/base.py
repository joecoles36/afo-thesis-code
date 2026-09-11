"""Decision-method interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Action:
    """A month's deliberate money moves, in GBP, returned by a decision method.
    The engine applies these as transfers into the three savings products and
    validates them against the available surplus. Instant access doubles as the
    liquid emergency buffer, so its top-up is named to_emergency."""

    to_emergency: float = 0.0  # into instant access, the liquid safety buffer
    to_isa: float = 0.0
    to_term_deposit: float = 0.0
    notes: dict = field(default_factory=dict)


class Method(Protocol):
    name: str

    def decide(self, state, surplus: float) -> Action: ...
