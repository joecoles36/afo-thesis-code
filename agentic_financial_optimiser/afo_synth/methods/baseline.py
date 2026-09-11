"""Do-Nothing baseline."""

from __future__ import annotations

from .base import Action


class NullMethod:
    name = "do_nothing"

    def decide(self, state, surplus: float) -> Action:
        return Action()
