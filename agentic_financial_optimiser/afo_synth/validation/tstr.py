"""Train-on-synthetic, test-on-real (stub)."""

from __future__ import annotations


def evaluate(*args, **kwargs) -> dict:
    return dict(
        status="pending_real_data",
        note="Awaiting the 1% sample. Then: fit on synthetic, score on real, report the gap.",
    )
