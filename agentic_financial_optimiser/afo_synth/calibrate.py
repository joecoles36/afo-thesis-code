"""Calibration hooks for fitting against a real sample, and applying fitted params."""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import config


def fit_marginals(sample, cohort_col: str = "cohort") -> dict:
    """Refit per-cohort trait marginals from a real sample. TODO."""
    raise NotImplementedError


def fit_copula(sample) -> list:
    """Estimate the trait rank-correlation matrix from a real sample. TODO."""
    raise NotImplementedError


def fit_dynamics(panel) -> dict:
    """Estimate persistence, innovation variance and seasonality from a real panel. TODO."""
    raise NotImplementedError


def _deep_update(base: dict, overrides: dict) -> None:
    """Recursively overlay overrides onto base, so a partial file only changes what it names."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def load_params(path: str) -> dict:
    """Overlay calibration params from a JSON file onto the config defaults.

    Partial files are fine; only the keys present get overridden. Returns the parsed dict.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if "cohorts" in data:
        _deep_update(config.COHORTS, data["cohorts"])
    if "product_apr_spread" in data:
        _deep_update(config.PRODUCT_APR_SPREAD, data["product_apr_spread"])
    return data


def apply_env_calibration() -> dict | None:
    """If AFO_CALIBRATION points at an existing file, overlay it onto config. No-op otherwise."""
    path = os.environ.get("AFO_CALIBRATION")
    if path and Path(path).is_file():
        return load_params(path)
    return None
