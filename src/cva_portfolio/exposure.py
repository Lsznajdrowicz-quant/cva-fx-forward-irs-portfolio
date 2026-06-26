from __future__ import annotations

import numpy as np


def expected_exposure(mtm: np.ndarray) -> np.ndarray:
    return np.maximum(mtm, 0).mean(axis=0)


def pfe(mtm: np.ndarray, percentile: float = 95) -> np.ndarray:
    return np.percentile(np.maximum(mtm, 0), percentile, axis=0)


def apply_monthly_variation_margin(portfolio_mtm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply monthly VM equal to previous month portfolio MtM."""

    vm = np.zeros_like(portfolio_mtm)
    vm[:, 1:] = portfolio_mtm[:, :-1]
    return portfolio_mtm - vm, vm
