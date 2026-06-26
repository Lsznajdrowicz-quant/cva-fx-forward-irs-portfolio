from __future__ import annotations

import numpy as np


def calculate_cva(expected_exposure: np.ndarray, discount_factors: np.ndarray, default_probabilities: np.ndarray, lgd: float) -> float:
    """Discrete CVA calculation."""

    return float(lgd * np.sum(expected_exposure[1:] * discount_factors[1:] * default_probabilities[1:]))
