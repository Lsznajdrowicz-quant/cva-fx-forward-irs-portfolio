from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq


def bootstrap_hazard_rates(
    cds: pd.DataFrame,
    time_grid: np.ndarray,
    lgd: float,
    cds_zero_rate: float = 0.02,
    premium_freq_months: int = 3,
    dt: float = 1 / 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap piecewise-constant hazard rates from CDS spreads."""

    n_steps = len(time_grid) - 1
    cds_df = np.exp(-cds_zero_rate * time_grid)
    hazard_monthly = np.zeros(n_steps + 1)
    hazard_rows = []
    previous_end = 0

    for i in range(len(cds)):
        maturity_month = int(cds.loc[i, "MaturityMonths"])
        market_spread = float(cds.loc[i, "Spread_decimal"])
        start_month = previous_end + 1
        end_month = maturity_month

        def model_spread_for_hazard(h: float) -> float:
            temp_hazard = hazard_monthly.copy()
            temp_hazard[start_month : end_month + 1] = h

            survival = np.ones(n_steps + 1)
            for m in range(1, n_steps + 1):
                survival[m] = survival[m - 1] * np.exp(-temp_hazard[m] * dt)

            protection_leg = 0.0
            premium_basis = 0.0
            for m in range(1, maturity_month + 1):
                default_prob = survival[m - 1] - survival[m]
                protection_leg += lgd * cds_df[m] * default_prob
                if m % premium_freq_months == 0:
                    premium_basis += cds_df[m] * survival[m] * (premium_freq_months / 12)
            return protection_leg / premium_basis

        h_calibrated = brentq(lambda h: model_spread_for_hazard(h) - market_spread, 1e-8, 2.0)
        hazard_monthly[start_month : end_month + 1] = h_calibrated
        model_spread = model_spread_for_hazard(h_calibrated)

        hazard_rows.append(
            {
                "Tenor": cds.loc[i, "Tenor"],
                "StartMonth": start_month,
                "EndMonth": end_month,
                "MarketSpread_bps": cds.loc[i, "Spread_bps"],
                "ModelSpread_bps": model_spread * 10_000,
                "HazardRate": h_calibrated,
                "HazardRate_pct": h_calibrated * 100,
            }
        )
        previous_end = end_month

    survival = np.ones(n_steps + 1)
    for m in range(1, n_steps + 1):
        survival[m] = survival[m - 1] * np.exp(-hazard_monthly[m] * dt)

    default_prob = np.zeros(n_steps + 1)
    for m in range(1, n_steps + 1):
        default_prob[m] = survival[m - 1] - survival[m]

    hazard_df = pd.DataFrame(hazard_rows)
    credit_curve = pd.DataFrame(
        {
            "Month": np.arange(n_steps + 1),
            "TimeYears": time_grid,
            "HazardRate": hazard_monthly,
            "SurvivalProbability": survival,
            "DefaultProbabilityInMonth": default_prob,
            "CDS_DF": cds_df,
        }
    )
    return hazard_df, credit_curve
