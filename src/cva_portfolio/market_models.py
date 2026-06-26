from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def calibrate_ir_abm(df_ir: pd.DataFrame, dt_hist: float = 1 / 12) -> dict:
    """Calibrate an arithmetic Brownian motion to monthly EURIBOR changes."""

    df = df_ir.copy()
    df["dr"] = df["EURIBOR3M"].diff()
    dr_series = df["dr"].dropna()
    mu_raw = dr_series.mean() / dt_hist
    sigma = dr_series.std(ddof=1) / np.sqrt(dt_hist)
    t_stat, p_value = stats.ttest_1samp(dr_series, 0)

    # Conservative modelling choice used in the academic project: non-significant drift is set to zero.
    mu_used = 0.0
    r_init = float(df["EURIBOR3M"].iloc[-1])

    return {
        "df_ir": df,
        "mu_raw": float(mu_raw),
        "mu_used": float(mu_used),
        "sigma": float(sigma),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "r_init": r_init,
    }


def estimate_ir_fx_correlation(df_ir: pd.DataFrame, df_fx: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Estimate Pearson correlation between EURIBOR changes and EUR/PLN log-returns."""

    df_corr = pd.merge(
        df_ir[["YearMonth", "Date", "EURIBOR3M", "dr"]],
        df_fx[["YearMonth", "EURPLN", "fx_log_return"]],
        on="YearMonth",
        how="inner",
    ).dropna().reset_index(drop=True)

    rho_empirical, p_value = stats.pearsonr(df_corr["fx_log_return"], df_corr["dr"])
    rho_used = 0.0 if p_value > alpha else float(rho_empirical)

    corr_matrix = np.array([[1.0, rho_used], [rho_used, 1.0]])
    cholesky_matrix = np.linalg.cholesky(corr_matrix)

    return {
        "df_corr": df_corr,
        "rho_empirical": float(rho_empirical),
        "p_value": float(p_value),
        "rho_used": float(rho_used),
        "corr_matrix": corr_matrix,
        "cholesky_matrix": cholesky_matrix,
    }


def simulate_market_paths(
    n_sim: int,
    n_steps: int,
    dt: float,
    s0: float,
    r_init: float,
    mu_fx: float,
    sigma_fx: float,
    mu_r: float,
    sigma_r: float,
    cholesky_matrix: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate correlated EUR/PLN GBM and EURIBOR ABM paths."""

    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, (n_sim, n_steps, 2))
    z_corr = z @ cholesky_matrix.T
    z_ir = z_corr[:, :, 0]
    z_fx = z_corr[:, :, 1]

    s_path = np.zeros((n_sim, n_steps + 1))
    s_path[:, 0] = s0
    for t in range(1, n_steps + 1):
        s_path[:, t] = s_path[:, t - 1] * np.exp(
            (mu_fx - 0.5 * sigma_fx**2) * dt + sigma_fx * np.sqrt(dt) * z_fx[:, t - 1]
        )

    r_path = np.zeros((n_sim, n_steps + 1))
    r_path[:, 0] = r_init
    for t in range(1, n_steps + 1):
        r_path[:, t] = r_path[:, t - 1] + mu_r * dt + sigma_r * np.sqrt(dt) * z_ir[:, t - 1]

    return s_path, r_path
