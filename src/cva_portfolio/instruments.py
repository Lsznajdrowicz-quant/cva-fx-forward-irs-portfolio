from __future__ import annotations

import numpy as np


def price_fx_forward_paths(
    s_path: np.ndarray,
    eur_df: np.ndarray,
    pln_df: np.ndarray,
    notional_eur: float,
    strike: float,
    maturity_idx: int,
) -> np.ndarray:
    """Compute FX Forward MtM in PLN for each scenario and month."""

    n_sim, n_cols = s_path.shape
    v = np.zeros((n_sim, n_cols))
    for t in range(n_cols):
        df_eur_t_t = eur_df[maturity_idx] / eur_df[t]
        df_pln_t_t = pln_df[maturity_idx] / pln_df[t]
        v[:, t] = notional_eur * (s_path[:, t] * df_eur_t_t - strike * df_pln_t_t)

    # Academic project convention: initial MtM is set to zero for VM consistency.
    v[:, 0] = 0.0
    return v


def price_receiver_irs_paths(
    r_path: np.ndarray,
    s_path: np.ndarray,
    eur_df: np.ndarray,
    notional_eur: float,
    fixed_rate: float,
    fixed_pay_idx: list[int],
    float_pay_idx: list[int],
    float_reset_idx: list[int],
    tau_fixed: float,
    tau_float: float,
    maturity_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Receiver IRS MtM in EUR and PLN for each scenario and month."""

    n_sim, n_cols = r_path.shape
    fixings = np.zeros((n_sim, len(float_reset_idx)))
    for j, reset_idx in enumerate(float_reset_idx):
        fixings[:, j] = r_path[:, reset_idx]

    v_eur = np.zeros((n_sim, n_cols))
    for t in range(n_cols):
        pv_fixed = np.zeros(n_sim)
        for pay_idx in fixed_pay_idx:
            if pay_idx > t:
                df_t_pay = eur_df[pay_idx] / eur_df[t]
                pv_fixed += notional_eur * fixed_rate * tau_fixed * df_t_pay

        pv_float = np.zeros(n_sim)
        for j, pay_idx in enumerate(float_pay_idx):
            reset_idx = float_reset_idx[j]
            if reset_idx <= t < pay_idx:
                df_t_pay = eur_df[pay_idx] / eur_df[t]
                pv_float += notional_eur * fixings[:, j] * tau_float * df_t_pay
                break

        next_reset = next((reset_idx for reset_idx in float_reset_idx if reset_idx > t), None)
        if next_reset is not None:
            df_next = eur_df[next_reset] / eur_df[t]
            df_end = eur_df[maturity_idx] / eur_df[t]
            pv_float += notional_eur * (df_next - df_end)

        v_eur[:, t] = pv_fixed - pv_float

    v_eur[:, 0] = 0.0
    v_pln = v_eur * s_path
    return v_eur, v_pln
