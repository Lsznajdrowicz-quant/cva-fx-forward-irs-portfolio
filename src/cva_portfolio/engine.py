from __future__ import annotations

from pathlib import Path
import pandas as pd

from .config import ModelConfig
from .data import load_market_curves, load_cds_spreads, download_euribor_3m_from_ecb, download_eurpln_from_yfinance
from .credit_curve import bootstrap_hazard_rates
from .market_models import calibrate_ir_abm, estimate_ir_fx_correlation, simulate_market_paths
from .instruments import price_fx_forward_paths, price_receiver_irs_paths
from .exposure import expected_exposure, pfe, apply_monthly_variation_margin
from .cva import calculate_cva
from .plots import save_ee_plot, save_cva_plot


def run_cva_analysis(
    market_data_path: str | Path = "data/Market_data.xlsx",
    output_dir: str | Path | None = "results",
    config: ModelConfig | None = None,
) -> dict:
    """Run the full CVA workflow for the FX Forward + Receiver IRS portfolio."""

    cfg = config or ModelConfig()
    market_data_path = Path(market_data_path)
    output_dir = Path(output_dir) if output_dir is not None else None

    curves = load_market_curves(market_data_path, cfg.months)
    cds = load_cds_spreads(market_data_path, max_months=cfg.n_steps)
    hazard_df, credit_curve = bootstrap_hazard_rates(
        cds=cds,
        time_grid=cfg.time_grid,
        lgd=cfg.lgd,
        cds_zero_rate=cfg.cds_zero_rate,
        premium_freq_months=cfg.premium_freq_months,
        dt=cfg.dt,
    )

    # Historical market factors are downloaded from public sources, as in the original project:
    # - EURIBOR 3M from the ECB Data API,
    # - EUR/PLN from Yahoo Finance via yfinance.
    df_ir = download_euribor_3m_from_ecb(cfg.historical_start_date, cfg.valuation_date)
    df_fx = download_eurpln_from_yfinance(cfg.historical_start_date, cfg.valuation_date)
    ir_calibration = calibrate_ir_abm(df_ir, dt_hist=cfg.dt)
    correlation = estimate_ir_fx_correlation(ir_calibration["df_ir"], df_fx, alpha=cfg.alpha)

    s0 = float(curves["eurpln_curve"][0])
    mu_fx = cfg.r_pln_flat - cfg.r_eur_flat
    s_path, r_path = simulate_market_paths(
        n_sim=cfg.n_sim,
        n_steps=cfg.n_steps,
        dt=cfg.dt,
        s0=s0,
        r_init=ir_calibration["r_init"],
        mu_fx=mu_fx,
        sigma_fx=cfg.sigma_fx,
        mu_r=ir_calibration["mu_used"],
        sigma_r=ir_calibration["sigma"],
        cholesky_matrix=correlation["cholesky_matrix"],
        seed=cfg.seed,
    )

    v_fxf_pln = price_fx_forward_paths(
        s_path=s_path,
        eur_df=curves["eur_df"],
        pln_df=curves["pln_df"],
        notional_eur=cfg.notional_fxf,
        strike=cfg.strike_fxf,
        maturity_idx=cfg.n_steps,
    )

    _, v_irs_pln = price_receiver_irs_paths(
        r_path=r_path,
        s_path=s_path,
        eur_df=curves["eur_df"],
        notional_eur=cfg.notional_irs,
        fixed_rate=cfg.k_fixed,
        fixed_pay_idx=cfg.fixed_pay_idx,
        float_pay_idx=cfg.float_pay_idx,
        float_reset_idx=cfg.float_reset_idx,
        tau_fixed=cfg.tau_fixed,
        tau_float=cfg.tau_float,
        maturity_idx=cfg.n_steps,
    )

    v_portfolio_pln = v_fxf_pln + v_irs_pln
    v_portfolio_vm_pln, vm_pln = apply_monthly_variation_margin(v_portfolio_pln)

    ee_fxf = expected_exposure(v_fxf_pln)
    ee_irs = expected_exposure(v_irs_pln)
    ee_portfolio = expected_exposure(v_portfolio_pln)
    ee_portfolio_vm = expected_exposure(v_portfolio_vm_pln)

    pfe_fxf = pfe(v_fxf_pln)
    pfe_irs = pfe(v_irs_pln)
    pfe_portfolio = pfe(v_portfolio_pln)
    pfe_portfolio_vm = pfe(v_portfolio_vm_pln)

    default_prob = credit_curve["DefaultProbabilityInMonth"].to_numpy()
    pln_df = curves["pln_df"]

    cva_fxf = calculate_cva(ee_fxf, pln_df, default_prob, cfg.lgd)
    cva_irs = calculate_cva(ee_irs, pln_df, default_prob, cfg.lgd)
    cva_portfolio = calculate_cva(ee_portfolio, pln_df, default_prob, cfg.lgd)
    cva_portfolio_vm = calculate_cva(ee_portfolio_vm, pln_df, default_prob, cfg.lgd)
    reduction_vm = 1 - cva_portfolio_vm / cva_portfolio

    cva_results = pd.DataFrame(
        {
            "Position": ["FX Forward", "Receiver IRS", "Portfolio without VM", "Portfolio with VM"],
            "CVA_PLN": [cva_fxf, cva_irs, cva_portfolio, cva_portfolio_vm],
        }
    )

    ee_profiles = pd.DataFrame(
        {
            "Month": cfg.months,
            "TimeYears": cfg.time_grid,
            "EE_FXForward_PLN": ee_fxf,
            "PFE95_FXForward_PLN": pfe_fxf,
            "EE_IRS_PLN": ee_irs,
            "PFE95_IRS_PLN": pfe_irs,
            "EE_Portfolio_PLN": ee_portfolio,
            "PFE95_Portfolio_PLN": pfe_portfolio,
            "EE_Portfolio_VM_PLN": ee_portfolio_vm,
            "PFE95_Portfolio_VM_PLN": pfe_portfolio_vm,
        }
    )

    # Basic sanity checks for model validation.
    assert s_path.shape == (cfg.n_sim, cfg.n_steps + 1)
    assert r_path.shape == (cfg.n_sim, cfg.n_steps + 1)
    assert (credit_curve["DefaultProbabilityInMonth"].iloc[1:] >= -1e-12).all()
    assert credit_curve["SurvivalProbability"].is_monotonic_decreasing
    assert cva_portfolio_vm < cva_portfolio

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        cva_results.to_csv(output_dir / "cva_results.csv", index=False)
        ee_profiles.to_csv(output_dir / "ee_profiles.csv", index=False)
        hazard_df.to_csv(output_dir / "hazard_rates.csv", index=False)
        credit_curve.to_csv(output_dir / "credit_curve.csv", index=False)
        save_ee_plot(ee_profiles, output_dir / "expected_exposure_profiles.png")
        save_cva_plot(cva_results, output_dir / "cva_comparison.png")

    return {
        "config": cfg,
        "market_curves": curves,
        "cds": cds,
        "hazard_rates": hazard_df,
        "credit_curve": credit_curve,
        "ir_calibration": ir_calibration,
        "correlation": correlation,
        "s_path": s_path,
        "r_path": r_path,
        "v_fxf_pln": v_fxf_pln,
        "v_irs_pln": v_irs_pln,
        "v_portfolio_pln": v_portfolio_pln,
        "v_portfolio_vm_pln": v_portfolio_vm_pln,
        "vm_pln": vm_pln,
        "ee_profiles": ee_profiles,
        "cva_results": cva_results,
        "reduction_vm": reduction_vm,
    }
