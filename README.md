# CVA for FX Forward and Receiver IRS Portfolio

This repository implements a Monte Carlo CVA engine for a derivatives portfolio consisting of:

- an **FX Forward** position: buy EUR / sell PLN,
- a **Receiver IRS** position in EUR.

The project estimates counterparty credit risk at both instrument and portfolio level and compares portfolio CVA **before and after monthly Variation Margin**.

## Project objective

The goal is to calculate **Credit Valuation Adjustment (CVA)** for a portfolio exposed to FX and interest-rate risk.

- CDS spreads and market curves are loaded from `data/Market_data.xlsx`,
- historical **EURIBOR 3M** data is downloaded from the **ECB Data API**,
- historical **EUR/PLN** data is downloaded from **Yahoo Finance** using `yfinance`,
- EUR/PLN is modelled with **Geometric Brownian Motion (GBM)**,
- EURIBOR 3M is modelled with **Arithmetic Brownian Motion (ABM)**,
- exposure profiles are estimated using **Monte Carlo simulation**,
- CVA is calculated using monthly default probabilities and PLN discount factors.

## Portfolio setup

| Instrument | Main assumptions |
|---|---|
| FX Forward | Buy EUR / sell PLN, notional 1,000,000 EUR, forward rate 4.4439 PLN/EUR, maturity 3 years |
| Receiver IRS | Notional 10,000,000 EUR, receive fixed 2.02%, pay EURIBOR 3M, maturity 3 years |

## Data sources

### Loaded from `Market_data.xlsx`

The Excel file is used only for project market assumptions:

- monthly time grid,
- EUR rate curve,
- PLN rate curve,
- EUR discount factors,
- PLN discount factors,
- initial EUR/PLN curve,
- CDS spreads.

### Downloaded from the internet

The historical calibration data is intentionally downloaded online:

- EURIBOR 3M from the ECB Data API,
- EUR/PLN from Yahoo Finance through `yfinance`.

## Methodology

### 1. CDS bootstrapping

CDS spreads are used to bootstrap piecewise-constant hazard rates. The resulting credit curve contains:

- hazard rates,
- survival probabilities,
- monthly default probabilities.

### 2. Market factor calibration

EURIBOR 3M is calibrated as an ABM process:

```text
r(t + dt) = r(t) + mu_r * dt + sigma_r * sqrt(dt) * Z_IR
```

EUR/PLN is calibrated as a GBM process:

```text
S(t + dt) = S(t) * exp((mu_fx - 0.5 * sigma_fx^2) * dt + sigma_fx * sqrt(dt) * Z_FX)
```

The IR-FX correlation is estimated using historical monthly EURIBOR changes and EUR/PLN log-returns. If the correlation is not statistically significant, the simulation uses zero correlation.

### 3. Monte Carlo simulation

The project simulates:

- 10,000 scenarios,
- 36 monthly time steps,
- EUR/PLN paths,
- EURIBOR 3M paths.

### 4. Exposure and CVA

For each instrument and each simulated month, the model calculates:

- Mark-to-Market (MtM),
- positive exposure,
- Expected Exposure (EE),
- Potential Future Exposure (PFE),
- CVA.

CVA is calculated in discrete form:

```text
CVA = LGD * sum(EE(t_i) * DF_PLN(t_i) * q(t_{i-1}, t_i))
```

where:

- `LGD = 60%`,
- `EE(t_i)` is expected exposure,
- `DF_PLN(t_i)` is the PLN discount factor,
- `q(t_{i-1}, t_i)` is the monthly default probability.

### 5. Variation Margin

Monthly Variation Margin is implemented as the previous month portfolio MtM:

```text
VM(T) = MtM_portfolio(T - 1)
```

The secured portfolio value is then:

```text
MtM_portfolio_VM(T) = MtM_portfolio(T) - VM(T)
```

This shows how monthly collateral exchange reduces unsecured counterparty exposure.

## Repository structure

```text
cva-fx-forward-irs-portfolio/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── data/
│   └── Market_data.xlsx
│
├── docs/
│   └── technical_report.pdf
│
├── notebooks/
│   ├── cva_fx_forward_irs.ipynb
│   └── original_student_notebook.ipynb
│
├── scripts/
│   └── run_cva.py
│
├── src/
│   └── cva_portfolio/
│       ├── config.py
│       ├── data.py
│       ├── credit_curve.py
│       ├── market_models.py
│       ├── instruments.py
│       ├── exposure.py
│       ├── cva.py
│       ├── plots.py
│       └── engine.py
│
└── results/
```

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full CVA workflow:

```bash
python scripts/run_cva.py
```

The script downloads EURIBOR 3M and EUR/PLN historical data from the internet, loads CDS and curve assumptions from `data/Market_data.xlsx`, runs the Monte Carlo simulation and saves outputs in `results/`.

## Outputs

After running the project, the following files are generated:

- `results/cva_results.csv`,
- `results/ee_profiles.csv`,
- `results/hazard_rates.csv`,
- `results/credit_curve.csv`,
- `results/expected_exposure_profiles.png`,
- `results/cva_comparison.png`.

## Validation checks

The code includes basic sanity checks:

- simulated FX and IR paths have expected dimensions,
- monthly default probabilities are non-negative,
- survival probability is decreasing over time,
- CVA after Variation Margin is lower than CVA without Variation Margin.

## Tech stack

Python | NumPy | pandas | SciPy | requests | yfinance | ECB Data API | Monte Carlo | CVA | FX Forward | Receiver IRS | CDS bootstrapping | hazard rates | Expected Exposure | PFE | MtM | Variation Margin | counterparty credit risk

## Notes

This project is intended as a quantitative finance portfolio project. It is not a production CVA engine and simplifies selected market conventions, collateral mechanics and interest-rate modelling assumptions.
