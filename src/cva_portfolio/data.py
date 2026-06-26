from __future__ import annotations

from pathlib import Path
import re
from io import StringIO
import numpy as np
import pandas as pd
import requests
import yfinance as yf


def load_market_curves(path: str | Path, months: np.ndarray) -> dict:
    """Load only project market assumptions from Market_data.xlsx.

    In the original project, Market_data.xlsx is used for:
    - monthly time grid,
    - EUR and PLN rate curves,
    - EUR and PLN discount factors,
    - initial EUR/PLN forward/market curve,
    - CDS spreads.

    Historical EURIBOR 3M and EUR/PLN observations are intentionally NOT loaded
    from this workbook. They are downloaded from the internet in separate functions.
    """

    path = Path(path)
    df_mkt = pd.read_excel(path, sheet_name="Market data", usecols="F:K", skiprows=3)
    df_mkt = df_mkt.dropna(how="all")
    df_mkt["Month"] = pd.to_numeric(df_mkt["Month"], errors="coerce")
    df_mkt = df_mkt.dropna(subset=["Month"])
    df_mkt["Month"] = df_mkt["Month"].astype(int)
    df_mkt = df_mkt.sort_values("Month").reset_index(drop=True)

    required_cols = ["EUR_Rate", "PLN_rate", "EUR_DF", "PLN_DF", "EUR/PLN"]
    for col in required_cols:
        if col not in df_mkt.columns:
            raise ValueError(f"Missing required column in Market_data.xlsx: {col}")
        df_mkt[col] = pd.to_numeric(df_mkt[col], errors="coerce")

    return {
        "raw": df_mkt,
        "eur_df": np.interp(months, df_mkt["Month"], df_mkt["EUR_DF"]),
        "pln_df": np.interp(months, df_mkt["Month"], df_mkt["PLN_DF"]),
        "eur_rate_curve": np.interp(months, df_mkt["Month"], df_mkt["EUR_Rate"]),
        "pln_rate_curve": np.interp(months, df_mkt["Month"], df_mkt["PLN_rate"]),
        "eurpln_curve": np.interp(months, df_mkt["Month"], df_mkt["EUR/PLN"]),
    }


def load_cds_spreads(path: str | Path, max_months: int = 36) -> pd.DataFrame:
    """Load CDS spreads from Market_data.xlsx and convert tenors to maturities."""

    path = Path(path)
    cds_raw = pd.read_excel(path, sheet_name="Market data", header=None, usecols="B:C")
    cds_raw.columns = ["Tenor", "Spread_bps"]
    cds_raw["Spread_bps"] = pd.to_numeric(cds_raw["Spread_bps"], errors="coerce")
    cds = cds_raw.dropna(subset=["Spread_bps"]).copy()
    cds["Tenor"] = cds["Tenor"].astype(str)

    maturities = []
    for tenor in cds["Tenor"].str.upper():
        found = re.search(r"(\d+)\s*([MY])", tenor)
        if not found:
            maturities.append(np.nan)
            continue
        number = int(found.group(1))
        unit = found.group(2)
        maturities.append(number if unit == "M" else number * 12)

    cds["MaturityMonths"] = maturities
    cds = cds.dropna(subset=["MaturityMonths"]).copy()
    cds["MaturityMonths"] = cds["MaturityMonths"].astype(int)
    cds = cds[cds["MaturityMonths"] <= max_months].sort_values("MaturityMonths").reset_index(drop=True)
    cds["MaturityYears"] = cds["MaturityMonths"] / 12
    cds["Spread_decimal"] = cds["Spread_bps"] / 10_000
    return cds


def download_eurpln_from_yfinance(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    ticker: str = "EURPLN=X",
) -> pd.DataFrame:
    """Download EUR/PLN data from Yahoo Finance via yfinance."""

    df = yf.download(
        ticker,
        start=pd.to_datetime(start_date).strftime("%Y-%m-%d"),
        end=pd.to_datetime(end_date).strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
    )

    if df.empty:
        raise ValueError(
            f"No EUR/PLN data downloaded from yfinance for ticker {ticker}. "
            "Check internet connection, ticker availability or date range."
        )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    price_col = "Close" if "Close" in df.columns else "Adj Close"
    fx = df[[price_col]].rename(columns={price_col: "EURPLN"}).dropna().reset_index()
    fx["Date"] = pd.to_datetime(fx["Date"])
    fx = fx.sort_values("Date").reset_index(drop=True)

    # Monthly observations are used for correlation and calibration consistency.
    fx_monthly = fx.set_index("Date").resample("ME").last().dropna().reset_index()
    fx_monthly["YearMonth"] = fx_monthly["Date"].dt.to_period("M")
    fx_monthly["fx_log_return"] = np.log(fx_monthly["EURPLN"] / fx_monthly["EURPLN"].shift(1))
    return fx_monthly


def download_euribor_3m_from_ecb(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Download EURIBOR 3M historical data from the ECB Data API.

    The function uses the ECB Data Portal SDMX endpoint for the monthly EURIBOR 3M
    series. Returned values are converted to decimals when ECB provides them in percent.
    """

    start = pd.to_datetime(start_date).strftime("%Y-%m")
    end = pd.to_datetime(end_date).strftime("%Y-%m")
    series_key = "M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA"

    urls = [
        f"https://data-api.ecb.europa.eu/service/data/FM/{series_key}?startPeriod={start}&endPeriod={end}&format=csvdata",
        f"https://sdw-wsrest.ecb.europa.eu/service/data/FM/{series_key}?startPeriod={start}&endPeriod={end}&format=csvdata",
    ]

    last_error = None
    for url in urls:
        try:
            response = requests.get(url, timeout=30, headers={"Accept": "text/csv"})
            response.raise_for_status()
            text = response.text.strip()
            if not text:
                continue
            df = pd.read_csv(StringIO(text))
            if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
                continue

            euribor = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
            euribor.columns = ["Date", "EURIBOR3M"]
            euribor["Date"] = pd.to_datetime(euribor["Date"], errors="coerce")
            euribor["EURIBOR3M"] = pd.to_numeric(euribor["EURIBOR3M"], errors="coerce")
            euribor = euribor.dropna(subset=["Date", "EURIBOR3M"]).sort_values("Date")

            if euribor.empty:
                continue

            # ECB rates are often quoted in percent, e.g. 2.02 instead of 0.0202.
            if euribor["EURIBOR3M"].abs().median() > 1:
                euribor["EURIBOR3M"] = euribor["EURIBOR3M"] / 100

            euribor["YearMonth"] = euribor["Date"].dt.to_period("M")
            return euribor.reset_index(drop=True)

        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise ValueError(
        "Could not download EURIBOR 3M data from ECB. "
        "Check internet connection or ECB API availability. "
        f"Last error: {last_error}"
    )
