"""Anomalies relative to a user-defined baseline period."""
from __future__ import annotations

import pandas as pd

import config as cfg


def validate_baseline(baseline: tuple[int, int], years: pd.Series) -> tuple[int, int]:
    """Check a baseline lies within the data and has at least 10 years."""
    lo, hi = baseline
    if lo > hi:
        raise ValueError("Baseline start year must not exceed end year.")
    if lo < years.min() or hi > years.max():
        raise ValueError(f"Baseline must lie within {years.min()}-{years.max()}.")
    if hi - lo + 1 < 10:
        raise ValueError("Baseline must span at least 10 years for a stable climatology.")
    return lo, hi


def monthly_climatology(monthly: pd.DataFrame, baseline: tuple[int, int] = cfg.DEFAULT_BASELINE) -> pd.Series:
    """Calendar-month mean over the baseline years (index 1..12)."""
    lo, hi = validate_baseline(baseline, monthly["YEAR"])
    b = monthly[(monthly["YEAR"] >= lo) & (monthly["YEAR"] <= hi)]
    return b.groupby("MONTH")["T2M"].mean()


def monthly_anomalies(monthly: pd.DataFrame, baseline: tuple[int, int] = cfg.DEFAULT_BASELINE) -> pd.DataFrame:
    """Add ANOMALY = T2M - baseline climatology for that calendar month."""
    clim = monthly_climatology(monthly, baseline)
    out = monthly.copy()
    out["CLIMATOLOGY"] = out["MONTH"].map(clim)
    out["ANOMALY"] = out["T2M"] - out["CLIMATOLOGY"]
    out.attrs["baseline"] = baseline
    return out


def anomaly_matrix(anom: pd.DataFrame) -> pd.DataFrame:
    """Year x month matrix of anomalies."""
    return anom.pivot(index="YEAR", columns="MONTH", values="ANOMALY").sort_index()


def series_anomaly(s: pd.Series, baseline: tuple[int, int]) -> pd.Series:
    """Anomaly of an annual-resolution series (index = year) vs its baseline mean."""
    lo, hi = baseline
    base = s[(s.index >= lo) & (s.index <= hi)].mean()
    return s - base
