"""Descriptive statistics: climatology, annual means, early-vs-late comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import config as cfg
from src.statistics import benjamini_hochberg


def monthly_stats(monthly: pd.DataFrame) -> pd.DataFrame:
    """Per-calendar-month mean/median/SD/min/max across all years."""
    rows = []
    for m in range(1, 13):
        s = monthly.loc[monthly["MONTH"] == m]
        v = s["T2M"].dropna()
        rows.append({
            "MONTH": m, "MONTH_NAME": cfg.MONTH_NAMES[m - 1], "N": int(len(v)),
            "MEAN": v.mean(), "MEDIAN": v.median(), "STD": v.std(ddof=1),
            "MIN": v.min(), "MIN_YEAR": int(s.loc[v.idxmin(), "YEAR"]) if len(v) else np.nan,
            "MAX": v.max(), "MAX_YEAR": int(s.loc[v.idxmax(), "YEAR"]) if len(v) else np.nan,
        })
    return pd.DataFrame(rows)


def weighted_mean(values: np.ndarray, days: np.ndarray, weighted: bool = cfg.DAY_WEIGHTED_AGGREGATION) -> float:
    """Mean of monthly values; NaN if any value is missing (complete-case only)."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or np.isnan(values).any():
        return float("nan")
    return float(np.average(values, weights=days if weighted else None))


def annual_series(monthly: pd.DataFrame, weighted: bool = cfg.DAY_WEIGHTED_AGGREGATION) -> pd.DataFrame:
    """Annual mean computed from the 12 monthly values (ANN column is never used).

    A year is valid only when all 12 months are present.
    """
    rows = []
    for year, g in monthly.groupby("YEAR"):
        rows.append({"YEAR": int(year), "N_MONTHS": int(g["T2M"].notna().sum()),
                     "T2M": weighted_mean(g["T2M"].to_numpy(), g["DAYS_IN_MONTH"].to_numpy(), weighted)})
    return pd.DataFrame(rows)


def period_comparison(matrix: pd.DataFrame, series: dict[str, pd.Series],
                      early: tuple[int, int] = cfg.EARLY_PERIOD,
                      late: tuple[int, int] = cfg.LATE_PERIOD) -> pd.DataFrame:
    """Exploratory late-minus-early mean differences with Mann-Whitney U tests.

    The split is fixed a priori at the sample midpoint (2001-2012 vs 2013-2025)
    and is *not* tuned to the data. Mann-Whitney p-values are BH-adjusted across
    all rows. Treat as descriptive: it duplicates information in the trend tests.
    """
    rows = []
    items = {cfg.MONTH_NAMES[m - 1]: matrix[m] for m in range(1, 13)}
    items.update(series)
    for name, s in items.items():
        s = s.dropna()
        a = s[(s.index >= early[0]) & (s.index <= early[1])]
        b = s[(s.index >= late[0]) & (s.index <= late[1])]
        p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue if len(a) > 2 and len(b) > 2 else np.nan
        rows.append({"SERIES": name, "N_EARLY": len(a), "N_LATE": len(b),
                     "MEAN_EARLY": a.mean(), "MEAN_LATE": b.mean(),
                     "DIFF_LATE_MINUS_EARLY": b.mean() - a.mean(), "P_MANN_WHITNEY": p})
    out = pd.DataFrame(rows)
    out["P_FDR"] = benjamini_hochberg(out["P_MANN_WHITNEY"].to_numpy())
    return out
