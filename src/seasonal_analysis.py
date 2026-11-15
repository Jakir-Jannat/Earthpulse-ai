"""Meteorological seasons, seasonal trends and seasonal amplitude."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg
from src.exploratory_analysis import weighted_mean
from src.statistics import hamed_rao_mk, mann_kendall, sens_slope


def month_to_season(month: int) -> str:
    """Return the season key (DJF/MAM/JJAS/ON) of a calendar month."""
    for key, months in cfg.SEASONS.items():
        if month in months:
            return key
    raise ValueError(f"Invalid month: {month}")


def season_year(year: int, month: int) -> int:
    """Year a month contributes to. December belongs to the *following* DJF.

    Example: Dec 2001 -> DJF 2002 (with Jan-Feb 2002).
    """
    return year + 1 if month == 12 else year


def seasonal_series(monthly: pd.DataFrame, weighted: bool = cfg.DAY_WEIGHTED_AGGREGATION) -> pd.DataFrame:
    """Seasonal mean per (season, season-year); only complete seasons get a value.

    DJF 2001 (Jan-Feb only, no Dec 2000) and DJF 2026 (Dec 2025 only) are incomplete
    and therefore NaN with ``N_MONTHS`` documenting why.
    """
    df = monthly.copy()
    df["SEASON"] = df["MONTH"].map(month_to_season)
    df["SEASON_YEAR"] = [season_year(y, m) for y, m in zip(df["YEAR"], df["MONTH"])]
    rows = []
    for (season, sy), g in df.groupby(["SEASON", "SEASON_YEAR"]):
        need = len(cfg.SEASONS[season])
        present = g.loc[g["T2M"].notna()]
        complete = len(g) == need and len(present) == need
        rows.append({"SEASON": season, "SEASON_YEAR": int(sy), "N_MONTHS": int(len(present)),
                     "N_EXPECTED": need, "COMPLETE": bool(complete),
                     "T2M": weighted_mean(g["T2M"].to_numpy(), g["DAYS_IN_MONTH"].to_numpy(), weighted) if complete else np.nan})
    return pd.DataFrame(rows).sort_values(["SEASON", "SEASON_YEAR"]).reset_index(drop=True)


def seasonal_matrix(seasonal: pd.DataFrame) -> pd.DataFrame:
    """Season-year (rows) x season (columns, config order) matrix."""
    m = seasonal.pivot(index="SEASON_YEAR", columns="SEASON", values="T2M")
    return m[list(cfg.SEASONS)].sort_index()


def series_trend_row(name: str, s: pd.Series, unit_note: str = "degC") -> dict:
    """Trend statistics for one annual-resolution series indexed by year."""
    s = s.dropna()
    if len(s) < cfg.MIN_VALID_FOR_TREND:
        return {"SERIES": name, "N": len(s)}
    t, x = s.index.to_numpy(dtype=float), s.to_numpy()
    mk, sen, hr = mann_kendall(x, t), sens_slope(x, t), hamed_rao_mk(x, t)
    return {
        "SERIES": name, "N": mk.n, "FIRST_YEAR": int(t.min()), "LAST_YEAR": int(t.max()),
        "MEAN": float(x.mean()), "MK_S": mk.s, "KENDALL_TAU": mk.tau, "MK_Z": mk.z, "P_MK": mk.p,
        "SEN_PER_YEAR": sen.slope, "SEN_PER_DECADE": sen.slope * 10,
        "CI_LOW_PER_DECADE": sen.lower * 10, "CI_HIGH_PER_DECADE": sen.upper * 10,
        "LAG1_ACF_RANK_DETRENDED": hr["lag1_acf"], "MMK_CORRECTION_FACTOR": hr["correction_factor"],
        "MMK_LAGS_USED": ",".join(map(str, hr["lags_used"])) or "none", "P_MMK": hr["p_mmk"],
    }


def seasonal_trends(smat: pd.DataFrame) -> pd.DataFrame:
    """Mann-Kendall (plain and Hamed-Rao), Sen slope, CI per season."""
    rows = [series_trend_row(cfg.SEASON_LABELS[k], smat[k]) for k in cfg.SEASONS]
    out = pd.DataFrame(rows)
    out.insert(1, "SEASON_KEY", list(cfg.SEASONS))
    from src.statistics import benjamini_hochberg
    out["P_MK_FDR"] = benjamini_hochberg(out["P_MK"].to_numpy())
    out["P_MMK_FDR"] = benjamini_hochberg(out["P_MMK"].to_numpy())
    return out


def annual_amplitude(matrix: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """Two measures of the annual cycle per calendar year (complete years only).

    RANGE    : hottest monthly mean minus coldest monthly mean in the year.
    HARMONIC : amplitude of the fitted first annual harmonic (uses all 12 months).
    """
    k = np.arange(12)
    c, s = np.cos(2 * np.pi * k / 12), np.sin(2 * np.pi * k / 12)
    rows = []
    for year, row in matrix.iterrows():
        v = row.to_numpy(dtype=float)
        if np.isnan(v).any():
            rows.append({"YEAR": int(year), "RANGE": np.nan, "HARMONIC": np.nan})
            continue
        a, b = (2 / 12) * (v * c).sum(), (2 / 12) * (v * s).sum()
        rows.append({"YEAR": int(year), "RANGE": float(v.max() - v.min()), "HARMONIC": float(np.hypot(a, b))})
    return pd.DataFrame(rows)
