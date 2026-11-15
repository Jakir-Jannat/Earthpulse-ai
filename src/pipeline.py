"""End-to-end analysis pipeline: raw CSV in, a fully computed results bundle out."""
from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import statsmodels

import config as cfg
from src import (change_point as cp, exploratory_analysis as ea,
                 seasonal_analysis as sa, trend_analysis as ta)
from src.data_loader import RawPowerFile, read_nasa_power_csv
from src.data_validation import audit_raw_file
from src.preprocessing import month_year_matrix, wide_to_long
from src.statistics import acf_bounds, rank_acf


class AuditFailure(RuntimeError):
    """Raised when the audit finds errors that make analysis unsafe."""


@dataclass
class Results:
    """Everything computed from the uploaded file; nothing here is hard-coded."""

    raw: RawPowerFile
    audit: dict[str, Any]
    monthly: pd.DataFrame
    matrix: pd.DataFrame
    annual: pd.DataFrame
    month_stats: pd.DataFrame
    seasonal: pd.DataFrame
    seasonal_mat: pd.DataFrame
    annual_trend: dict[str, Any]
    month_trends: pd.DataFrame
    season_trends: pd.DataFrame
    month_comparison: dict[str, Any]
    season_comparison: dict[str, Any]
    amplitude: pd.DataFrame
    amplitude_trends: pd.DataFrame
    change_points: pd.DataFrame
    period_comparison: pd.DataFrame
    annual_sensitivity: dict[str, Any]
    acf_table: pd.DataFrame
    run_info: dict[str, Any] = field(default_factory=dict)


def acf_diagnostics(series: dict[str, pd.Series], max_lag: int = 8) -> pd.DataFrame:
    """Lag 1..max_lag autocorrelation of Sen-detrended ranks with Hamed-Rao bounds."""
    from scipy.stats import rankdata
    from src.statistics import sens_slope
    rows = []
    for name, s in series.items():
        s = s.dropna()
        t, x = s.index.to_numpy(dtype=float), s.to_numpy()
        res = rankdata(x - sens_slope(x, t).slope * t)
        rho = rank_acf(res, max_lag)
        for k in range(1, max_lag + 1):
            lo, hi = acf_bounds(len(x), k)
            rows.append({"SERIES": name, "LAG": k, "ACF": rho[k - 1], "LOWER": lo, "UPPER": hi,
                         "OUTSIDE_BOUNDS": bool(rho[k - 1] < lo or rho[k - 1] > hi)})
    return pd.DataFrame(rows)


def run_pipeline(raw_path: str | Path, allow_audit_errors: bool = False) -> Results:
    """Run ingestion -> audit -> preprocessing -> all analyses."""
    raw = read_nasa_power_csv(raw_path)
    audit = audit_raw_file(raw)
    if audit["status"] == "FAIL" and not allow_audit_errors:
        msgs = "; ".join(i["message"] for i in audit["issues"] if i["severity"] == "error")
        raise AuditFailure(f"Data audit failed: {msgs}")

    monthly = wide_to_long(raw.table)
    matrix = month_year_matrix(monthly)
    annual = ea.annual_series(monthly)
    seasonal = sa.seasonal_series(monthly)
    smat = sa.seasonal_matrix(seasonal)

    a_trend = ta.annual_trend(annual)
    m_trends = ta.monthly_trends(matrix)
    s_trends = sa.seasonal_trends(smat)

    # Bootstrap comparisons (paired year resampling, fixed seed)
    ann_aligned = annual.set_index("YEAR")["T2M"].reindex(matrix.index).to_numpy()
    years = matrix.index.to_numpy()
    month_names = [cfg.MONTH_NAMES[m - 1] for m in range(1, 13)]
    month_cmp = ta.compare_slopes(years, matrix.to_numpy(), month_names, ann_aligned)
    s_aligned = smat.reindex(matrix.index)
    season_cmp = ta.compare_slopes(years, s_aligned.to_numpy(), [cfg.SEASON_LABELS[k] for k in cfg.SEASONS], ann_aligned)

    amp = sa.annual_amplitude(matrix, monthly)
    amp_trends = pd.DataFrame([
        sa.series_trend_row("Annual range (hottest - coldest month)", amp.set_index("YEAR")["RANGE"]),
        sa.series_trend_row("First-harmonic amplitude", amp.set_index("YEAR")["HARMONIC"]),
    ])

    ann_s = annual.set_index("YEAR")["T2M"]
    cp_series = {"Annual mean": ann_s, **{cfg.SEASON_LABELS[k]: smat[k] for k in cfg.SEASONS}}
    cpt = cp.change_point_table(cp_series)
    period = ea.period_comparison(matrix, {"Annual mean": ann_s, **{cfg.SEASON_LABELS[k]: smat[k] for k in cfg.SEASONS}})

    # Sensitivity: simple (unweighted) annual mean
    simple = ea.annual_series(monthly, weighted=False).set_index("YEAR")["T2M"]
    sens_row = sa.series_trend_row("Annual mean (unweighted)", simple)

    acf = acf_diagnostics({"Annual mean": ann_s, **{cfg.SEASON_LABELS[k]: smat[k] for k in cfg.SEASONS}})

    info = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "statsmodels": statsmodels.__version__,
            "random_seed": cfg.RANDOM_SEED, "bootstrap_replicates": cfg.BOOTSTRAP_N,
            "raw_sha256": raw.sha256}
    return Results(raw, audit, monthly, matrix, annual, ea.monthly_stats(monthly), seasonal, smat,
                   a_trend, m_trends, s_trends, month_cmp, season_cmp, amp, amp_trends, cpt,
                   period, {"unweighted_annual": sens_row}, acf, info)
