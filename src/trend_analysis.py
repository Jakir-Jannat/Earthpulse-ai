"""Annual and month-wise trend detection, plus uncertainty-aware slope comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config as cfg
from src.seasonal_analysis import series_trend_row
from src.statistics import (benjamini_hochberg, bootstrap_diff_summary, hamed_rao_mk,
                            mann_kendall, paired_year_bootstrap, sens_slope)


def annual_trend(annual: pd.DataFrame) -> dict:
    """Mann-Kendall + Sen (primary) and OLS (complementary, descriptive) for annual mean T2M."""
    a = annual.dropna(subset=["T2M"])
    t, x = a["YEAR"].to_numpy(dtype=float), a["T2M"].to_numpy()
    row = series_trend_row("Annual mean", pd.Series(x, index=a["YEAR"].to_numpy()))
    sen = sens_slope(x, t)
    ols = sm.OLS(x, sm.add_constant(t)).fit()
    pred = ols.get_prediction(sm.add_constant(t)).summary_frame(alpha=1 - cfg.CI_LEVEL)
    ci = ols.conf_int(alpha=1 - cfg.CI_LEVEL)[1]
    resid_acf1 = float(np.corrcoef(ols.resid[:-1], ols.resid[1:])[0, 1])
    return {
        "row": row, "sen": sen.to_dict(),
        "ols": {"slope_per_year": float(ols.params[1]), "slope_per_decade": float(ols.params[1] * 10),
                "ci_low_per_decade": float(ci[0] * 10), "ci_high_per_decade": float(ci[1] * 10),
                "p_value": float(ols.pvalues[1]), "r_squared": float(ols.rsquared),
                "residual_lag1_autocorr": resid_acf1,
                "note": "OLS assumes independent normal residuals; descriptive companion only."},
        "years": t.astype(int), "values": x,
        "ols_fit": pred["mean"].to_numpy(), "ols_band_low": pred["mean_ci_lower"].to_numpy(),
        "ols_band_high": pred["mean_ci_upper"].to_numpy(),
        "sen_fit": sen.intercept + sen.slope * t,
    }


def monthly_trends(matrix: pd.DataFrame) -> pd.DataFrame:
    """Month-wise MK / Sen / CI for 12 annual series with BH-FDR across the 12 tests.

    The primary p-value is the plain Mann-Kendall p (each calendar-month series
    has one value per year, so serial dependence is expected to be weak; the
    lag-1 diagnostics and Hamed-Rao sensitivity p-values are reported beside it).
    """
    rows = []
    n_years = len(matrix)
    for m in range(1, 13):
        s = matrix[m]
        valid = s.dropna()
        t, x = valid.index.to_numpy(dtype=float), valid.to_numpy()
        mk, sen, hr = mann_kendall(x, t), sens_slope(x, t), hamed_rao_mk(x, t)
        rows.append({
            "MONTH": m, "MONTH_NAME": cfg.MONTH_NAMES[m - 1], "N_VALID": mk.n, "N_MISSING": n_years - mk.n,
            "MEAN_T2M": float(x.mean()), "MK_S": mk.s, "KENDALL_TAU": mk.tau, "MK_Z": mk.z, "P_RAW": mk.p,
            "SEN_PER_YEAR": sen.slope, "SEN_PER_DECADE": sen.slope * 10,
            "CI_LOW_PER_DECADE": sen.lower * 10, "CI_HIGH_PER_DECADE": sen.upper * 10,
            "LAG1_ACF": hr["lag1_acf"], "MMK_FACTOR": hr["correction_factor"], "P_MMK": hr["p_mmk"],
        })
    out = pd.DataFrame(rows)
    out["P_FDR"] = benjamini_hochberg(out["P_RAW"].to_numpy())
    out["P_MMK_FDR"] = benjamini_hochberg(out["P_MMK"].to_numpy())
    out["SIG_RAW"] = out["P_RAW"] < cfg.ALPHA
    out["SIG_FDR"] = out["P_FDR"] < cfg.FDR_ALPHA
    out["RANK_WARMING"] = out["SEN_PER_DECADE"].rank(ascending=False, method="min").astype(int)
    out["CI_EXCLUDES_ZERO"] = (out["CI_LOW_PER_DECADE"] > 0) | (out["CI_HIGH_PER_DECADE"] < 0)
    return out


def _compare(boot: np.ndarray, names: list[str], annual_col: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise and vs-annual bootstrap slope differences (per decade), BH-adjusted."""
    k = boot.shape[1] if annual_col is None else boot.shape[1] - 1
    pairs = []
    for a in range(k):
        for b in range(a + 1, k):
            d = bootstrap_diff_summary(boot * 10, a, b)
            pairs.append({"A": names[a], "B": names[b], "DIFF_A_MINUS_B_PER_DECADE": d["diff"],
                          "CI_LOW": d["lower"], "CI_HIGH": d["upper"], "P_BOOT": d["p_boot"]})
    pairs_df = pd.DataFrame(pairs)
    pairs_df["P_BOOT_FDR"] = benjamini_hochberg(pairs_df["P_BOOT"].to_numpy())
    pairs_df["SUPPORTED_DIFFERENCE"] = pairs_df["P_BOOT_FDR"] < cfg.FDR_ALPHA
    vs = pd.DataFrame()
    if annual_col is not None:
        rows = []
        for a in range(k):
            d = bootstrap_diff_summary(boot * 10, a, annual_col)
            rows.append({"SERIES": names[a], "DIFF_FROM_ANNUAL_PER_DECADE": d["diff"],
                         "CI_LOW": d["lower"], "CI_HIGH": d["upper"], "P_BOOT": d["p_boot"]})
        vs = pd.DataFrame(rows)
        vs["P_BOOT_FDR"] = benjamini_hochberg(vs["P_BOOT"].to_numpy())
        vs["SUPPORTED_DIFFERENCE"] = vs["P_BOOT_FDR"] < cfg.FDR_ALPHA
    return pairs_df, vs


def heterogeneity_test(boot: np.ndarray, point: np.ndarray) -> dict:
    """Global bootstrap test that all k Sen slopes are equal.

    Primary statistic (precision-weighted, Cochran-Q style):
        Q_w = sum_k w_k (s_k - s_w)^2,  w_k = 1 / bootstrap variance of s_k,
        s_w = weighted mean slope.
    An unweighted version (Q = sum (s_k - mean)^2) is returned as a sensitivity
    check. The null is imposed by re-centring every bootstrap slope on its own
    point estimate; whole years are resampled so cross-series dependence is kept.
    p = share of null statistics >= the observed one.

    A cluster-robust Wald test on a month x year interaction was evaluated and
    rejected: with 25 year-clusters it rejected 52% of null simulations at a
    nominal 5% (see scripts/wald_calibration_check.py).
    """
    w = 1.0 / np.var(boot, axis=0, ddof=1)

    def q_w(s: np.ndarray) -> np.ndarray:
        s = np.atleast_2d(s)
        mean_w = (s * w).sum(axis=1, keepdims=True) / w.sum()
        return (w * (s - mean_w) ** 2).sum(axis=1)

    def q_u(s: np.ndarray) -> np.ndarray:
        s = np.atleast_2d(s)
        return ((s - s.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)

    centred = boot - point[None, :]
    out = {"n_boot": int(len(boot))}
    for key, fn in (("weighted", q_w), ("unweighted", q_u)):
        obs = float(fn(point)[0])
        out[f"Q_{key}"] = obs
        out[f"p_{key}"] = float(np.mean(fn(centred) >= obs))
    out["p_value"] = out["p_weighted"]  # primary
    return out


def compare_slopes(years: np.ndarray, X: np.ndarray, names: list[str], annual: np.ndarray | None) -> dict:
    """Run the bootstrap comparison for k series (and optionally the annual mean)."""
    full = X if annual is None else np.column_stack([X, annual])
    boot = paired_year_bootstrap(years, full)
    point = np.array([sens_slope(full[:, j], years).slope for j in range(full.shape[1])])
    pairs, vs = _compare(boot, names + (["Annual mean"] if annual is not None else []), None if annual is None else full.shape[1] - 1)
    k = X.shape[1]
    return {"pairs": pairs, "vs_annual": vs, "heterogeneity": heterogeneity_test(boot[:, :k], point[:k]),
            "boot_slopes_per_decade": pd.DataFrame(boot * 10, columns=names + (["Annual mean"] if annual is not None else []))}
