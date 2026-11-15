"""Core trend statistics implemented from first principles.

Implemented here (rather than imported) so each assumption is visible and unit-tested:
Mann-Kendall, Sen's slope with confidence interval, Hamed-Rao modified
Mann-Kendall, Benjamini-Hochberg FDR, paired-year bootstrap, ACF diagnostics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from scipy import stats

import config as cfg


@dataclass
class MKResult:
    """Mann-Kendall test output."""

    n: int
    s: float
    var_s: float
    z: float
    p: float
    tau: float
    trend: str  # 'increasing' | 'decreasing' | 'no trend' (at alpha)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SenResult:
    """Sen's slope (per unit of t) with a distribution-free confidence interval."""

    slope: float
    intercept: float
    lower: float
    upper: float
    n: int

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(x: Sequence[float], t: Sequence[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Drop NaN pairs; t defaults to 0..n-1."""
    x = np.asarray(x, dtype=float)
    t = np.arange(len(x), dtype=float) if t is None else np.asarray(t, dtype=float)
    if x.shape != t.shape:
        raise ValueError("x and t must have the same length")
    ok = np.isfinite(x) & np.isfinite(t)
    return x[ok], t[ok]


def _tie_corrected_var_s(x: np.ndarray) -> float:
    """Variance of S with the tie correction (Kendall 1975)."""
    n = len(x)
    _, counts = np.unique(x, return_counts=True)
    tie = sum(c * (c - 1) * (2 * c + 5) for c in counts if c > 1)
    return (n * (n - 1) * (2 * n + 5) - tie) / 18.0


def mann_kendall(x: Sequence[float], t: Sequence[float] | None = None,
                 alpha: float = cfg.ALPHA) -> MKResult:
    """Two-sided Mann-Kendall trend test with tie and continuity corrections.

    Assumes serially independent observations (see :func:`hamed_rao_mk`).
    Kendall's tau-b is reported via scipy for cross-reference.
    """
    x, t = _clean(x, t)
    n = len(x)
    if n < 3:
        return MKResult(n, np.nan, np.nan, np.nan, np.nan, np.nan, "insufficient data")
    order = np.argsort(t, kind="stable")
    x = x[order]
    diff = x[None, :] - x[:, None]
    s = float(np.sign(diff[np.triu_indices(n, k=1)]).sum())
    var_s = _tie_corrected_var_s(x)
    z = 0.0 if s == 0 else (s - np.sign(s)) / np.sqrt(var_s) if var_s > 0 else np.nan
    p = float(2 * (1 - stats.norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
    tau = float(stats.kendalltau(np.arange(n), x).statistic) if np.ptp(x) > 0 else 0.0
    trend = "no trend" if not (p < alpha) else ("increasing" if z > 0 else "decreasing")
    return MKResult(n, s, float(var_s), float(z), p, tau, trend)


def _rank_value(sorted_vals: np.ndarray, rank: float) -> float:
    """Linear interpolation at a 1-based fractional rank, clipped to [1, N]."""
    rank = min(max(rank, 1.0), float(len(sorted_vals)))
    return float(np.interp(rank, np.arange(1, len(sorted_vals) + 1), sorted_vals))


def sens_slope(x: Sequence[float], t: Sequence[float] | None = None,
               ci_level: float = cfg.CI_LEVEL) -> SenResult:
    """Theil-Sen slope with the Gilbert (1987) rank-based confidence interval.

    Pairs sharing the same t are skipped. The intercept is median(x - slope*t).
    """
    x, t = _clean(x, t)
    n = len(x)
    if n < 3:
        return SenResult(np.nan, np.nan, np.nan, np.nan, n)
    i, j = np.triu_indices(n, k=1)
    dt = t[j] - t[i]
    ok = dt != 0
    slopes = np.sort((x[j] - x[i])[ok] / dt[ok])
    big_n = len(slopes)
    slope = float(np.median(slopes))
    z = stats.norm.ppf(1 - (1 - ci_level) / 2)
    c = z * np.sqrt(_tie_corrected_var_s(x))
    lower = _rank_value(slopes, (big_n - c) / 2.0)
    upper = _rank_value(slopes, (big_n + c) / 2.0 + 1.0)
    return SenResult(slope, float(np.median(x - slope * t)), lower, upper, n)


def rank_acf(r: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation of a series for lags 1..max_lag."""
    r = np.asarray(r, dtype=float) - np.mean(r)
    denom = float(np.sum(r ** 2))
    if denom == 0:
        return np.zeros(max_lag)
    return np.array([np.sum(r[:-k] * r[k:]) / denom for k in range(1, max_lag + 1)])


def acf_bounds(n: int, lag: int, level: float = cfg.CI_LEVEL) -> tuple[float, float]:
    """Hamed & Rao (1998) confidence bounds for the lag-k autocorrelation."""
    z = stats.norm.ppf(1 - (1 - level) / 2)
    return (-1 - z * np.sqrt(n - lag - 1)) / (n - lag), (-1 + z * np.sqrt(n - lag - 1)) / (n - lag)


def hamed_rao_mk(x: Sequence[float], t: Sequence[float] | None = None,
                 alpha: float = cfg.ALPHA, conservative: bool = True) -> dict:
    """Hamed-Rao (1998) modified Mann-Kendall test.

    The variance of S is multiplied by a correction factor n/n* computed from
    the significant autocorrelations of the *ranks of the Sen-detrended series*.
    With ``conservative=True`` (default) the factor is floored at 1, so the test
    can only become more cautious: lag estimates are noisy at n ~ 25 and a
    negative-autocorrelation "correction" would otherwise manufacture significance.
    This is the only autocorrelation correction applied to a series (no prior
    pre-whitening), so there is no double correction. It is not reliable at
    n ~ 25 for long lags, so it is reported alongside, never instead of, the
    plain MK result.
    """
    x, t = _clean(x, t)
    n = len(x)
    base = mann_kendall(x, t, alpha)
    if n < 5 or not np.isfinite(base.var_s):
        return {**base.to_dict(), "correction_factor": np.nan, "p_mmk": np.nan,
                "z_mmk": np.nan, "lags_used": [], "lag1_acf": np.nan, "correction_factor_raw": np.nan}
    sen = sens_slope(x, t)
    detrended = x - sen.slope * t
    ranks = stats.rankdata(detrended)
    max_lag = n - 2
    rho = rank_acf(ranks, max_lag)
    used, acc = [], 0.0
    for k in range(1, max_lag + 1):
        lo, hi = acf_bounds(n, k)
        if rho[k - 1] < lo or rho[k - 1] > hi:
            used.append(k)
            acc += (n - k) * (n - k - 1) * (n - k - 2) * rho[k - 1]
    factor_raw = 1.0 + 2.0 / (n * (n - 1) * (n - 2)) * acc
    factor = max(1.0, factor_raw) if conservative else factor_raw
    var_star = base.var_s * factor
    z = 0.0 if base.s == 0 else (base.s - np.sign(base.s)) / np.sqrt(var_star)
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {**base.to_dict(), "correction_factor": float(factor), "correction_factor_raw": float(factor_raw),
            "z_mmk": float(z), "p_mmk": p,
            "lags_used": used, "lag1_acf": float(rho[0])}


def lag1_autocorr_detrended(x: Sequence[float], t: Sequence[float] | None = None) -> float:
    """Lag-1 autocorrelation of residuals about the Sen line (diagnostic)."""
    x, t = _clean(x, t)
    if len(x) < 5:
        return float("nan")
    res = x - sens_slope(x, t).slope * t
    return float(rank_acf(res, 1)[0])


def benjamini_hochberg(p: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values); NaN inputs stay NaN."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / (np.arange(m) + 1)
    adj_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(adj_sorted, 1.0)
    out[ok] = adj
    return out


def format_ci(estimate: float, lower: float, upper: float, digits: int = 3, unit: str = "") -> str:
    """Format 'estimate [lower, upper] unit' consistently."""
    if not all(np.isfinite([estimate, lower, upper])):
        return "n/a"
    u = f" {unit}" if unit else ""
    return f"{estimate:+.{digits}f} [{lower:+.{digits}f}, {upper:+.{digits}f}]{u}"


def sen_vectorised(t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Sen slopes for many resamples at once.

    ``t`` has shape (B, n) and ``X`` shape (B, n, k). Pairs with equal t
    (duplicated bootstrap years) are excluded. Returns (B, k).
    """
    n = t.shape[1]
    i, j = np.triu_indices(n, k=1)
    dt = t[:, j] - t[:, i]                      # (B, P)
    dx = X[:, j, :] - X[:, i, :]                # (B, P, k)
    with np.errstate(divide="ignore", invalid="ignore"):
        sl = dx / dt[:, :, None]
    sl = np.where((dt != 0)[:, :, None], sl, np.nan)
    return np.nanmedian(sl, axis=1)


def paired_year_bootstrap(years: np.ndarray, X: np.ndarray, n_boot: int = cfg.BOOTSTRAP_N,
                          seed: int = cfg.RANDOM_SEED) -> np.ndarray:
    """Bootstrap Sen slopes by resampling *years* jointly across all k series.

    Resampling whole years keeps cross-month/season dependence intact, so
    differences between series are bootstrapped correctly. It assumes years
    are exchangeable, which is a limitation if there is serial dependence.
    Returns (n_boot, k) slopes per year.
    """
    rng = np.random.default_rng(seed)
    n = len(years)
    idx = rng.integers(0, n, size=(n_boot, n))
    return sen_vectorised(years[idx].astype(float), X[idx, :])


def bootstrap_diff_summary(boot: np.ndarray, a: int, b: int, level: float = cfg.CI_LEVEL) -> dict:
    """Percentile CI and two-sided bootstrap p for slope_a - slope_b."""
    d = boot[:, a] - boot[:, b]
    lo, hi = np.percentile(d, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    p = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
    return {"diff": float(np.median(d)), "lower": float(lo), "upper": float(hi), "p_boot": float(min(p, 1.0))}
