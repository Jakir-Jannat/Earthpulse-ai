"""Pettitt change-point test (secondary, exploratory analysis)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg
from src.statistics import benjamini_hochberg


def pettitt(x: np.ndarray) -> dict:
    """Pettitt (1979) non-parametric single change-point test.

    K = max_t |U_t| with U_t = sum_{i<=t} sum_{j>t} sign(x_i - x_j);
    approximate p = 2 exp(-6 K^2 / (n^3 + n^2)) (capped at 1). The approximation is
    conservative-to-unreliable at small n and assumes independence.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 8 or np.isnan(x).any():
        return {"n": n, "K": np.nan, "p": np.nan, "change_index": None}
    u = np.array([np.sign(x[: t + 1][:, None] - x[t + 1:][None, :]).sum() for t in range(n - 1)])
    k_idx = int(np.argmax(np.abs(u)))
    k = float(np.abs(u[k_idx]))
    p = float(min(1.0, 2 * np.exp(-6 * k ** 2 / (n ** 3 + n ** 2))))
    return {"n": n, "K": k, "p": p, "change_index": k_idx, "u_series": u}


def change_point_table(series: dict[str, pd.Series], trim: int = 2) -> pd.DataFrame:
    """Pettitt on each series plus a trimmed-ends sensitivity run.

    Support is only flagged when the raw p < alpha in the full run, the trimmed run
    also gives p < alpha, and the BH-adjusted p across all series < alpha.
    A single-step shift test cannot separate a step from a steady trend.
    """
    rows = []
    for name, s in series.items():
        s = s.dropna()
        full = pettitt(s.to_numpy())
        cut = pettitt(s.to_numpy()[trim:-trim])
        idx = full["change_index"]
        yr = int(s.index[idx]) if idx is not None else None
        before = s.iloc[: idx + 1].mean() if idx is not None else np.nan
        after = s.iloc[idx + 1:].mean() if idx is not None else np.nan
        cidx = cut["change_index"]
        yr_t = int(s.index[trim:-trim][cidx]) if cidx is not None else None
        rows.append({"SERIES": name, "N": full["n"], "LAST_YEAR_BEFORE_SHIFT": yr,
                     "MEAN_BEFORE": before, "MEAN_AFTER": after, "SHIFT": after - before,
                     "P_PETTITT": full["p"], "TRIMMED_LAST_YEAR_BEFORE_SHIFT": yr_t, "P_PETTITT_TRIMMED": cut["p"]})
    out = pd.DataFrame(rows)
    out["P_FDR"] = benjamini_hochberg(out["P_PETTITT"].to_numpy())
    out["SUPPORTED"] = (out["P_PETTITT"] < cfg.ALPHA) & (out["P_PETTITT_TRIMMED"] < cfg.ALPHA) & (out["P_FDR"] < cfg.FDR_ALPHA)
    return out
