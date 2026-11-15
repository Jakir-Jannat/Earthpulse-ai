"""Why EARTHPULSE AI does NOT use a cluster-robust Wald test for slope heterogeneity.

Simulates a null world with the real data's month means and residual structure
but ONE common trend for all months (resampling whole years to preserve
cross-month correlation), then counts how often a month x year interaction Wald
test with year-clustered SEs rejects at 5%. A calibrated test rejects ~5%.

Run:  python scripts/wald_calibration_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config as cfg
from src.data_loader import read_nasa_power_csv
from src.preprocessing import month_year_matrix, wide_to_long


def wald_p(df: pd.DataFrame) -> float:
    d = df.copy()
    d["YC"] = d["YEAR"] - d["YEAR"].mean()
    X = pd.get_dummies(d["MONTH"], prefix="m", drop_first=True, dtype=float)
    inter = X.mul(d["YC"], axis=0)
    exog = pd.concat([pd.Series(1.0, index=d.index, name="const"), X, d["YC"], inter.add_prefix("x_")], axis=1)
    res = sm.OLS(d["T2M"].to_numpy(), exog).fit(cov_type="cluster", cov_kwds={"groups": d["YEAR"].to_numpy()})
    R = np.zeros((inter.shape[1], exog.shape[1]))
    for r in range(inter.shape[1]):
        R[r, exog.shape[1] - inter.shape[1] + r] = 1.0
    return float(res.wald_test(R, use_f=True, scalar=True).pvalue)


def main(n_rep: int = 300, seed: int = 7) -> None:
    raw = read_nasa_power_csv(cfg.RAW_DIR / cfg.RAW_FILENAME)
    M = month_year_matrix(wide_to_long(raw.table))
    yrs, X = M.index.to_numpy(float), M.to_numpy()
    coef = np.array([np.polyfit(yrs, X[:, m], 1) for m in range(12)])
    resid = X - (coef[:, 0][None, :] * yrs[:, None] + coef[:, 1][None, :])
    common = coef[:, 0].mean()
    rng, rej = np.random.default_rng(seed), 0
    for _ in range(n_rep):
        Y = resid[rng.integers(0, len(yrs), len(yrs))] + coef[:, 1][None, :] + common * yrs[:, None]
        df = pd.DataFrame({"YEAR": np.repeat(yrs.astype(int), 12), "MONTH": np.tile(np.arange(1, 13), len(yrs)), "T2M": Y.ravel()})
        rej += wald_p(df) < 0.05
    print(f"Empirical rejection rate under the null: {rej / n_rep:.3f} (nominal 0.050, {n_rep} replicates, seed {seed})")


if __name__ == "__main__":
    main()
