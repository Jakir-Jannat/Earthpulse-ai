import itertools

import numpy as np
import pytest
from scipy import stats as sst
from statsmodels.stats.multitest import multipletests

from src import statistics as S
from src.change_point import pettitt


def brute_mk_s(x):
    return sum(np.sign(x[j] - x[i]) for i, j in itertools.combinations(range(len(x)), 2))


def brute_sen(x, t):
    return float(np.median([(x[j] - x[i]) / (t[j] - t[i]) for i, j in itertools.combinations(range(len(x)), 2)]))


def test_mk_known_fixture():
    r = S.mann_kendall([1, 2, 3, 4, 5])
    assert r.s == 10 and r.var_s == pytest.approx(5 * 4 * 15 / 18)
    assert r.z == pytest.approx(9 / np.sqrt(5 * 4 * 15 / 18)) and r.p == pytest.approx(0.02749, abs=1e-4)
    assert r.trend == "increasing" and r.tau == pytest.approx(1.0)


def test_mk_decreasing_and_flat():
    assert S.mann_kendall([5, 4, 3, 2, 1, 0, -1, -2]).trend == "decreasing"
    flat = S.mann_kendall([3.0] * 10)
    assert flat.s == 0 and flat.p == pytest.approx(1.0)


def test_mk_tie_correction():
    r = S.mann_kendall([1, 1, 2, 2, 3])
    assert r.var_s == pytest.approx((5 * 4 * 15 - 2 * 18) / 18)


def test_mk_matches_independent_calculations():
    rng = np.random.default_rng(3)
    x = rng.normal(size=25) + 0.05 * np.arange(25)
    r = S.mann_kendall(x)
    assert r.s == brute_mk_s(x)
    assert r.tau == pytest.approx(sst.kendalltau(np.arange(25), x).statistic)
    assert r.p == pytest.approx(sst.kendalltau(np.arange(25), x, method="asymptotic").pvalue, abs=0.02)


def test_mk_nan_handling_and_small_n():
    assert S.mann_kendall([1, np.nan, 3, 4, 5, 6]).n == 5
    assert np.isnan(S.mann_kendall([1, 2]).p)


def test_sen_perfect_line_and_brute_force():
    t = np.arange(10.0)
    s = S.sens_slope(2 * t + 1, t)
    assert s.slope == 2 and s.intercept == 1 and s.lower == 2 and s.upper == 2
    rng = np.random.default_rng(5)
    x = rng.normal(size=25)
    t = np.arange(2001, 2026.0)
    r = S.sens_slope(x, t)
    assert r.slope == pytest.approx(brute_sen(x, t))
    assert r.slope == pytest.approx(sst.theilslopes(x, t).statistic if hasattr(sst.theilslopes(x, t), "statistic") else sst.theilslopes(x, t)[0])
    assert r.lower <= r.slope <= r.upper


def test_sen_ci_widens_with_noise():
    t = np.arange(25.0)
    rng = np.random.default_rng(1)
    quiet = S.sens_slope(0.1 * t + rng.normal(0, 0.1, 25), t)
    noisy = S.sens_slope(0.1 * t + rng.normal(0, 2.0, 25), t)
    assert (noisy.upper - noisy.lower) > (quiet.upper - quiet.lower)


def test_against_pymannkendall_if_available():
    pmk = pytest.importorskip("pymannkendall")
    rng = np.random.default_rng(11)
    x = rng.normal(size=25) + 0.04 * np.arange(25)
    ref = pmk.original_test(x)
    mine, sen = S.mann_kendall(x), S.sens_slope(x)
    assert mine.p == pytest.approx(ref.p) and mine.z == pytest.approx(ref.z) and mine.s == ref.s
    assert sen.slope == pytest.approx(ref.slope)
    hr = S.hamed_rao_mk(x, conservative=False)
    ref_hr = pmk.hamed_rao_modification_test(x)
    assert hr["p_mmk"] == pytest.approx(ref_hr.p, abs=1e-6)


def test_bh_matches_statsmodels_and_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert S.benjamini_hochberg(p) == pytest.approx([0.02, 0.04, 0.04, 0.02])
    rng = np.random.default_rng(2)
    q = rng.uniform(size=12)
    assert S.benjamini_hochberg(q) == pytest.approx(multipletests(q, method="fdr_bh")[1])
    assert (S.benjamini_hochberg(q) >= q - 1e-12).all() and (S.benjamini_hochberg(q) <= 1).all()


def test_bh_ignores_nan():
    out = S.benjamini_hochberg([0.01, np.nan, 0.02])
    assert np.isnan(out[1]) and out[0] == pytest.approx(0.02)


def test_ci_formatting():
    assert S.format_ci(0.1234, -0.05, 0.3) == "+0.123 [-0.050, +0.300]"
    assert S.format_ci(0.1, 0.0, 0.2, digits=1, unit="°C/decade") == "+0.1 [+0.0, +0.2] °C/decade"
    assert S.format_ci(np.nan, 0, 1) == "n/a"


def test_hamed_rao_white_noise_vs_autocorrelated():
    rng = np.random.default_rng(8)
    white = S.hamed_rao_mk(rng.normal(size=25))
    assert white["correction_factor"] >= 1.0
    e = np.zeros(60)
    for i in range(1, 60):
        e[i] = 0.85 * e[i - 1] + rng.normal()
    ar = S.hamed_rao_mk(e)
    assert ar["correction_factor"] > 1.0 and ar["p_mmk"] >= ar["p"]


def test_hamed_rao_is_inflate_only_by_default():
    rng = np.random.default_rng(4)
    for _ in range(20):
        r = S.hamed_rao_mk(rng.normal(size=25))
        assert r["correction_factor"] >= 1.0 and r["p_mmk"] >= r["p"] - 1e-12


def test_vectorised_sen_matches_scalar_and_bootstrap_is_deterministic():
    rng = np.random.default_rng(6)
    t = np.arange(2001, 2026.0)
    X = rng.normal(size=(25, 3))
    v = S.sen_vectorised(t[None, :], X[None, :, :])[0]
    assert v == pytest.approx([S.sens_slope(X[:, k], t).slope for k in range(3)])
    a = S.paired_year_bootstrap(t, X, n_boot=200, seed=1)
    b = S.paired_year_bootstrap(t, X, n_boot=200, seed=1)
    c = S.paired_year_bootstrap(t, X, n_boot=200, seed=2)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_bootstrap_difference_detects_real_difference():
    rng = np.random.default_rng(9)
    t = np.arange(2001, 2026.0)
    X = np.column_stack([0.10 * (t - 2001) + rng.normal(0, 0.2, 25), 0.00 * (t - 2001) + rng.normal(0, 0.2, 25)])
    boot = S.paired_year_bootstrap(t, X, n_boot=1000)
    d = S.bootstrap_diff_summary(boot, 0, 1)
    assert d["lower"] > 0 and d["p_boot"] < 0.01


def test_pettitt_detects_step_and_rejects_noise():
    step = np.r_[np.zeros(12), np.ones(13)] + np.random.default_rng(0).normal(0, 0.05, 25)
    r = pettitt(step)
    assert r["change_index"] == 11 and r["p"] < 0.01
    assert pettitt(np.random.default_rng(1).normal(size=25))["p"] > 0.05
