"""Integration tests on the real uploaded file (skipped if it is absent)."""
import numpy as np
import pandas as pd
import pytest

import config as cfg
from src.pipeline import AuditFailure, run_pipeline
from tests.helpers import wide_table, write_csv

RAW = cfg.RAW_DIR / cfg.RAW_FILENAME
pytestmark = pytest.mark.skipif(not RAW.exists(), reason="raw NASA POWER file not present")


@pytest.fixture(scope="module")
def R():
    return run_pipeline(RAW)


def test_raw_file_is_untouched_and_audit_passes(R):
    from src.data_loader import sha256_of
    assert sha256_of(RAW) == R.raw.sha256
    assert R.audit["status"] in ("PASS", "PASS_WITH_WARNINGS") and len(R.monthly) == 300


def test_annual_mean_agrees_with_nasa_ann_only_as_a_check(R):
    ann_col = pd.to_numeric(R.raw.table["ANN"]).to_numpy()
    mine = R.annual.sort_values("YEAR")["T2M"].to_numpy()
    assert np.abs(mine - ann_col).max() < 0.01


def test_twelve_monthly_tests_with_adjusted_p(R):
    mt = R.month_trends
    assert len(mt) == 12 and (mt["N_VALID"] == 25).all()
    assert (mt["P_FDR"] >= mt["P_RAW"] - 1e-12).all()
    np.testing.assert_allclose(mt["SEN_PER_DECADE"], mt["SEN_PER_YEAR"] * 10)
    assert (mt["CI_LOW_PER_DECADE"] <= mt["SEN_PER_DECADE"]).all() and (mt["SEN_PER_DECADE"] <= mt["CI_HIGH_PER_DECADE"]).all()


def test_results_are_computed_not_hardcoded(R, tmp_path):
    """Add an exact +0.05 degC/year trend to every month: every Sen slope must rise by exactly 0.5 degC/decade."""
    t = R.raw.table.copy()
    for i, row in t.iterrows():
        add = 0.05 * (int(row["YEAR"]) - 2001)
        for m in cfg.MONTH_ABBR:
            t.loc[i, m] = repr(round(float(row[m]) + add, 6))
    R2 = run_pipeline(write_csv(tmp_path / "shift.csv", t))
    assert R2.month_trends["SEN_PER_DECADE"].to_numpy() == pytest.approx(R.month_trends["SEN_PER_DECADE"].to_numpy() + 0.5, abs=1e-6)
    assert R2.annual_trend["row"]["SEN_PER_DECADE"] == pytest.approx(R.annual_trend["row"]["SEN_PER_DECADE"] + 0.5, abs=1e-6)


def test_pipeline_is_deterministic(R):
    R2 = run_pipeline(RAW)
    pd.testing.assert_frame_equal(R.month_comparison["pairs"], R2.month_comparison["pairs"])
    assert R.month_comparison["heterogeneity"] == R2.month_comparison["heterogeneity"]


def test_pipeline_refuses_failed_audit(tmp_path):
    t = wide_table()
    t.loc[0, "JAN"] = "abc"
    with pytest.raises(AuditFailure):
        run_pipeline(write_csv(tmp_path / "bad.csv", t))


def test_missing_values_stay_missing_through_pipeline(tmp_path):
    t = wide_table()
    t.loc[4, "JUL"] = "-999"
    R = run_pipeline(write_csv(tmp_path / "gap.csv", t))
    assert R.monthly["T2M"].isna().sum() == 1
    assert R.month_trends.loc[R.month_trends["MONTH"] == 7, "N_VALID"].iloc[0] == 24
    assert R.month_trends.loc[R.month_trends["MONTH"] == 7, "N_MISSING"].iloc[0] == 1
