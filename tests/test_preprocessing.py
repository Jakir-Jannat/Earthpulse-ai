import numpy as np
import pandas as pd
import pytest

import config as cfg
from src.data_loader import DataLoadError, read_nasa_power_csv
from src.data_validation import audit_raw_file
from src.preprocessing import PreprocessingError, month_year_matrix, wide_to_long
from tests.helpers import HEADER, wide_table, write_csv


def test_wide_to_long_gives_300_records_sorted():
    long = wide_to_long(wide_table())
    assert len(long) == 300
    assert list(long.columns[:5]) == ["DATE", "YEAR", "MONTH", "MONTH_NAME", "T2M"]
    assert long["DATE"].is_monotonic_increasing
    assert long["T2M"].notna().all()


def test_dates_are_valid_month_starts():
    long = wide_to_long(wide_table())
    assert (long["DATE"].dt.day == 1).all()
    assert long["DATE"].iloc[0] == pd.Timestamp("2001-01-01")
    assert long["DATE"].iloc[-1] == pd.Timestamp("2025-12-01")
    assert (long["DATE"].dt.year == long["YEAR"]).all() and (long["DATE"].dt.month == long["MONTH"]).all()


def test_leap_year_days_in_month():
    long = wide_to_long(wide_table(range(2000, 2005)), 2000, 2004)
    feb = long[long["MONTH"] == 2].set_index("YEAR")["DAYS_IN_MONTH"]
    assert feb[2000] == 29 and feb[2004] == 29 and feb[2001] == 28
    assert long["DAYS_IN_MONTH"].sum() == 366 + 365 * 3 + 366


def test_ann_column_is_never_used_as_a_monthly_value():
    a = wide_to_long(wide_table(ann=-1.0))
    b = wide_to_long(wide_table(ann=999.0))
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 300 and a["T2M"].max() < 40


def test_sentinel_nonnumeric_and_implausible_become_nan_with_flags():
    t = wide_table()
    t.loc[0, "JAN"] = "-999"
    t.loc[1, "FEB"] = "abc"
    t.loc[2, "MAR"] = "88.5"
    long = wide_to_long(t).set_index(["YEAR", "MONTH"])
    assert np.isnan(long.loc[(2001, 1), "T2M"]) and long.loc[(2001, 1), "QC_FLAG"] == "missing_sentinel"
    assert long.loc[(2002, 2), "QC_FLAG"] == "nonnumeric"
    assert long.loc[(2003, 3), "QC_FLAG"] == "implausible"
    assert long["T2M"].isna().sum() == 3


def test_duplicate_years_raise():
    t = wide_table()
    t = pd.concat([t, t.iloc[[3]]], ignore_index=True)
    with pytest.raises(PreprocessingError, match="Duplicate"):
        wide_to_long(t)


def test_missing_year_is_explicit_nan_rows():
    t = wide_table().drop(index=5).reset_index(drop=True)  # drop 2006
    long = wide_to_long(t)
    assert len(long) == 300
    sub = long[long["YEAR"] == 2006]
    assert sub["T2M"].isna().all() and (sub["QC_FLAG"] == "missing_row").all()
    assert month_year_matrix(long).shape == (25, 12)


def test_non_integer_year_raises():
    t = wide_table()
    t.loc[0, "YEAR"] = "20x1"
    with pytest.raises(PreprocessingError):
        wide_to_long(t)


@pytest.mark.parametrize("content", ["", "\n\n", "just,some,text\n1,2,3\n"])
def test_empty_or_invalid_csv(tmp_path, content):
    p = tmp_path / "bad.csv"
    p.write_text(content)
    with pytest.raises(DataLoadError):
        read_nasa_power_csv(p)


def test_header_only_and_missing_file(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text(HEADER + "PARAMETER,YEAR," + ",".join(cfg.MONTH_ABBR) + ",ANN\n")
    with pytest.raises(DataLoadError, match="no data rows"):
        read_nasa_power_csv(p)
    with pytest.raises(DataLoadError, match="not found"):
        read_nasa_power_csv(tmp_path / "nope.csv")


def test_header_is_located_dynamically(tmp_path):
    extra = "EXTRA LINE 1\nEXTRA LINE 2\nEXTRA LINE 3\n"
    r = read_nasa_power_csv(write_csv(tmp_path / "a.csv", wide_table(), header=extra + HEADER))
    assert r.header_line_index == 9 + 3
    assert r.metadata["latitude"] == pytest.approx(24.8949) and r.metadata["declared_missing_value"] == -999.0
    assert r.metadata["parameter_code"] == "T2M" and r.metadata["grid_lon_deg"] == 0.625


def test_audit_flags_problems(tmp_path):
    t = wide_table().drop(index=0)                 # 2001 missing
    t = pd.concat([t, t.iloc[[2]]], ignore_index=True)  # duplicate
    t.loc[4, "JUN"] = "-999"
    a = audit_raw_file(read_nasa_power_csv(write_csv(tmp_path / "b.csv", t)))
    codes = {i["code"] for i in a["issues"]}
    assert a["status"] == "FAIL" and {"DUP_YEARS", "MISSING_YEARS", "SENTINEL"} <= codes


def test_audit_passes_clean_synthetic(tmp_path):
    a = audit_raw_file(read_nasa_power_csv(write_csv(tmp_path / "c.csv", wide_table())))
    assert a["status"] == "PASS" and a["coverage"]["valid_monthly_observations"] == 300
