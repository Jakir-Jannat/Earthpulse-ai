import numpy as np
import pandas as pd
import pytest

import config as cfg
from src.preprocessing import wide_to_long
from src.seasonal_analysis import annual_amplitude, month_to_season, season_year, seasonal_matrix, seasonal_series
from tests.helpers import wide_table


def test_month_to_season_mapping():
    got = {m: month_to_season(m) for m in range(1, 13)}
    assert got == {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJAS", 7: "JJAS", 8: "JJAS", 9: "JJAS", 10: "ON", 11: "ON"}


def test_december_belongs_to_following_djf():
    assert season_year(2001, 12) == 2002 and season_year(2002, 1) == 2002 and season_year(2002, 2) == 2002
    assert season_year(2002, 11) == 2002


def _monthly_with(values: dict[tuple[int, int], float]) -> pd.DataFrame:
    long = wide_to_long(wide_table())
    for (y, m), v in values.items():
        long.loc[(long["YEAR"] == y) & (long["MONTH"] == m), "T2M"] = v
    return long


def test_djf_uses_previous_december_and_is_day_weighted():
    long = _monthly_with({(2001, 12): 10.0, (2002, 1): 20.0, (2002, 2): 30.0})
    s = seasonal_series(long).set_index(["SEASON", "SEASON_YEAR"])
    days = np.array([31, 31, 28])  # Dec 2001, Jan 2002, Feb 2002
    assert s.loc[("DJF", 2002), "T2M"] == pytest.approx(np.average([10, 20, 30], weights=days))
    assert s.loc[("DJF", 2002), "N_MONTHS"] == 3


def test_incomplete_boundary_winters_are_nan():
    s = seasonal_series(wide_to_long(wide_table())).set_index(["SEASON", "SEASON_YEAR"])
    assert np.isnan(s.loc[("DJF", 2001), "T2M"]) and s.loc[("DJF", 2001), "N_MONTHS"] == 2
    assert np.isnan(s.loc[("DJF", 2026), "T2M"]) and s.loc[("DJF", 2026), "N_MONTHS"] == 1
    mat = seasonal_matrix(seasonal_series(wide_to_long(wide_table())))
    assert mat["DJF"].notna().sum() == 24 and mat["MAM"].notna().sum() == 25 and mat["JJAS"].notna().sum() == 25 and mat["ON"].notna().sum() == 25


def test_missing_month_makes_season_nan():
    long = _monthly_with({(2010, 7): np.nan})
    s = seasonal_series(long).set_index(["SEASON", "SEASON_YEAR"])
    assert np.isnan(s.loc[("JJAS", 2010), "T2M"]) and s.loc[("JJAS", 2010), "N_MONTHS"] == 3


def test_season_lengths_follow_config():
    s = seasonal_series(wide_to_long(wide_table()))
    for k, months in cfg.SEASONS.items():
        assert (s.loc[s["SEASON"] == k, "N_EXPECTED"] == len(months)).all()


def test_amplitude_of_known_cycle():
    mat = pd.DataFrame([[20 + 5 * np.cos(2 * np.pi * k / 12) for k in range(12)]] * 3, index=[2001, 2002, 2003], columns=range(1, 13))
    amp = annual_amplitude(mat, None)
    assert amp["HARMONIC"].iloc[0] == pytest.approx(5.0) and amp["RANGE"].iloc[0] == pytest.approx(10.0)


def test_amplitude_nan_for_incomplete_year():
    mat = pd.DataFrame([[1.0] * 11 + [np.nan]], index=[2001], columns=range(1, 13))
    assert annual_amplitude(mat, None)[["RANGE", "HARMONIC"]].isna().all().all()
