"""Reproducible conversion of NASA's wide table into a tidy monthly series."""
from __future__ import annotations

import calendar

import numpy as np
import pandas as pd

import config as cfg


class PreprocessingError(ValueError):
    """Raised when the raw table cannot be safely converted."""


def _to_number(value: str) -> float:
    """Parse one raw cell; returns NaN for non-numeric text."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def wide_to_long(
    table: pd.DataFrame,
    start_year: int = cfg.EXPECTED_START_YEAR,
    end_year: int = cfg.EXPECTED_END_YEAR,
) -> pd.DataFrame:
    """Convert NASA POWER wide format (YEAR, JAN..DEC, ANN) to tidy monthly records.

    * Only the twelve month columns are used. The ANN column is never melted,
      so an annual value can not leak into the monthly series.
    * -999 and non-numeric entries become NaN, and implausible values are
      masked to NaN. The reason is kept in ``QC_FLAG`` so nothing disappears silently.
    * The output is reindexed to every month of ``start_year``..``end_year``.
      Months absent from the file therefore appear as explicit NaN rows
      with QC_FLAG ``missing_row``.

    Raises
    ------
    PreprocessingError
        If YEAR is missing/non-integer, or the same year appears twice.
    """
    if "YEAR" not in table.columns:
        raise PreprocessingError("The table has no YEAR column.")
    years = pd.to_numeric(table["YEAR"], errors="coerce")
    if years.isna().any() or (years % 1 != 0).any():
        bad = table.loc[years.isna() | (years % 1 != 0), "YEAR"].tolist()
        raise PreprocessingError(f"Non-integer YEAR values found: {bad}")
    years = years.astype(int)
    dup = sorted(set(years[years.duplicated()]))
    if dup:
        raise PreprocessingError(
            f"Duplicate year rows found for {dup}. Resolve them in the source file "
            "before analysis; values are never merged or dropped silently.")

    records: list[dict] = []
    lo, hi = cfg.PLAUSIBLE_T2M_RANGE_C
    for year, (_, row) in zip(years, table.iterrows()):
        for month, abbr in enumerate(cfg.MONTH_ABBR, start=1):
            raw = row[abbr]
            val = _to_number(raw)
            flag = "ok"
            if np.isnan(val):
                flag = "nonnumeric" if str(raw).strip() not in ("", "nan") else "empty"
            elif val == cfg.MISSING_SENTINEL:
                val, flag = np.nan, "missing_sentinel"
            elif not (lo <= val <= hi):
                val, flag = np.nan, "implausible"
            records.append({"YEAR": int(year), "MONTH": month, "T2M": val, "QC_FLAG": flag})
    long = pd.DataFrame.from_records(records)

    grid = pd.MultiIndex.from_product(
        [range(start_year, end_year + 1), range(1, 13)], names=["YEAR", "MONTH"])
    long = long.set_index(["YEAR", "MONTH"])
    extra = long.index.difference(grid)
    outside = long.loc[extra].reset_index() if len(extra) else long.iloc[0:0].reset_index()
    long = long.reindex(grid)
    long["QC_FLAG"] = long["QC_FLAG"].fillna("missing_row")
    long = long.reset_index()
    if len(outside):
        long.attrs["years_outside_expected_range"] = sorted(set(outside["YEAR"]))

    long["DATE"] = pd.to_datetime(dict(year=long["YEAR"], month=long["MONTH"], day=1))
    long["MONTH_NAME"] = long["MONTH"].map(lambda m: cfg.MONTH_NAMES[m - 1])
    long["DAYS_IN_MONTH"] = [calendar.monthrange(y, m)[1] for y, m in zip(long["YEAR"], long["MONTH"])]
    long = long.sort_values("DATE").reset_index(drop=True)
    return long[["DATE", "YEAR", "MONTH", "MONTH_NAME", "T2M", "QC_FLAG", "DAYS_IN_MONTH"]]


def month_year_matrix(monthly: pd.DataFrame) -> pd.DataFrame:
    """Year (rows) x calendar month (columns 1..12) matrix of T2M, NaN kept."""
    return monthly.pivot(index="YEAR", columns="MONTH", values="T2M").sort_index()
