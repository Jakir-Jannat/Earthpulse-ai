"""Data audit: structural and quality checks on the raw NASA POWER file."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

import config as cfg
from src.data_loader import RawPowerFile


def _issue(severity: str, code: str, message: str, detail: Any = None) -> dict[str, Any]:
    return {"severity": severity, "code": code, "message": message, "detail": detail}


def audit_raw_file(raw: RawPowerFile) -> dict[str, Any]:
    """Audit the parsed raw file. Returns a JSON-serialisable dict.

    Nothing is modified: every anomaly is listed with its location.
    """
    t, meta, issues = raw.table, raw.metadata, []
    years_num = pd.to_numeric(t["YEAR"], errors="coerce")
    years = [int(y) for y in years_num.dropna() if float(y).is_integer()]

    # ---- parameter -------------------------------------------------------
    params_in_rows = sorted(set(t["PARAMETER"])) if "PARAMETER" in t.columns else []
    if meta.get("parameter_code") != cfg.EXPECTED_PARAMETER:
        issues.append(_issue("error", "PARAM_HEADER", f"Header parameter is {meta.get('parameter_code')!r}, expected 'T2M'."))
    if params_in_rows != [cfg.EXPECTED_PARAMETER]:
        issues.append(_issue("error", "PARAM_ROWS", f"Row parameter values: {params_in_rows}"))
    if meta.get("parameter_units") != "C":
        issues.append(_issue("warning", "UNITS", f"Units are {meta.get('parameter_units')!r}, expected 'C'."))

    # ---- coordinates -----------------------------------------------------
    lat, lon = meta.get("latitude"), meta.get("longitude")
    coords_ok = (lat is not None and lon is not None
                 and abs(lat - cfg.EXPECTED_LAT) <= cfg.COORD_TOLERANCE_DEG
                 and abs(lon - cfg.EXPECTED_LON) <= cfg.COORD_TOLERANCE_DEG)
    if not coords_ok:
        issues.append(_issue("warning", "COORDS", f"Coordinates {lat}, {lon} differ from expected {cfg.EXPECTED_LAT}, {cfg.EXPECTED_LON}."))

    # ---- year coverage ---------------------------------------------------
    expected_years = list(range(cfg.EXPECTED_START_YEAR, cfg.EXPECTED_END_YEAR + 1))
    dup_years = sorted({y for y in years if years.count(y) > 1})
    missing_years = [y for y in expected_years if y not in years]
    extra_years = [y for y in sorted(set(years)) if y not in expected_years]
    if dup_years:
        issues.append(_issue("error", "DUP_YEARS", f"Duplicate year rows: {dup_years}"))
    if missing_years:
        issues.append(_issue("error", "MISSING_YEARS", f"Years missing from file: {missing_years}"))
    if extra_years:
        issues.append(_issue("warning", "EXTRA_YEARS", f"Years outside expected range: {extra_years}"))

    # ---- cell-level checks ----------------------------------------------
    lo, hi = cfg.PLAUSIBLE_T2M_RANGE_C
    sentinel, nonnumeric, implausible, empty = [], [], [], []
    n_valid = 0
    for _, row in t.iterrows():
        y = row["YEAR"]
        for abbr in cfg.MONTH_ABBR:
            raw_val = row[abbr]
            try:
                v = float(raw_val)
            except ValueError:
                (empty if raw_val == "" else nonnumeric).append({"year": y, "month": abbr, "raw": raw_val})
                continue
            if v == cfg.MISSING_SENTINEL:
                sentinel.append({"year": y, "month": abbr})
            elif not np.isfinite(v) or not lo <= v <= hi:
                implausible.append({"year": y, "month": abbr, "value": v})
            else:
                n_valid += 1
    for lst, sev, code, label in [
        (sentinel, "warning", "SENTINEL", "values equal to -999 (treated as missing)"),
        (nonnumeric, "error", "NONNUMERIC", "non-numeric monthly entries"),
        (empty, "warning", "EMPTY", "empty monthly entries"),
        (implausible, "error", "IMPLAUSIBLE", f"values outside [{lo}, {hi}] degC"),
    ]:
        if lst:
            issues.append(_issue(sev, code, f"{len(lst)} {label}", lst))

    expected_obs = len(expected_years) * 12
    if n_valid != expected_obs:
        issues.append(_issue("warning", "VALID_COUNT", f"{n_valid} valid monthly values vs {expected_obs} expected."))

    # ---- ANN consistency (QA only; ANN is never used as data) ------------
    ann_check: dict[str, Any] = {"used_in_analysis": False}
    if "ANN" in t.columns:
        ann = pd.to_numeric(t["ANN"], errors="coerce").replace(cfg.MISSING_SENTINEL, np.nan)
        mon = t[list(cfg.MONTH_ABBR)].apply(pd.to_numeric, errors="coerce").replace(cfg.MISSING_SENTINEL, np.nan)
        simple = mon.mean(axis=1, skipna=False)
        days = np.array([[31, 29 if (int(y) % 4 == 0 and (int(y) % 100 != 0 or int(y) % 400 == 0)) else 28,
                          31, 30, 31, 30, 31, 31, 30, 31, 30, 31] for y in years_num.fillna(2001)])
        weighted = (mon.to_numpy() * days).sum(axis=1) / days.sum(axis=1)
        ann_check.update(
            max_abs_diff_vs_simple_mean=float(np.nanmax(np.abs(ann - simple))),
            max_abs_diff_vs_day_weighted_mean=float(np.nanmax(np.abs(ann - weighted))),
            note="ANN is NASA's annual summary; used only as a consistency check.")

    status = ("FAIL" if any(i["severity"] == "error" for i in issues)
              else "PASS_WITH_WARNINGS" if issues else "PASS")
    return {
        "status": status,
        "file": {"name": raw.path.name, "sha256": raw.sha256, "size_bytes": raw.size_bytes,
                 "text_lines": raw.n_text_lines, "header_lines": raw.header_line_index,
                 "table_header_line_number": raw.header_line_index + 1},
        "metadata": {k: v for k, v in meta.items() if k != "raw_header"},
        "table": {"columns": list(t.columns), "n_rows": int(len(t)), "years_found": sorted(set(years)),
                  "parameters_in_rows": params_in_rows},
        "coverage": {"expected_years": len(expected_years), "years_found": len(set(years)),
                     "expected_monthly_observations": expected_obs, "valid_monthly_observations": n_valid,
                     "missing_or_invalid_monthly_observations": expected_obs - n_valid,
                     "duplicate_years": dup_years, "missing_years": missing_years,
                     "coordinates_match_expected": bool(coords_ok)},
        "cell_checks": {"sentinel_-999": sentinel, "nonnumeric": nonnumeric, "empty": empty, "implausible": implausible},
        "ann_column_check": ann_check,
        "issues": issues,
    }


def audit_to_text(a: dict[str, Any]) -> str:
    """Human-readable audit report."""
    m, c, f = a["metadata"], a["coverage"], a["file"]
    lines = [
        "EARTHPULSE AI - DATA AUDIT REPORT", "=" * 36,
        f"Status            : {a['status']}",
        f"File              : {f['name']}  ({f['size_bytes']} bytes)",
        f"SHA-256           : {f['sha256']}",
        f"Table header at   : line {f['table_header_line_number']} ({f['header_lines']} metadata lines above)",
        f"Product           : {m.get('product_line')}",
        f"Parameter         : {m.get('parameter_code')} - {m.get('parameter_description')} ({m.get('parameter_units')})",
        f"Coordinate        : lat {m.get('latitude')}, lon {m.get('longitude')} (single grid-cell estimate)",
        f"Source resolution : {m.get('grid_lat_deg')} x {m.get('grid_lon_deg')} degree (lat x lon), MERRA-2",
        f"Elevation (model) : {m.get('elevation_m')} m",
        f"Period / time     : {m.get('start_date')} to {m.get('end_date')} ({m.get('time_standard')})",
        f"Table             : {a['table']['n_rows']} rows; columns {', '.join(a['table']['columns'])}",
        f"Years found       : {c['years_found']} of {c['expected_years']} expected",
        f"Monthly obs       : {c['valid_monthly_observations']} valid of {c['expected_monthly_observations']} expected",
        f"Duplicates/missing: duplicate years {c['duplicate_years'] or 'none'}; missing years {c['missing_years'] or 'none'}",
        f"-999 markers      : {len(a['cell_checks']['sentinel_-999'])}",
        f"Non-numeric       : {len(a['cell_checks']['nonnumeric'])}",
        f"Implausible       : {len(a['cell_checks']['implausible'])}",
    ]
    ac = a["ann_column_check"]
    if "max_abs_diff_vs_simple_mean" in ac:
        lines.append("ANN column        : NOT used as data. Max |ANN - mean of 12 months| = "
                     f"{ac['max_abs_diff_vs_simple_mean']:.3f} degC (simple mean), "
                     f"{ac['max_abs_diff_vs_day_weighted_mean']:.3f} degC (day-weighted).")
    lines.append("")
    lines.append("Issues:" if a["issues"] else "Issues: none")
    for i in a["issues"]:
        lines.append(f"  [{i['severity'].upper()}] {i['code']}: {i['message']}")
    return "\n".join(lines)


def audit_to_json(a: dict[str, Any]) -> str:
    return json.dumps(a, indent=2, default=str)
