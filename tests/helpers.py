"""Shared synthetic fixtures. Used ONLY by tests, never to produce reported findings."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

HEADER = """-BEGIN HEADER-
NASA/POWER Source Native Resolution Monthly and Annual 
Dates (month/day/year): 01/01/2001 through 12/31/2025 in LST
Location: Latitude  24.8949   Longitude 91.8687 
Elevation from MERRA-2: Average for 0.5 x 0.625 degree lat/lon region = 103.47 meters
The value for missing source data that cannot be computed or is outside of the sources availability range: -999 
Parameter(s): 
T2M     MERRA-2 Temperature at 2 Meters (C) 
-END HEADER-
"""


def wide_table(years=range(2001, 2026), base=None, ann=-1.0) -> pd.DataFrame:
    """A wide NASA-style table of strings with a smooth seasonal cycle."""
    rows = []
    for y in years:
        vals = [round(24 + 5 * np.cos(2 * np.pi * (m - 7) / 12) + 0.01 * (y - 2001), 2) for m in range(12)]
        if base is not None:
            vals = list(base)
        rows.append({"PARAMETER": "T2M", "YEAR": str(y), **{a: str(v) for a, v in zip(cfg.MONTH_ABBR, vals)}, "ANN": str(ann)})
    return pd.DataFrame(rows)


def write_csv(path: Path, table: pd.DataFrame, header: str = HEADER) -> Path:
    path.write_text(header + ",".join(table.columns) + "\n" + "\n".join(",".join(r) for r in table.astype(str).to_numpy()) + "\n")
    return path
