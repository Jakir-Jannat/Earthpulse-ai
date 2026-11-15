"""Central configuration for EARTHPULSE AI.

Every methodological constant lives here so that results are reproducible and
no magic numbers are scattered through the code base.
"""
from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent
RAW_DIR: Path = ROOT / "data" / "raw"
PROCESSED_DIR: Path = ROOT / "data" / "processed"
FIG_DIR: Path = ROOT / "outputs" / "figures"
TABLE_DIR: Path = ROOT / "outputs" / "tables"
REPORT_DIR: Path = ROOT / "outputs" / "reports"

RAW_FILENAME: str = "POWER_Point_Monthly_20010101_20251231_024d89N_091d87E_LST.csv"

# --- Expectations used for validation (never used to overwrite data) -------
EXPECTED_PARAMETER: str = "T2M"
EXPECTED_START_YEAR: int = 2001
EXPECTED_END_YEAR: int = 2025
EXPECTED_LAT: float = 24.8949
EXPECTED_LON: float = 91.8687
COORD_TOLERANCE_DEG: float = 1e-3
MISSING_SENTINEL: float = -999.0
PLAUSIBLE_T2M_RANGE_C: tuple[float, float] = (-10.0, 50.0)

MONTH_ABBR: tuple[str, ...] = (
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
)
MONTH_NAMES: tuple[str, ...] = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

# --- Meteorological seasons (as specified by the project brief) ------------
# DJF is labelled by the year of its January and February (the "winter-ending"
# year): DJF 2002 = Dec 2001 + Jan 2002 + Feb 2002.
SEASONS: dict[str, tuple[int, ...]] = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJAS": (6, 7, 8, 9),
    "ON": (10, 11),
}
SEASON_LABELS: dict[str, str] = {
    "DJF": "Winter (DJF)",
    "MAM": "Pre-monsoon (MAM)",
    "JJAS": "Monsoon (JJAS)",
    "ON": "Post-monsoon (ON)",
}

# --- Statistics ------------------------------------------------------------
ALPHA: float = 0.05            # two-sided significance level
CI_LEVEL: float = 0.95         # confidence level for Sen-slope intervals
FDR_ALPHA: float = 0.05        # Benjamini-Hochberg target FDR
BOOTSTRAP_N: int = 5000        # paired year-resampling replicates
RANDOM_SEED: int = 20260       # fixed for deterministic bootstrap results
MIN_VALID_FOR_TREND: int = 10  # minimum non-missing values for a trend test

# --- Baselines / splits ----------------------------------------------------
DEFAULT_BASELINE: tuple[int, int] = (2001, 2025)
# A-priori midpoint split (not data-driven): 2001-2012 vs 2013-2025.
EARLY_PERIOD: tuple[int, int] = (2001, 2012)
LATE_PERIOD: tuple[int, int] = (2013, 2025)

PROJECT_TITLE: str = "EARTHPULSE AI"
PROJECT_SUBTITLE: str = "Beyond Average Temperature: Discovering Sylhet's Hidden Seasonal Climate Shifts"

# --- Aggregation -----------------------------------------------------------
# NASA's ANN column agrees with the day-weighted mean of the 12 monthly values
# (max diff 0.005 degC in this file) better than with the simple mean (0.035),
# so annual and seasonal means are day-weighted by default.
DAY_WEIGHTED_AGGREGATION: bool = True
