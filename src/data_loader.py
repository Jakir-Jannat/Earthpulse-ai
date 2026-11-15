"""Read a raw NASA POWER monthly CSV without modifying it."""
from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from config import MONTH_ABBR


class DataLoadError(ValueError):
    """Raised when the file cannot be parsed as a NASA POWER monthly CSV."""


@dataclass
class RawPowerFile:
    """Container for the parsed (but unmodified) NASA POWER file."""

    path: Path
    sha256: str
    size_bytes: int
    header_lines: list[str]
    header_line_index: int
    metadata: dict[str, Any]
    table: pd.DataFrame  # every cell kept as the original string
    n_text_lines: int = 0
    notes: list[str] = field(default_factory=list)


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of a file (proves the raw data is untouched)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_header_index(lines: list[str]) -> int:
    """Locate the table header dynamically (the line with YEAR and the month columns)."""
    for i, line in enumerate(lines):
        tokens = [t.strip().upper() for t in line.split(",")]
        if "YEAR" in tokens and "JAN" in tokens and "DEC" in tokens:
            return i
    raise DataLoadError(
        "Could not find a table header containing YEAR, JAN ... DEC. "
        "Is this a NASA POWER monthly CSV?"
    )


def parse_metadata(header_lines: list[str]) -> dict[str, Any]:
    """Extract coordinates, period, elevation, resolution and parameter details."""
    text = "\n".join(header_lines)
    meta: dict[str, Any] = {"raw_header": [ln.rstrip() for ln in header_lines]}
    non_marker = [ln.strip() for ln in header_lines if "HEADER" not in ln]
    if non_marker:
        meta["product_line"] = non_marker[0]
    m = re.search(r"Latitude\s+(-?\d+(?:\.\d+)?)\s+Longitude\s+(-?\d+(?:\.\d+)?)", text)
    if m:
        meta["latitude"], meta["longitude"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"Dates \(month/day/year\):\s*(\S+)\s+through\s+(\S+)\s+in\s+(\w+)", text)
    if m:
        meta["start_date"], meta["end_date"], meta["time_standard"] = m.groups()
    m = re.search(r"Elevation from (\S+):.*?=\s*([\d.]+)\s*meters", text)
    if m:
        meta["elevation_source"], meta["elevation_m"] = m.group(1), float(m.group(2))
    m = re.search(r"Average for ([\d.]+)\s*x\s*([\d.]+)\s*degree", text)
    if m:
        meta["grid_lat_deg"], meta["grid_lon_deg"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"missing source data[^\n]*?:\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        meta["declared_missing_value"] = float(m.group(1))
    m = re.search(r"^([A-Z0-9_]+)\s{2,}(.+?)\s*\(([^)]+)\)\s*$", text, re.MULTILINE)
    if m:
        meta["parameter_code"] = m.group(1)
        meta["parameter_description"] = m.group(2).strip()
        meta["parameter_units"] = m.group(3).strip()
    return meta


def read_nasa_power_csv(path: str | Path) -> RawPowerFile:
    """Read a NASA POWER monthly CSV, keeping header lines and all cells as raw strings."""
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"File not found: {path}")
    if path.stat().st_size == 0:
        raise DataLoadError(f"File is empty: {path.name}")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DataLoadError(f"File is not valid UTF-8 text: {exc}") from exc
    lines = text.splitlines()
    if not any(ln.strip() for ln in lines):
        raise DataLoadError(f"File contains no text: {path.name}")
    idx = _find_header_index(lines)
    body = [ln for ln in lines[idx:] if ln.strip()]
    try:
        table = pd.read_csv(io.StringIO("\n".join(body)), dtype=str, keep_default_na=False)
    except Exception as exc:  # pandas raises several parser error types
        raise DataLoadError(f"Could not parse the data table: {exc}") from exc
    table.columns = [str(c).strip().upper() for c in table.columns]
    table = table.apply(lambda col: col.str.strip())
    if table.empty:
        raise DataLoadError("The data table has a header but no data rows.")
    missing = [m for m in MONTH_ABBR if m not in table.columns]
    if missing:
        raise DataLoadError(f"Missing month columns: {missing}")
    return RawPowerFile(
        path=path, sha256=sha256_of(path), size_bytes=path.stat().st_size,
        header_lines=lines[:idx], header_line_index=idx,
        metadata=parse_metadata(lines[:idx]), table=table, n_text_lines=len(lines),
    )
