"""Command-line entry point: run the full analysis and write all outputs.

    python run_analysis.py [path/to/POWER_...csv] [--no-figures]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config as cfg
from src.data_loader import DataLoadError
from src.pipeline import AuditFailure, run_pipeline
from src.preprocessing import PreprocessingError
from src.reporting import export_all


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="EARTHPULSE AI analysis pipeline")
    ap.add_argument("csv", nargs="?", default=str(cfg.RAW_DIR / cfg.RAW_FILENAME), help="NASA POWER monthly CSV")
    ap.add_argument("--no-figures", action="store_true", help="skip figure export")
    args = ap.parse_args(argv)
    try:
        R = run_pipeline(Path(args.csv))
    except (DataLoadError, PreprocessingError, AuditFailure) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    written = export_all(R, figures=not args.no_figures)
    print(f"Audit status: {R.audit['status']}")
    for k, v in written.items():
        print(f"{k}: {len(v)} files")
    print(f"Report: {cfg.REPORT_DIR / 'scientific_findings_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
