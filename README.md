# EARTHPULSE AI

**Beyond Average Temperature: Discovering Sylhet's Hidden Seasonal Climate Shifts**
NASA Space Apps Challenge 2026 - *Be An Earth System Trend Detective!*

> **Research question:** Is Sylhet warming uniformly throughout the year, or are specific months and seasons changing at different rates?

EARTHPULSE AI is a reproducible statistical analysis of NASA POWER monthly 2 m air temperature (T2M) for one grid cell, with a Streamlit dashboard. Findings are **computed from the data and written by code**; this README deliberately contains no results. See `outputs/reports/scientific_findings_report.md` after running the analysis.

## Data

| Item | Value |
|---|---|
| Source | NASA POWER, *Source Native Resolution Monthly and Annual*, MERRA-2 reanalysis |
| Parameter | `T2M`, MERRA-2 Temperature at 2 Meters, °C |
| Location | 24.8949°N, 91.8687°E (Sylhet, Bangladesh), **one grid-cell estimate** |
| Source resolution | 0.5° x 0.625° (lat x lon); model elevation 103.47 m |
| Period | 2001-01 to 2025-12, LST |
| Missing marker | `-999` (treated as missing, never as a temperature) |

Place the original file in `data/raw/` (default name in `config.py`). It is never modified (its SHA-256 is recorded in every audit). Raw CSVs are git-ignored by default; remove that rule in `.gitignore` only if you are sure you may publish the file.

Cite: NASA Prediction Of Worldwide Energy Resources (POWER) Project; Gelaro et al. (2017), MERRA-2, *J. Climate* 30, 5419-5454.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_analysis.py                  # audit, tables, figures (HTML), report -> data/processed, outputs/
streamlit run app.py                    # dashboard at http://localhost:8501
pip install -r requirements-dev.txt && pytest           # 46 tests
python scripts/wald_calibration_check.py                # why a cluster-Wald heterogeneity test was rejected
```

`python run_analysis.py path/to/other_POWER_file.csv` analyses a different NASA POWER monthly CSV. Figures export as interactive HTML by default; PNG needs `kaleido` plus Chrome (`plotly_get_chrome`). Every chart in the app also has a camera button for PNG download.

## Deploy on Streamlit Community Cloud

1. Push the repository to GitHub (raw CSV excluded by `.gitignore`).
2. On share.streamlit.io choose the repo, branch and `app.py`.
3. The app needs the data file. Either (a) commit the CSV after confirming you may publish it, or (b) let visitors upload it in the sidebar ("Analyse a different NASA POWER monthly CSV"). Without a file the app shows a clear message and stops.

## Project layout

```
app.py                  Streamlit dashboard (6 tabs)
run_analysis.py         CLI: full pipeline + exports
config.py               every constant: paths, seasons, alpha, seed, baselines
src/data_loader.py      dynamic header detection, metadata parsing, SHA-256
src/data_validation.py  audit (JSON + text)
src/preprocessing.py    wide -> long, -999 handling, QC flags, date grid
src/exploratory_analysis.py   climatology, annual means, early-vs-late
src/seasonal_analysis.py      DJF/MAM/JJAS/ON, seasonal trends, amplitude
src/anomaly_analysis.py       baseline-relative anomalies
src/statistics.py       Mann-Kendall, Sen + CI, Hamed-Rao, BH-FDR, bootstrap
src/trend_analysis.py   annual/month-wise trends, slope comparison
src/change_point.py     Pettitt
src/pipeline.py         orchestration -> Results
src/visualizations.py   Plotly figures
src/reporting.py        tables, figures, generated Markdown report
tests/                  46 tests (fixtures are synthetic and only for tests)
scripts/                calibration check for a rejected method
```

## Methodology (all choices are in `config.py` or documented here)

1. **Audit** (no silent drops): parameter, units, coordinates, year coverage, duplicates, `-999`, non-numeric and implausible values (outside -10 to 50 °C), ANN consistency. Bad values become `NaN` with a `QC_FLAG`; the pipeline refuses to analyse a file whose audit has errors.
2. **Tidy series:** `DATE, YEAR, MONTH, MONTH_NAME, T2M` (+ `QC_FLAG`, `DAYS_IN_MONTH`) on a complete 2001-2025 x 12 grid. The **ANN column is never used as data**; it is only compared against the computed annual mean.
3. **Aggregation:** annual and seasonal means are *day-weighted* and need every month present (NASA's ANN matches the day-weighted mean more closely than the simple mean; the unweighted result is reported as a sensitivity run).
4. **Seasons:** DJF = Dec-Feb, MAM, JJAS, ON. DJF is labelled by its January year (Dec Y-1 + Jan-Feb Y), so DJF 2001 and DJF 2026 are incomplete and excluded (24 winters).
5. **Trends:** two-sided Mann-Kendall (tie + continuity correction), Sen's slope with the Gilbert rank CI, per decade = 10 x per year. OLS is a descriptive companion.
6. **Multiple testing:** Benjamini-Hochberg over the 12 monthly tests; seasons form a separate family of 4; a pooled 16-test sensitivity is reported.
7. **Autocorrelation:** lag diagnostics on Sen-detrended ranks; Hamed-Rao modified MK applied once per series (no prior pre-whitening), floored at factor 1 so it can only be more cautious.
8. **Comparing slopes:** paired-year bootstrap (5,000 resamples, seed 20260) for month-vs-month, month-vs-annual and season comparisons, BH-adjusted; global test that all slopes are equal (precision-weighted bootstrap Q, unweighted as sensitivity). Ranking by point estimate is never treated as evidence.
9. **Amplitude:** yearly hottest-minus-coldest month, and first-harmonic amplitude; each tested for trend.
10. **Change points (secondary):** Pettitt, trimmed-ends sensitivity, BH across series; flagged only if all three pass.
11. **Anomalies:** value minus the mean of the same calendar month over a baseline (default 2001-2025, selectable, minimum 10 years). The baseline never affects trend tests.

## Limitations

- One MERRA-2 grid cell. Not district-wide, not a station record; reanalysis may contain model bias and inhomogeneity.
- 25 values per series gives low statistical power; absence of significance is not evidence of uniformity.
- A 25-year window is short relative to multi-decadal climate variability.
- The bootstrap treats years as exchangeable; serial dependence would make it optimistic. Hamed-Rao lag estimates are noisy at n = 25.
- Seasonal and monthly tests use overlapping data and are not independent confirmations.
- Association is not attribution. No prediction model is implemented or validated, and none is claimed.
- The fingerprint is a display of 12 measured slopes, not an established climate index.

## Reproducibility

Fixed seed, pinned method choices, input SHA-256 and package versions written to `data/processed/data_provenance.json`. The report timestamp is the only non-deterministic text.
