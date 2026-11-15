"""EARTHPULSE AI - Streamlit dashboard.

Run:  streamlit run app.py
All numbers shown are computed from the NASA POWER CSV (default: data/raw/...).
"""
from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import config as cfg
from src import anomaly_analysis as an
from src import visualizations as V
from src.data_loader import DataLoadError
from src.data_validation import audit_to_json, audit_to_text
from src.pipeline import AuditFailure, Results, run_pipeline
from src.preprocessing import PreprocessingError
from src.reporting import build_report, ci, fmt_p, joint_fdr_table, key_findings

st.set_page_config(page_title="EARTHPULSE AI", layout="wide", initial_sidebar_state="collapsed")

CSS = f"""
<style>
html, body, [class*="css"], .stApp {{ font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", Arial, sans-serif; }}
.block-container {{ padding-top: 2.4rem; max-width: 1340px; }}
h1 {{ font-weight: 650; letter-spacing: -0.025em; margin-bottom: 0.1rem; }}
h2, h3 {{ font-weight: 600; letter-spacing: -0.01em; }}
.ep-sub {{ color: {V.MUTED}; font-size: 1.05rem; margin-bottom: 1.2rem; }}
.ep-note {{ border-left: 3px solid {V.ACCENT}; background: {V.PANEL}; padding: .65rem .95rem; color: #b9c6d3; font-size: .93rem; border-radius: 0 4px 4px 0; margin: .4rem 0 1rem; }}
.ep-warn {{ border-left: 3px solid {V.WARM}; background: {V.PANEL}; padding: .65rem .95rem; color: #d6c3ab; font-size: .93rem; border-radius: 0 4px 4px 0; margin: .4rem 0 1rem; }}
[data-testid="stMetric"] {{ background: {V.PANEL}; border: 1px solid {V.GRID}; padding: 12px 16px; border-radius: 4px; }}
[data-testid="stMetricLabel"] {{ color: {V.MUTED}; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {V.GRID}; flex-wrap: wrap; }}
.stTabs [data-baseweb="tab"] {{ padding: 8px 14px; }}
footer {{ visibility: hidden; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PLOT_CFG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}}


def note(text: str, warn: bool = False) -> None:
    st.markdown(f"<div class='{'ep-warn' if warn else 'ep-note'}'>{text}</div>", unsafe_allow_html=True)


def show(fig, key: str, caption: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOT_CFG, key=key)
    if caption:
        note(caption)


@st.cache_data(show_spinner="Running the full analysis...")
def cached_results(path: str, mtime: float) -> Results:
    return run_pipeline(Path(path))


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ----------------------------------------------------------------- data source
default_path = cfg.RAW_DIR / cfg.RAW_FILENAME
with st.sidebar:
    st.header("Data source")
    up = st.file_uploader("Analyse a different NASA POWER monthly CSV", type=["csv"], help="The original file is never modified. Processing happens on a temporary copy.")
    st.caption("Default: the bundled Sylhet T2M file in data/raw/.")

try:
    if up is not None:
        tmp = Path(tempfile.gettempdir()) / f"earthpulse_{abs(hash(up.name + str(up.size)))}.csv"
        tmp.write_bytes(up.getvalue())
        R = cached_results(str(tmp), tmp.stat().st_mtime)
        source_label = up.name
    else:
        if not default_path.exists():
            st.error(f"Raw data file not found at {default_path}. Place the NASA POWER CSV in data/raw/ or upload one in the sidebar.")
            st.stop()
        R = cached_results(str(default_path), default_path.stat().st_mtime)
        source_label = default_path.name
except (DataLoadError, PreprocessingError, AuditFailure) as exc:
    st.error(f"This file cannot be analysed: {exc}")
    st.stop()

meta, cov = R.audit["metadata"], R.audit["coverage"]
mt, st_tr, ar = R.month_trends, R.season_trends, R.annual_trend["row"]
bl_default = cfg.DEFAULT_BASELINE
if R.matrix.index.min() > bl_default[0] or R.matrix.index.max() < bl_default[1]:
    bl_default = (int(R.matrix.index.min()), int(R.matrix.index.max()))

st.title(cfg.PROJECT_TITLE)
st.markdown(f"<div class='ep-sub'>{cfg.PROJECT_SUBTITLE}</div>", unsafe_allow_html=True)

tabs = st.tabs(["Executive overview", "Seasonal climate fingerprint", "Monthly & seasonal analysis",
                "Climate anomaly explorer", "Statistical evidence", "Data & methodology"])

# ============================================================ 1. OVERVIEW
with tabs[0]:
    st.subheader("Is Sylhet warming uniformly throughout the year?")
    c = st.columns(4)
    c[0].metric("Grid cell", f"{meta.get('latitude')}°N, {meta.get('longitude')}°E")
    c[1].metric("Period", f"{int(R.matrix.index.min())}–{int(R.matrix.index.max())}")
    c[2].metric("Valid monthly values", f"{cov['valid_monthly_observations']} / {cov['expected_monthly_observations']}")
    c[3].metric("Data audit", R.audit["status"].replace("_", " ").title())
    note(f"NASA POWER, {meta.get('product_line')}. Parameter {meta.get('parameter_code')} ({meta.get('parameter_description')}, {meta.get('parameter_units')}). "
         f"Source resolution {meta.get('grid_lat_deg')}° × {meta.get('grid_lon_deg')}° (lat × lon). This is a <b>single grid-cell estimate</b>, not district-wide coverage and not a station record.")
    left, right = st.columns([3, 2])
    with left:
        show(V.fig_annual_trend(R.annual_trend, int(R.annual["T2M"].isna().sum())), "ov_annual",
             f"The annual trend estimate is {ar['SEN_PER_DECADE']:+.2f} °C/decade with a 95% CI of {ci(ar['CI_LOW_PER_DECADE'], ar['CI_HIGH_PER_DECADE'])}. "
             + ("The interval excludes zero." if (ar['CI_LOW_PER_DECADE'] > 0 or ar['CI_HIGH_PER_DECADE'] < 0) else "The interval includes zero, so a flat trend is compatible with the data."))
    with right:
        st.markdown("#### Key verified findings")
        for f in key_findings(R):
            st.markdown(f"- {f}")
        note("These statements are generated from the computed results. They describe statistical trends at one grid cell and do not establish causes.", warn=True)

# ============================================================ 2. FINGERPRINT
with tabs[1]:
    st.subheader("Sylhet seasonal climate fingerprint")
    st.markdown("The twelve measured month-wise Sen slopes, shown directly. No composite index is defined.")
    l, r = st.columns([1, 1])
    with l:
        show(V.fig_fingerprint(mt), "fp_polar")
    with r:
        order = st.radio("Table order", ["Ranked by slope", "Calendar order"], horizontal=True, key="fp_order")
        tbl = mt.sort_values("SEN_PER_DECADE", ascending=False) if order == "Ranked by slope" else mt.sort_values("MONTH")
        view = pd.DataFrame({
            "Rank": tbl["RANK_WARMING"], "Month": tbl["MONTH_NAME"], "Sen (°C/decade)": tbl["SEN_PER_DECADE"],
            "95% CI": [ci(a, b) for a, b in zip(tbl["CI_LOW_PER_DECADE"], tbl["CI_HIGH_PER_DECADE"])],
            "p (raw)": tbl["P_RAW"], "p (FDR)": tbl["P_FDR"],
            "Supported after FDR": np.where(tbl["SIG_FDR"], "Yes", "No"),
        })
        st.dataframe(view, hide_index=True, width="stretch", height=470, column_config={
            "Sen (°C/decade)": st.column_config.NumberColumn(format="%+.3f"),
            "p (raw)": st.column_config.NumberColumn(format="%.4f"), "p (FDR)": st.column_config.NumberColumn(format="%.4f")})
        st.caption("Rank is descriptive. Only the FDR column supports a statistical statement about a single month.")
    n_sig = int(mt["SIG_FDR"].sum())
    show(V.fig_monthly_slopes(mt), "fp_dot",
         f"{n_sig} of 12 months remain supported after Benjamini–Hochberg correction"
         + (f" ({', '.join(mt.loc[mt['SIG_FDR'], 'MONTH_NAME'])})." if n_sig else ".")
         + f" {int(mt['CI_EXCLUDES_ZERO'].sum())} month(s) have a 95% CI that excludes zero before any multiplicity adjustment is considered.")
    with st.expander("What the fingerprint means and what it cannot establish", expanded=True):
        st.markdown(
            "- **What it shows:** the estimated rate of change of each calendar month's mean temperature at this grid cell, 2001–2025, with distribution-free 95% confidence intervals.\n"
            "- **How to read it:** bars are measured Sen slopes from the zero ring; vertical lines are confidence intervals; bright outlined bars passed FDR correction.\n"
            "- **It cannot establish:** that months differ from each other (see the Statistical evidence tab), that the pattern persists beyond 2025, or why it occurs. "
            "With 25 values per month, a spread in point estimates is expected from noise alone.\n"
            "- **Status:** this is a visual summary of measured slopes. It is not an established climate index.")

# ============================================================ 3. MONTHLY & SEASONAL
with tabs[2]:
    st.subheader("Monthly and seasonal structure")
    show(V.fig_month_year_heatmap(R.matrix), "ms_heat", "Warm months cluster April–October; the coldest values are in December–February. Colour shows absolute temperature, not change.")
    a1, a2 = st.columns(2)
    with a1:
        show(V.fig_climatology(R.month_stats, bl_default), "ms_clim", f"Hottest month on average: {R.month_stats.loc[R.month_stats['MEAN'].idxmax(), 'MONTH_NAME']}; coldest: {R.month_stats.loc[R.month_stats['MEAN'].idxmin(), 'MONTH_NAME']}.")
    with a2:
        show(V.fig_boxplot_months(R.matrix), "ms_box", "Winter months vary more from year to year than monsoon months, which affects how precisely their trends can be estimated.")
    b1, b2 = st.columns(2)
    with b1:
        show(V.fig_seasonal_climatology(R.seasonal_mat, bl_default), "ms_sclim")
    with b2:
        show(V.fig_seasonal_anomalies(R.seasonal_mat, bl_default), "ms_sanom")
    show(V.fig_seasonal_trends(st_tr, R.annual_trend["row"]), "ms_strend",
         "Seasons whose interval excludes zero: " + (", ".join(st_tr.loc[(st_tr["CI_LOW_PER_DECADE"] > 0) | (st_tr["CI_HIGH_PER_DECADE"] < 0), "SERIES"]) or "none") + ". Overlapping intervals do not by themselves show that two seasons differ; see the bootstrap comparison below.")
    sv = R.season_comparison["vs_annual"]
    c1, c2 = st.columns([3, 2])
    with c1:
        show(V.fig_vs_annual(sv[sv["SERIES"] != "Annual mean"], "Seasonal slope minus annual-mean slope"), "ms_vsann")
    with c2:
        st.dataframe(R.season_comparison["pairs"].round(4), hide_index=True, width="stretch", height=300)
        st.caption("Pairwise seasonal slope differences (°C/decade), BH-adjusted over 6 pairs.")
    show(V.fig_amplitude(R.amplitude, R.amplitude_trends), "ms_amp", "Amplitude is the contrast between the hottest and coldest months of each year, plus the amplitude of the fitted annual harmonic.")

# ============================================================ 4. ANOMALY EXPLORER
with tabs[3]:
    st.subheader("Climate anomaly explorer")
    ymin, ymax = int(R.matrix.index.min()), int(R.matrix.index.max())
    c1, c2, c3 = st.columns([2, 2, 3])
    bl = c1.slider("Baseline years (inclusive)", ymin, ymax, bl_default, key="bl", help="Anomaly = value − mean of the same calendar month over these years. At least 10 years.")
    yr = c2.slider("Years shown", ymin, ymax, (ymin, ymax), key="yr")
    months = c3.multiselect("Months shown", list(range(1, 13)), default=list(range(1, 13)), format_func=lambda m: cfg.MONTH_NAMES[m - 1], key="mo")
    try:
        an.validate_baseline(bl, R.monthly["YEAR"])
        anom = an.monthly_anomalies(R.monthly, bl)
        note(f"<b>Baseline: {bl[0]}–{bl[1]}</b> ({bl[1] - bl[0] + 1} years), calendar-month means. Shorter baselines shift anomalies up or down but never change the trend tests, which always use the full record.")
        sel = anom[(anom["YEAR"] >= yr[0]) & (anom["YEAR"] <= yr[1]) & (anom["MONTH"].isin(months))]
        if sel.empty:
            st.info("No data for this selection. Choose at least one month and a valid year range.")
        else:
            view = st.radio("View", ["Time series", "Heatmap"], horizontal=True, key="anom_view")
            if view == "Time series":
                show(V.fig_anomaly_timeseries(sel, bl), "an_ts")
            else:
                am = an.anomaly_matrix(sel)
                show(V.fig_anomaly_heatmap(am.reindex(columns=sorted(months)), bl), "an_hm")
            m1, m2, m3 = st.columns(3)
            m1.metric("Mean anomaly in selection", f"{round(float(sel['ANOMALY'].mean()), 2) + 0.0:+.2f} °C")
            m2.metric("Warmest month in selection", f"{sel.loc[sel['ANOMALY'].idxmax(), 'MONTH_NAME'][:3]} {int(sel.loc[sel['ANOMALY'].idxmax(), 'YEAR'])}", f"{sel['ANOMALY'].max():+.2f} °C", delta_color="off")
            m3.metric("Coolest month in selection", f"{sel.loc[sel['ANOMALY'].idxmin(), 'MONTH_NAME'][:3]} {int(sel.loc[sel['ANOMALY'].idxmin(), 'YEAR'])}", f"{sel['ANOMALY'].min():+.2f} °C", delta_color="off")
        note("A trend line is deliberately not offered for arbitrary sub-windows: picking start and end years after looking at the chart is a common way to manufacture or hide trends.", warn=True)
    except ValueError as exc:
        st.warning(str(exc))

# ============================================================ 5. STATISTICAL EVIDENCE
with tabs[4]:
    st.subheader("Statistical evidence")
    with st.expander("Test methodology", expanded=False):
        st.markdown(
            "- **Mann–Kendall** (two-sided, tie and continuity corrected) tests for a monotonic trend; **Sen's slope** estimates its size; the 95% CI uses the Gilbert rank method.\n"
            "- **Month-wise:** one 25-value annual series per calendar month; **Benjamini–Hochberg** FDR across the 12 tests.\n"
            "- **Seasons:** DJF is labelled by its January year (Dec Y−1 + Jan–Feb Y), giving 24 complete winters. Seasonal and annual means are day-weighted and require all months.\n"
            "- **Autocorrelation:** diagnostics on Sen-detrended ranks; **Hamed–Rao modified MK** applied once per series (never on pre-whitened data), inflate-only so it cannot make a result more significant.\n"
            f"- **Comparing slopes:** paired-year bootstrap ({cfg.BOOTSTRAP_N:,} resamples, seed {cfg.RANDOM_SEED}) so cross-month dependence is preserved; BH-adjusted; ranking is never treated as significance.\n"
            "- **Change point:** Pettitt test, secondary, with trimmed-ends sensitivity.")
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("#### Annual trend")
        o = R.annual_trend["ols"]
        st.dataframe(pd.DataFrame({
            "Method": ["Mann–Kendall / Sen", "Hamed–Rao MMK", "OLS (descriptive)"],
            "Slope (°C/decade)": [ar["SEN_PER_DECADE"], ar["SEN_PER_DECADE"], o["slope_per_decade"]],
            "95% CI": [ci(ar["CI_LOW_PER_DECADE"], ar["CI_HIGH_PER_DECADE"])] * 2 + [ci(o["ci_low_per_decade"], o["ci_high_per_decade"])],
            "p": [ar["P_MK"], ar["P_MMK"], o["p_value"]]}), hide_index=True, width="stretch",
            column_config={"Slope (°C/decade)": st.column_config.NumberColumn(format="%+.3f"), "p": st.column_config.NumberColumn(format="%.4f")})
        st.caption(f"Kendall tau = {ar['KENDALL_TAU']:.3f}, S = {ar['MK_S']:.0f}, n = {ar['N']}. OLS R² = {o['r_squared']:.3f}; OLS residual lag-1 autocorrelation = {o['residual_lag1_autocorr']:+.2f}.")
    with e2:
        show(V.fig_pvalues(mt), "ev_p", "Bars above the dashed line are not significant at 0.05. The adjusted bars show which months remain after accounting for 12 simultaneous tests.")
    st.markdown("#### Month-wise results")
    mview = mt[["MONTH_NAME", "N_VALID", "N_MISSING", "MK_S", "KENDALL_TAU", "MK_Z", "P_RAW", "P_FDR", "SEN_PER_YEAR", "SEN_PER_DECADE", "CI_LOW_PER_DECADE", "CI_HIGH_PER_DECADE", "LAG1_ACF", "P_MMK", "P_MMK_FDR"]]
    st.dataframe(mview, hide_index=True, width="stretch")
    st.markdown("#### Seasonal results")
    st.dataframe(st_tr, hide_index=True, width="stretch")
    st.markdown("#### Do monthly trend rates differ?")
    h = R.month_comparison["heterogeneity"]
    k = st.columns(3)
    k[0].metric("Global test, equal monthly slopes", f"p = {fmt_p(h['p_weighted'])}", "precision-weighted bootstrap", delta_color="off")
    k[1].metric("Unweighted sensitivity", f"p = {fmt_p(h['p_unweighted'])}")
    k[2].metric("Supported pairwise differences", f"{int(R.month_comparison['pairs']['SUPPORTED_DIFFERENCE'].sum())} of 66")
    cc1, cc2 = st.columns(2)
    with cc1:
        show(V.fig_vs_annual(R.month_comparison["vs_annual"][R.month_comparison["vs_annual"]["SERIES"] != "Annual mean"], "Monthly slope minus annual-mean slope"), "ev_vsann")
    with cc2:
        pr = R.month_comparison["pairs"].sort_values("P_BOOT").head(10)
        st.dataframe(pr.round(4), hide_index=True, width="stretch", height=430)
        st.caption("Ten smallest pairwise p-values among 66; compare raw and BH-adjusted columns.")
    st.markdown("#### Autocorrelation diagnostics")
    series = st.selectbox("Series", R.acf_table["SERIES"].unique(), key="acf_sel")
    show(V.fig_acf(R.acf_table, series), "ev_acf", "Bars outside the dashed bounds indicate a statistically notable lag. Correction factors are listed in the seasonal and month-wise tables.")
    st.markdown("#### Multiple-testing sensitivity (12 months + 4 seasons pooled)")
    st.dataframe(joint_fdr_table(R), hide_index=True, width="stretch", height=300)
    st.markdown("#### Secondary analyses")
    st.dataframe(R.change_points, hide_index=True, width="stretch")
    st.caption("Pettitt change-point test. A regime shift is only suggested if SUPPORTED is true (raw, trimmed and FDR-adjusted p all pass); with 25 values a step cannot be separated from a steady trend.")
    st.dataframe(R.period_comparison, hide_index=True, width="stretch", height=300)
    st.caption(f"Early ({cfg.EARLY_PERIOD[0]}–{cfg.EARLY_PERIOD[1]}) vs late ({cfg.LATE_PERIOD[0]}–{cfg.LATE_PERIOD[1]}) comparison; midpoint split fixed in advance. Exploratory.")
    note("<b>Limitations.</b> One reanalysis grid cell; 25 annual values per series (low power); short window relative to multi-decadal variability; bootstrap treats years as exchangeable; "
         "statistical association is not attribution; no predictive model is implemented.", warn=True)

# ============================================================ 6. DATA & METHODOLOGY
with tabs[5]:
    st.subheader("Data and methodology")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("#### Original NASA POWER metadata")
        st.code("\n".join(meta.get("raw_header", [])), language="text")
        st.markdown("#### Coordinate and grid")
        st.markdown(f"Point {meta.get('latitude')}°N, {meta.get('longitude')}°E · source cell {meta.get('grid_lat_deg')}° × {meta.get('grid_lon_deg')}° · model elevation {meta.get('elevation_m')} m · time standard {meta.get('time_standard')}.")
        st.markdown("#### Processing")
        st.markdown("1. Locate the table header dynamically; keep every cell as text.\n2. Exclude the ANN column from monthly data (used only as a consistency check).\n"
                    "3. Convert −999, non-numeric and implausible entries to missing with a QC flag.\n4. Reindex to every month of the period; absent months remain explicit missing rows.\n"
                    "5. Month-start dates; day-weighted annual and seasonal means (complete cases only).")
    with d2:
        st.markdown("#### Data quality report")
        st.code(audit_to_text(R.audit), language="text")
        st.dataframe(R.monthly["QC_FLAG"].value_counts().rename_axis("QC flag").reset_index(name="Months"), hide_index=True, width="stretch")
    st.markdown("#### Downloads")
    buf = io.BytesIO()
    tables = {"monthly_tidy": R.monthly, "monthly_trends": R.month_trends, "seasonal_trends": R.season_trends, "annual_means": R.annual,
              "seasonal_means": R.seasonal, "month_slope_pairwise_differences": R.month_comparison["pairs"], "change_points": R.change_points, "joint_fdr_16_tests": joint_fdr_table(R)}
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n, df in tables.items():
            z.writestr(f"{n}.csv", df.to_csv(index=False))
        z.writestr("data_audit.json", audit_to_json(R.audit))
        z.writestr("scientific_findings_report.md", build_report(R))
    dc = st.columns(4)
    dc[0].download_button("Processed monthly data (CSV)", csv_bytes(R.monthly), "sylhet_t2m_monthly_tidy.csv", "text/csv")
    dc[1].download_button("Month-wise trends (CSV)", csv_bytes(R.month_trends), "monthly_trends.csv", "text/csv")
    dc[2].download_button("Seasonal trends (CSV)", csv_bytes(R.season_trends), "seasonal_trends.csv", "text/csv")
    dc[3].download_button("Everything (ZIP)", buf.getvalue(), "earthpulse_results.zip", "application/zip")
    dd = st.columns(2)
    dd[0].download_button("Audit report (JSON)", audit_to_json(R.audit), "data_audit.json", "application/json")
    dd[1].download_button("Findings report (Markdown)", build_report(R), "scientific_findings_report.md", "text/markdown")
    st.markdown("#### Reproducibility")
    st.json({**R.run_info, "source_file": source_label}, expanded=False)
    st.code("pip install -r requirements.txt\npython run_analysis.py\nstreamlit run app.py\npytest", language="bash")
