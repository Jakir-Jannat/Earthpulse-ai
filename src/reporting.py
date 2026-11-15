"""Export tables/figures and generate the Markdown findings report from computed results."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from src import anomaly_analysis as an
from src import visualizations as V
from src.data_validation import audit_to_json, audit_to_text
from src.pipeline import Results
from src.statistics import benjamini_hochberg


def fmt_p(p: float) -> str:
    """Format a p-value (never rounds to exactly 0)."""
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}" if 0.045 <= p <= 0.055 else f"{p:.3f}"  # extra digits near the 0.05 threshold


def ci(lo: float, hi: float) -> str:
    return f"[{lo:+.2f}, {hi:+.2f}]"


def joint_fdr_table(R: Results) -> pd.DataFrame:
    """Sensitivity: BH across all 12 monthly + 4 seasonal tests together."""
    m = R.month_trends[["MONTH_NAME", "SEN_PER_DECADE", "P_RAW"]].rename(columns={"MONTH_NAME": "SERIES", "P_RAW": "P"})
    m.insert(0, "FAMILY", "month")
    s = R.season_trends[["SERIES", "SEN_PER_DECADE", "P_MK"]].rename(columns={"P_MK": "P"})
    s.insert(0, "FAMILY", "season")
    out = pd.concat([m, s], ignore_index=True)
    out["P_FDR_JOINT_16"] = benjamini_hochberg(out["P"].to_numpy())
    return out


def _trend_phrase(row: dict, p: float) -> str:
    slope, lo, hi = row["SEN_PER_DECADE"], row["CI_LOW_PER_DECADE"], row["CI_HIGH_PER_DECADE"]
    if p < cfg.ALPHA:
        return f"statistically supported {'warming' if slope > 0 else 'cooling'} ({slope:+.2f} °C/decade, 95% CI {ci(lo, hi)}, p = {fmt_p(p)})"
    direction = "positive" if slope > 0 else "negative"
    return (f"a {direction} point estimate ({slope:+.2f} °C/decade, 95% CI {ci(lo, hi)}) that is NOT statistically distinguishable from zero "
            f"(p = {fmt_p(p)})")


def build_report(R: Results) -> str:
    """Compose the findings report. Every number and conclusion branch is computed from R."""
    a, mt, st = R.annual_trend, R.month_trends, R.season_trends
    ar = a["row"]
    joint = joint_fdr_table(R)
    meta = R.audit["metadata"]
    cov = R.audit["coverage"]
    L: list[str] = []
    w = L.append
    w("# EARTHPULSE AI - Scientific Findings Report")
    w("\n*Beyond Average Temperature: Discovering Sylhet's Hidden Seasonal Climate Shifts*\n")
    w(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from `{R.audit['file']['name']}` (SHA-256 `{R.raw.sha256[:16]}...`). "
      "All numbers below are computed by the code in this repository from the uploaded file.\n")
    w("## 0. Data and scope")
    w(f"- Source: NASA POWER, {meta.get('product_line')}, parameter **{meta.get('parameter_code')}** ({meta.get('parameter_description')}, {meta.get('parameter_units')}).")
    w(f"- Location: a **single grid-cell estimate** at {meta.get('latitude')}°N, {meta.get('longitude')}°E; source cell {meta.get('grid_lat_deg')}° x {meta.get('grid_lon_deg')}° (lat x lon), model elevation {meta.get('elevation_m')} m. It does not describe the whole Sylhet district.")
    w(f"- Period: {meta.get('start_date')} to {meta.get('end_date')} ({meta.get('time_standard')}); {cov['valid_monthly_observations']} of {cov['expected_monthly_observations']} monthly values valid; audit status **{R.audit['status']}**.")
    w("- Values are reanalysis (MERRA-2) output, not station observations.\n")

    w("## 1. Is the selected Sylhet location warming over 2001-2025?")
    p_ann = ar["P_MMK"]
    w(f"Annual mean T2M (computed from the 12 monthly values; ANN column not used) shows {_trend_phrase(ar, p_ann)}. "
      f"Mann-Kendall S = {ar['MK_S']:.0f}, Kendall tau = {ar['KENDALL_TAU']:.2f}; Hamed-Rao p = {fmt_p(ar['P_MMK'])} (correction factor {ar['MMK_CORRECTION_FACTOR']:.2f}). "
      f"The OLS companion gives {a['ols']['slope_per_decade']:+.2f} °C/decade (95% CI {ci(a['ols']['ci_low_per_decade'], a['ols']['ci_high_per_decade'])}, p = {fmt_p(a['ols']['p_value'])}, R² = {a['ols']['r_squared']:.2f}). "
      f"Using an unweighted annual mean changes the Sen slope to {R.annual_sensitivity['unweighted_annual']['SEN_PER_DECADE']:+.2f} °C/decade (p = {fmt_p(R.annual_sensitivity['unweighted_annual']['P_MK'])}).\n")

    w("## 2. Is warming uniform throughout the year?")
    het, hs = R.month_comparison["heterogeneity"], R.season_comparison["heterogeneity"]
    pairs, vs = R.month_comparison["pairs"], R.month_comparison["vs_annual"]
    lo_row, hi_row = mt.loc[mt["SEN_PER_DECADE"].idxmin()], mt.loc[mt["SEN_PER_DECADE"].idxmax()]
    w(f"Month-wise Sen slopes range from {lo_row['SEN_PER_DECADE']:+.2f} °C/decade ({lo_row['MONTH_NAME']}) to {hi_row['SEN_PER_DECADE']:+.2f} °C/decade ({hi_row['MONTH_NAME']}). "
      "A spread in point estimates is expected from sampling noise alone with 25 values per month, so it is tested explicitly:")
    w(f"- Global bootstrap test that all 12 monthly slopes are equal: p = {fmt_p(het['p_weighted'])} (precision-weighted; unweighted p = {fmt_p(het['p_unweighted'])}).")
    w(f"- Pairwise month differences (66 pairs, paired-year bootstrap, BH-adjusted): **{int(pairs['SUPPORTED_DIFFERENCE'].sum())}** supported; smallest adjusted p = {fmt_p(pairs['P_BOOT_FDR'].min())}.")
    w(f"- Month slope minus annual slope (12 comparisons): **{int(vs['SUPPORTED_DIFFERENCE'].sum())}** supported after BH; smallest adjusted p = {fmt_p(vs['P_BOOT_FDR'].min())}.")
    if het["p_weighted"] < cfg.ALPHA or pairs["SUPPORTED_DIFFERENCE"].any():
        w("\nConclusion: there is statistical evidence that monthly trend rates differ.\n")
    else:
        w("\nConclusion: the data do **not** provide statistically supported evidence that monthly trend rates differ from one another or from the annual trend. "
          "Uniform warming cannot be rejected, and equally the data are too short to establish that it is uniform.\n")

    w("## 3. Which months show the strongest estimated warming / cooling? (descriptive ranking)")
    top = mt.sort_values("SEN_PER_DECADE", ascending=False)
    w("| Rank | Month | Sen slope (°C/decade) | 95% CI | Raw p | FDR p |")
    w("|---:|---|---:|---|---:|---:|")
    for _, r in top.iterrows():
        w(f"| {int(r['RANK_WARMING'])} | {r['MONTH_NAME']} | {r['SEN_PER_DECADE']:+.3f} | {ci(r['CI_LOW_PER_DECADE'], r['CI_HIGH_PER_DECADE'])} | {fmt_p(r['P_RAW'])} | {fmt_p(r['P_FDR'])} |")
    w("\nRanking by point estimate is descriptive only; it is not evidence that one month changes faster than another (see Section 2).\n")

    w("## 4. Which monthly trends remain supported after FDR correction?")
    sig = mt[mt["SIG_FDR"]]
    raw_sig = mt[mt["SIG_RAW"]]
    if sig.empty:
        w(f"None. {len(raw_sig)} month(s) have raw p < {cfg.ALPHA}, but none survive Benjamini-Hochberg correction across 12 tests.")
    else:
        for _, r in sig.iterrows():
            w(f"- **{r['MONTH_NAME']}**: {r['SEN_PER_DECADE']:+.2f} °C/decade, 95% CI {ci(r['CI_LOW_PER_DECADE'], r['CI_HIGH_PER_DECADE'])}, raw p = {fmt_p(r['P_RAW'])}, BH-adjusted p = {fmt_p(r['P_FDR'])}.")
        w(f"\n{len(raw_sig)} month(s) have raw p < {cfg.ALPHA}; {len(sig)} remain supported after correction. "
          "In a joint BH family of all 16 monthly + seasonal tests, adjusted p for "
          + ", ".join(f"{r['SERIES']} = {fmt_p(r['P_FDR_JOINT_16'])}" for _, r in joint[joint['SERIES'].isin(list(sig['MONTH_NAME']))].iterrows()) + ".")
        w("A single month passing among twelve is a screening result: it merits follow-up with independent data, not a firm conclusion.")
    w("")

    w("## 5. Are seasonal trends different?")
    w("| Season | n | Sen (°C/decade) | 95% CI | p (MK) | p (Hamed-Rao) | BH p (4 seasons) | Lag-1 ACF |")
    w("|---|---:|---:|---|---:|---:|---:|---:|")
    for _, r in st.iterrows():
        w(f"| {r['SERIES']} | {int(r['N'])} | {r['SEN_PER_DECADE']:+.3f} | {ci(r['CI_LOW_PER_DECADE'], r['CI_HIGH_PER_DECADE'])} | {fmt_p(r['P_MK'])} | {fmt_p(r['P_MMK'])} | {fmt_p(r['P_MK_FDR'])} | {r['LAG1_ACF_RANK_DETRENDED']:+.2f} |")
    sig_s = st[st["P_MK_FDR"] < cfg.FDR_ALPHA]
    joint_s = {r["SERIES"]: r["P_FDR_JOINT_16"] for _, r in joint.iterrows()}
    spair = R.season_comparison["pairs"]
    w(f"\nSeasons with BH-adjusted p < {cfg.FDR_ALPHA}: {', '.join(sig_s['SERIES']) if len(sig_s) else 'none'}"
      + (" (joint 16-test BH-adjusted p: " + ", ".join(f"{n} = {fmt_p(joint_s[n])}" for n in sig_s["SERIES"]) + ")" if len(sig_s) else "") + ". "
      f"Global bootstrap test that the four seasonal slopes are equal: p = {fmt_p(hs['p_weighted'])}. "
      f"Pairwise seasonal differences supported after BH: {int(spair['SUPPORTED_DIFFERENCE'].sum())} of {len(spair)}"
      + (f" (closest: {spair.loc[spair['P_BOOT_FDR'].idxmin(), 'A']} vs {spair.loc[spair['P_BOOT_FDR'].idxmin(), 'B']}, adjusted p = {fmt_p(spair['P_BOOT_FDR'].min())})." if len(spair) else ".")
      + " Seasonal and monthly tests use overlapping data, so they are not independent confirmations of each other. "
      "DJF is labelled by its January year (Dec Y-1 + Jan-Feb Y), leaving 24 complete winters.\n")

    w("## 6. Is seasonal amplitude changing?")
    at = R.amplitude_trends
    for _, r in at.iterrows():
        w(f"- {r['SERIES']}: Sen {r['SEN_PER_DECADE']:+.2f} °C/decade, 95% CI {ci(r['CI_LOW_PER_DECADE'], r['CI_HIGH_PER_DECADE'])}, p = {fmt_p(r['P_MK'])} "
          f"({'supported' if r['P_MK'] < cfg.ALPHA else 'not statistically supported'}).")
    w("")

    w("## 7. Secondary analyses")
    cpt = R.change_points
    w(f"- Pettitt change-point test on the annual mean and four seasons: {int(cpt['SUPPORTED'].sum())} series show a change point supported by raw p, trimmed-ends sensitivity and BH correction. "
      f"Smallest raw p = {fmt_p(cpt['P_PETTITT'].min())} ({cpt.loc[cpt['P_PETTITT'].idxmin(), 'SERIES']}); its trimmed-ends p = {fmt_p(cpt.loc[cpt['P_PETTITT'].idxmin(), 'P_PETTITT_TRIMMED'])}. No regime shift is claimed.")
    pc = R.period_comparison
    w(f"- Early ({cfg.EARLY_PERIOD[0]}-{cfg.EARLY_PERIOD[1]}) vs late ({cfg.LATE_PERIOD[0]}-{cfg.LATE_PERIOD[1]}) means (midpoint split fixed a priori): "
      f"{int((pc['P_FDR'] < cfg.FDR_ALPHA).sum())} of {len(pc)} comparisons significant after BH; smallest adjusted p = {fmt_p(pc['P_FDR'].min())}.\n")

    w("## 8. Uncertainties and limitations")
    for t in [
        f"One MERRA-2 grid cell (0.5° x 0.625°, roughly {0.5 * 111.2:.0f} x {0.625 * 111.32 * math.cos(math.radians(meta.get('latitude', 24.9))):.0f} km at this latitude) is not Sylhet district and is not a weather-station record; reanalysis temperatures may carry model biases and inhomogeneities.",
        "Twenty-five annual values per series give low power: real differences smaller than the confidence intervals above cannot be detected.",
        "A 25-year window is short relative to multi-decadal variability (e.g. ENSO, monsoon variability); trends here are not claimed to be long-term climate trends.",
        "Multiple testing: 12 monthly tests were BH-corrected as one family, 4 seasonal tests as another; the joint 16-test sensitivity is reported in Section 4.",
        "Autocorrelation: lag diagnostics and the Hamed-Rao modified MK (inflate-only variant) are reported; with n = 25 the lag estimates are themselves uncertain.",
        "Bootstrap comparisons resample years as exchangeable units and may be optimistic if years are serially dependent.",
        "Statistical association is not attribution: nothing here establishes causes, and no prediction model is implemented or validated.",
    ]:
        w(f"- {t}")
    w("")
    w("## 9. What can and cannot be claimed")
    w("**Justified by this analysis**")
    if p_ann >= cfg.ALPHA:
        w(f"- No statistically detectable linear warming trend in annual mean T2M at this grid cell over {ar['FIRST_YEAR']}-{ar['LAST_YEAR']} (Sen {ar['SEN_PER_DECADE']:+.2f} °C/decade, CI {ci(ar['CI_LOW_PER_DECADE'], ar['CI_HIGH_PER_DECADE'])}).")
    else:
        w(f"- Statistically supported annual warming of {ar['SEN_PER_DECADE']:+.2f} °C/decade at this grid cell.")
    for _, r in sig.iterrows():
        w(f"- A statistically supported {'warming' if r['SEN_PER_DECADE'] > 0 else 'cooling'} trend in {r['MONTH_NAME']} after FDR correction (screening-level).")
    for _, r in sig_s.iterrows():
        j = joint_s[r["SERIES"]]
        w(f"- A statistically supported {'warming' if r['SEN_PER_DECADE'] > 0 else 'cooling'} trend in {r['SERIES']} within the four-season family "
          f"({r['SEN_PER_DECADE']:+.2f} °C/decade, CI {ci(r['CI_LOW_PER_DECADE'], r['CI_HIGH_PER_DECADE'])}); joint 16-test adjusted p = {fmt_p(j)}"
          f"{' (does not meet 0.05 when pooled with monthly tests)' if j >= cfg.FDR_ALPHA else ''}.")
    if het["p_weighted"] >= cfg.ALPHA and not pairs["SUPPORTED_DIFFERENCE"].any():
        w("- Point estimates differ between months, but that difference is not statistically established.")
    w("\n**Not justified**")
    for t in ["Claims about Sylhet district as a whole.", "Claims that any month or season warms *faster* than another unless a supported difference is listed above.",
              "Causal attribution to greenhouse gases, urbanisation, land use or monsoon dynamics.", "Predictions or projections.",
              "A climate regime shift (no change point passed all checks)." if not cpt["SUPPORTED"].any() else "Interpretation of the supported change point beyond a statistical step-shift.",
              "Treating the monthly ranking or the fingerprint shape as a validated climate index."]:
        w(f"- {t}")
    return "\n".join(L) + "\n"


def export_all(R: Results, root: Path = cfg.ROOT, figures: bool = True) -> dict[str, list[str]]:
    """Write processed data, audit, result tables, figures and the report. Raw data is never touched."""
    out: dict[str, list[str]] = {"data": [], "tables": [], "reports": [], "figures": []}
    for d in (cfg.PROCESSED_DIR, cfg.TABLE_DIR, cfg.REPORT_DIR, cfg.FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    p = cfg.PROCESSED_DIR / "sylhet_t2m_monthly_tidy.csv"
    R.monthly.to_csv(p, index=False, date_format="%Y-%m-%d")
    out["data"].append(str(p))
    prov = {"source": "NASA POWER (MERRA-2) monthly, source native resolution", "raw_file": R.audit["file"], "metadata": R.audit["metadata"],
            "processing": ["Header located dynamically", "ANN column excluded", "-999/non-numeric/implausible -> NaN with QC_FLAG",
                           "Reindexed to complete 2001-2025 x 12 month grid", "Month-start dates (YYYY-MM-01)"], "run_info": R.run_info}
    (cfg.PROCESSED_DIR / "data_provenance.json").write_text(json.dumps(prov, indent=2, default=str))

    (cfg.REPORT_DIR / "data_audit.json").write_text(audit_to_json(R.audit))
    (cfg.REPORT_DIR / "data_audit.txt").write_text(audit_to_text(R.audit) + "\n")
    out["reports"] += [str(cfg.REPORT_DIR / "data_audit.json"), str(cfg.REPORT_DIR / "data_audit.txt")]

    anom = an.monthly_anomalies(R.monthly, cfg.DEFAULT_BASELINE)
    tables: dict[str, pd.DataFrame] = {
        "annual_means": R.annual, "annual_trend": pd.DataFrame([R.annual_trend["row"]]),
        "monthly_stats": R.month_stats, "monthly_trends": R.month_trends, "seasonal_means": R.seasonal,
        "seasonal_trends": R.season_trends, "monthly_anomalies_baseline_2001_2025": anom,
        "month_slope_pairwise_differences": R.month_comparison["pairs"], "month_slope_vs_annual": R.month_comparison["vs_annual"],
        "season_slope_pairwise_differences": R.season_comparison["pairs"], "season_slope_vs_annual": R.season_comparison["vs_annual"],
        "amplitude_by_year": R.amplitude, "amplitude_trends": R.amplitude_trends, "change_points": R.change_points,
        "early_vs_late": R.period_comparison, "acf_diagnostics": R.acf_table, "joint_fdr_16_tests": joint_fdr_table(R),
        "heterogeneity_tests": pd.DataFrame([{"FAMILY": "months", **{k: v for k, v in R.month_comparison["heterogeneity"].items()}},
                                             {"FAMILY": "seasons", **{k: v for k, v in R.season_comparison["heterogeneity"].items()}}]),
    }
    for name, df in tables.items():
        f = cfg.TABLE_DIR / f"{name}.csv"
        df.to_csv(f, index=False)
        out["tables"].append(str(f))

    rep = cfg.REPORT_DIR / "scientific_findings_report.md"
    rep.write_text(build_report(R), encoding="utf-8")
    out["reports"].append(str(rep))

    if figures:
        bl = cfg.DEFAULT_BASELINE
        amat = an.anomaly_matrix(anom)
        figs = {
            "01_annual_trend": V.fig_annual_trend(R.annual_trend, int(R.annual["T2M"].isna().sum())),
            "02_monthly_climatology": V.fig_climatology(R.month_stats, bl),
            "03_month_year_heatmap": V.fig_month_year_heatmap(R.matrix),
            "04_anomaly_heatmap": V.fig_anomaly_heatmap(amat, bl),
            "05_monthly_sen_slopes": V.fig_monthly_slopes(R.month_trends),
            "06_seasonal_fingerprint": V.fig_fingerprint(R.month_trends),
            "07_seasonal_trends": V.fig_seasonal_trends(R.season_trends, R.annual_trend["row"]),
            "08_seasonal_anomalies": V.fig_seasonal_anomalies(R.seasonal_mat, bl),
            "09_amplitude": V.fig_amplitude(R.amplitude, R.amplitude_trends),
            "10_pvalues_raw_vs_fdr": V.fig_pvalues(R.month_trends),
        }
        for n, f in figs.items():
            out["figures"] += [str(x) for x in V.save_figure(f, n)]
    return out


def key_findings(R: Results) -> list[str]:
    """Short, computed, verified-only statements for the dashboard overview."""
    ar, mt, st = R.annual_trend["row"], R.month_trends, R.season_trends
    het, pairs = R.month_comparison["heterogeneity"], R.month_comparison["pairs"]
    out = [f"Annual mean T2M: Sen slope {ar['SEN_PER_DECADE']:+.2f} °C/decade (95% CI {ci(ar['CI_LOW_PER_DECADE'], ar['CI_HIGH_PER_DECADE'])}), "
           f"Mann–Kendall p = {fmt_p(ar['P_MMK'])} - " + ("statistically supported." if ar["P_MMK"] < cfg.ALPHA else "not statistically distinguishable from zero.")]
    sig = mt[mt["SIG_FDR"]]
    if sig.empty:
        out.append("No calendar month has a trend that survives Benjamini–Hochberg correction across the 12 tests.")
    for _, r in sig.iterrows():
        out.append(f"{r['MONTH_NAME']}: {r['SEN_PER_DECADE']:+.2f} °C/decade (95% CI {ci(r['CI_LOW_PER_DECADE'], r['CI_HIGH_PER_DECADE'])}), raw p = {fmt_p(r['P_RAW'])}, FDR-adjusted p = {fmt_p(r['P_FDR'])}.")
    for _, r in st[st["P_MK_FDR"] < cfg.FDR_ALPHA].iterrows():
        out.append(f"{r['SERIES']}: {r['SEN_PER_DECADE']:+.2f} °C/decade (95% CI {ci(r['CI_LOW_PER_DECADE'], r['CI_HIGH_PER_DECADE'])}), "
                   f"p = {fmt_p(r['P_MK'])}, adjusted over 4 seasons p = {fmt_p(r['P_MK_FDR'])}.")
    out.append(f"Is warming uniform across months? Monthly slopes span {mt['SEN_PER_DECADE'].min():+.2f} to {mt['SEN_PER_DECADE'].max():+.2f} °C/decade, "
               f"but the global test of equal slopes gives p = {fmt_p(het['p_weighted'])} and {int(pairs['SUPPORTED_DIFFERENCE'].sum())} of 66 pairwise differences are supported after correction.")
    a = R.amplitude_trends.iloc[0]
    out.append(f"Seasonal amplitude (annual range): {a['SEN_PER_DECADE']:+.2f} °C/decade, p = {fmt_p(a['P_MK'])} - "
               + ("a supported change." if a["P_MK"] < cfg.ALPHA else "no statistically supported change."))
    return out
