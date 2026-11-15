"""Plotly figures. Every function returns a ``go.Figure`` so it works in the app and in exports."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import config as cfg

# Restrained dark earth-observation palette
BG, PANEL, GRID, TEXT, MUTED = "#0b1118", "#121b25", "#233140", "#e4ebf2", "#8a9bad"
WARM, COOL, NEUTRAL, ACCENT = "#e9a35b", "#5aa8e8", "#7f8fa0", "#5fd0c0"
DIVERGING = [[0.0, "#2f7fc4"], [0.25, "#6fb0e0"], [0.5, "#1a2531"], [0.75, "#eab271"], [1.0, "#d9722b"]]
FONT = "Avenir Next, Segoe UI, Helvetica Neue, Arial, sans-serif"


def _base(fig: go.Figure, title: str, subtitle: str = "", height: int = 440) -> go.Figure:
    text = f"<b>{title}</b>" + (f"<br><span style='font-size:12px;color:{MUTED}'>{subtitle}</span>" if subtitle else "")
    fig.update_layout(
        title=dict(text=text, x=0.0, xanchor="left", font=dict(size=16, color=TEXT)),
        paper_bgcolor=BG, plot_bgcolor=PANEL, font=dict(family=FONT, color=TEXT, size=12),
        height=height, margin=dict(l=60, r=30, t=85 if subtitle else 60, b=55),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=GRID, font=dict(color=TEXT)),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickcolor=GRID)
    return fig


def _missing_note(n_missing: int) -> str:
    return "no missing values" if n_missing == 0 else f"{n_missing} missing values left blank (not interpolated)"


def fig_annual_trend(annual_trend: dict, n_missing: int = 0) -> go.Figure:
    """Annual mean T2M with Sen line and OLS fit + 95% CI band of the mean."""
    at = annual_trend
    y, v = at["years"], at["values"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[y, y[::-1]], y=np.r_[at["ols_band_high"], at["ols_band_low"][::-1]], fill="toself",
                             fillcolor="rgba(95,208,192,0.14)", line=dict(width=0), hoverinfo="skip", name="OLS 95% CI of fitted mean"))
    fig.add_trace(go.Scatter(x=y, y=v, mode="lines+markers", name="Annual mean T2M", line=dict(color=TEXT, width=1.2),
                             marker=dict(size=7, color=WARM), hovertemplate="%{x}: %{y:.2f} °C<extra></extra>"))
    fig.add_trace(go.Scatter(x=y, y=at["sen_fit"], mode="lines", name="Sen's slope line", line=dict(color=ACCENT, width=2.5)))
    fig.add_trace(go.Scatter(x=y, y=at["ols_fit"], mode="lines", name="OLS fit (descriptive)", line=dict(color=COOL, width=1.5, dash="dash")))
    s = at["row"]
    sub = (f"Sen {s['SEN_PER_DECADE']:+.2f} °C/decade (95% CI {s['CI_LOW_PER_DECADE']:+.2f} to {s['CI_HIGH_PER_DECADE']:+.2f}), "
           f"Mann–Kendall p = {s['P_MK']:.3f}; {_missing_note(n_missing)}")
    _base(fig, "Annual mean 2 m air temperature, Sylhet grid cell", sub)
    fig.update_xaxes(title="Year", dtick=2)
    fig.update_yaxes(title="Annual mean T2M (°C)")
    return fig


def fig_climatology(month_stats: pd.DataFrame, baseline: tuple[int, int], monthly_df: pd.DataFrame | None = None) -> go.Figure:
    """Monthly climatology with ±1 SD band and min/max range."""
    ms = month_stats
    x = ms["MONTH_NAME"].str[:3]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=ms["MAX"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=ms["MIN"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(127,143,160,0.14)", name="Min–max range"))
    fig.add_trace(go.Scatter(x=x, y=ms["MEAN"] + ms["STD"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=ms["MEAN"] - ms["STD"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(233,163,91,0.22)", name="±1 SD"))
    fig.add_trace(go.Scatter(x=x, y=ms["MEAN"], mode="lines+markers", name="Mean", line=dict(color=WARM, width=2.5), marker=dict(size=7),
                             customdata=np.c_[ms["MEDIAN"], ms["STD"], ms["MIN"], ms["MAX"]],
                             hovertemplate="%{x}<br>Mean %{y:.2f} °C<br>Median %{customdata[0]:.2f}<br>SD %{customdata[1]:.2f}<br>Min %{customdata[2]:.2f} / Max %{customdata[3]:.2f}<extra></extra>"))
    _base(fig, "Long-term monthly climatology of T2M", f"All calendar months {baseline[0]}–{baseline[1]}; n = {int(ms['N'].min())}–{int(ms['N'].max())} values per month")
    fig.update_yaxes(title="T2M (°C)")
    fig.update_xaxes(title="Calendar month")
    return fig


def fig_month_year_heatmap(matrix: pd.DataFrame) -> go.Figure:
    """Absolute T2M, year x month."""
    fig = go.Figure(go.Heatmap(z=matrix.to_numpy(), x=[cfg.MONTH_NAMES[m - 1][:3] for m in matrix.columns], y=matrix.index,
                               colorscale="Turbo", colorbar=dict(title="°C"), hovertemplate="%{y} %{x}: %{z:.2f} °C<extra></extra>"))
    _base(fig, "Monthly mean T2M by year", f"Raw values (°C); {_missing_note(int(matrix.isna().sum().sum()))}", height=620)
    fig.update_yaxes(title="Year", autorange="reversed", dtick=2)
    fig.update_xaxes(title="Calendar month")
    return fig


def fig_anomaly_heatmap(amat: pd.DataFrame, baseline: tuple[int, int]) -> go.Figure:
    """Monthly anomaly heatmap (year x month), diverging scale centred on 0."""
    lim = float(np.nanmax(np.abs(amat.to_numpy())))
    fig = go.Figure(go.Heatmap(z=amat.to_numpy(), x=[cfg.MONTH_NAMES[m - 1][:3] for m in amat.columns], y=amat.index,
                               zmin=-lim, zmax=lim, zmid=0, colorscale=DIVERGING, colorbar=dict(title="°C"),
                               hovertemplate="%{y} %{x}: %{z:+.2f} °C vs baseline<extra></extra>"))
    _base(fig, "Monthly T2M anomalies by year", f"Anomaly = monthly T2M − mean of the same calendar month over {baseline[0]}–{baseline[1]}; {_missing_note(int(amat.isna().sum().sum()))}", height=620)
    fig.update_yaxes(title="Year", autorange="reversed", dtick=2)
    fig.update_xaxes(title="Calendar month")
    return fig


def fig_anomaly_timeseries(anom: pd.DataFrame, baseline: tuple[int, int]) -> go.Figure:
    """Chronological anomaly series; gaps are left as gaps."""
    a = anom.sort_values("DATE")
    colors = np.where(a["ANOMALY"] >= 0, WARM, COOL)
    fig = go.Figure(go.Bar(x=a["DATE"], y=a["ANOMALY"], marker_color=colors,
                           hovertemplate="%{x|%b %Y}: %{y:+.2f} °C<extra></extra>", name="Monthly anomaly"))
    _base(fig, "Monthly T2M anomaly time series", f"Baseline {baseline[0]}–{baseline[1]}; {len(a)} months shown; {_missing_note(int(a['ANOMALY'].isna().sum()))}")
    fig.update_yaxes(title="Anomaly (°C)")
    fig.update_xaxes(title="Date")
    return fig


def fig_monthly_slopes(mt: pd.DataFrame) -> go.Figure:
    """Month-wise Sen slope (°C/decade) with 95% CI; filled marker = FDR-supported."""
    colors = [WARM if s > 0 else COOL for s in mt["SEN_PER_DECADE"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mt["MONTH_NAME"].str[:3], y=mt["SEN_PER_DECADE"], mode="markers", name="Sen slope (95% CI)",
        error_y=dict(type="data", symmetric=False, array=mt["CI_HIGH_PER_DECADE"] - mt["SEN_PER_DECADE"],
                     arrayminus=mt["SEN_PER_DECADE"] - mt["CI_LOW_PER_DECADE"], color=MUTED, thickness=1.6, width=5),
        marker=dict(size=13, color=[c if f else "rgba(0,0,0,0)" for c, f in zip(colors, mt["SIG_FDR"])],
                    line=dict(color=colors, width=2)),
        customdata=np.c_[mt["P_RAW"], mt["P_FDR"], mt["CI_LOW_PER_DECADE"], mt["CI_HIGH_PER_DECADE"]],
        hovertemplate="%{x}<br>Sen %{y:+.3f} °C/decade<br>95% CI [%{customdata[2]:+.3f}, %{customdata[3]:+.3f}]<br>p raw %{customdata[0]:.4f}<br>p FDR %{customdata[1]:.4f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dot"))
    _base(fig, "Month-wise Sen slope of T2M, 2001–2025", "Filled marker: BH-FDR adjusted p < 0.05 across 12 tests. Hollow: not supported after correction. Bars: 95% CI.")
    fig.update_yaxes(title="Sen slope (°C per decade)")
    fig.update_xaxes(title="Calendar month")
    return fig


def fig_fingerprint(mt: pd.DataFrame) -> go.Figure:
    """Circular Seasonal Climate Fingerprint: 12 measured Sen slopes, radial, with CIs."""
    theta = [(m - 0.5) * 30 for m in mt["MONTH"]]
    rmax = float(max(abs(mt["CI_LOW_PER_DECADE"]).max(), abs(mt["CI_HIGH_PER_DECADE"]).max()) * 1.1)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[0] * 361, theta=list(range(361)), mode="lines", line=dict(color=MUTED, width=1, dash="dot"), hoverinfo="skip", name="Zero trend"))
    fig.add_trace(go.Barpolar(
        r=mt["SEN_PER_DECADE"], base=0, theta=theta, width=24, name="Sen slope",
        marker=dict(color=[(WARM if s > 0 else COOL) for s in mt["SEN_PER_DECADE"]],
                    opacity=[0.95 if f else 0.38 for f in mt["SIG_FDR"]], line=dict(color=TEXT, width=[1.5 if f else 0 for f in mt["SIG_FDR"]])),
        customdata=np.c_[mt["MONTH_NAME"], mt["CI_LOW_PER_DECADE"], mt["CI_HIGH_PER_DECADE"], mt["P_RAW"], mt["P_FDR"]],
        hovertemplate="<b>%{customdata[0]}</b><br>Sen %{r:+.3f} °C/decade<br>95% CI [%{customdata[1]:+.3f}, %{customdata[2]:+.3f}]<br>p raw %{customdata[3]:.4f} | p FDR %{customdata[4]:.4f}<extra></extra>"))
    rr, tt = [], []
    for th, lo, hi in zip(theta, mt["CI_LOW_PER_DECADE"], mt["CI_HIGH_PER_DECADE"]):
        rr += [lo, hi, None]
        tt += [th, th, None]
    fig.add_trace(go.Scatterpolar(r=rr, theta=tt, mode="lines", line=dict(color=TEXT, width=2), name="95% CI", hoverinfo="skip"))
    fig.update_layout(polar=dict(bgcolor=PANEL, radialaxis=dict(range=[-rmax, rmax], gridcolor=GRID, linecolor=GRID, ticksuffix="", tickfont=dict(color=MUTED), angle=90, title=dict(text="°C/decade")),
                                 angularaxis=dict(direction="clockwise", rotation=90, tickmode="array", tickvals=theta, ticktext=[n[:3] for n in mt["MONTH_NAME"]], gridcolor=GRID, linecolor=GRID)))
    _base(fig, "Sylhet seasonal climate fingerprint: 12 measured monthly trends", "Bar length = Sen slope (°C/decade) measured from the dotted zero ring; outward = warming, inward = cooling. Bright + outlined = FDR-supported; faded = not supported. Line = 95% CI.", height=560)
    return fig


def fig_seasonal_trends(season_trends: pd.DataFrame, annual_row: dict) -> go.Figure:
    """Seasonal and annual Sen slopes with 95% CI."""
    rows = pd.concat([pd.DataFrame([annual_row]), season_trends], ignore_index=True)
    rows = rows.rename(columns={"SERIES": "S"})
    colors = [WARM if s > 0 else COOL for s in rows["SEN_PER_DECADE"]]
    fig = go.Figure(go.Scatter(
        x=rows["S"], y=rows["SEN_PER_DECADE"], mode="markers", marker=dict(size=13, color=colors),
        error_y=dict(type="data", symmetric=False, array=rows["CI_HIGH_PER_DECADE"] - rows["SEN_PER_DECADE"],
                     arrayminus=rows["SEN_PER_DECADE"] - rows["CI_LOW_PER_DECADE"], color=MUTED, thickness=1.6, width=6),
        customdata=np.c_[rows["P_MK"], rows["P_MMK"], rows["N"], rows["CI_LOW_PER_DECADE"], rows["CI_HIGH_PER_DECADE"]],
        hovertemplate="%{x}<br>Sen %{y:+.3f} °C/decade<br>95% CI [%{customdata[3]:+.3f}, %{customdata[4]:+.3f}]<br>p MK %{customdata[0]:.4f} | p Hamed–Rao %{customdata[1]:.4f}<br>n = %{customdata[2]:.0f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dot"))
    _base(fig, "Seasonal and annual trend of T2M", "Sen slope with 95% CI. DJF is labelled by the year of its January (Dec Y−1 + Jan–Feb Y).")
    fig.update_yaxes(title="Sen slope (°C per decade)")
    return fig


def fig_seasonal_climatology(smat: pd.DataFrame, baseline: tuple[int, int]) -> go.Figure:
    """Mean seasonal T2M with ±1 SD."""
    sub = smat[(smat.index >= baseline[0]) & (smat.index <= baseline[1])]
    m, sd = sub.mean(), sub.std(ddof=1)
    fig = go.Figure(go.Bar(x=[cfg.SEASON_LABELS[k] for k in smat.columns], y=m, marker_color=ACCENT,
                           error_y=dict(type="data", array=sd, color=MUTED), hovertemplate="%{x}: %{y:.2f} °C<extra></extra>"))
    _base(fig, "Seasonal climatology of T2M", f"Mean of complete seasons, {baseline[0]}–{baseline[1]}; bars = ±1 SD. Day-weighted means; incomplete seasons excluded.")
    fig.update_yaxes(title="T2M (°C)", range=[0, float(m.max() + sd.max() + 2)])
    return fig


def fig_seasonal_anomalies(smat: pd.DataFrame, baseline: tuple[int, int]) -> go.Figure:
    """Seasonal anomaly lines vs baseline."""
    sub = smat[(smat.index >= baseline[0]) & (smat.index <= baseline[1])]
    anom = smat - sub.mean()
    pal = {"DJF": COOL, "MAM": WARM, "JJAS": "#d9d27a", "ON": ACCENT}
    fig = go.Figure()
    for k in smat.columns:
        fig.add_trace(go.Scatter(x=anom.index, y=anom[k], mode="lines+markers", name=cfg.SEASON_LABELS[k], line=dict(color=pal[k], width=1.8), marker=dict(size=5),
                                 connectgaps=False, hovertemplate="%{x}: %{y:+.2f} °C<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dot"))
    _base(fig, "Seasonal T2M anomalies", f"Relative to each season's mean over {baseline[0]}–{baseline[1]}; DJF 2001 is incomplete and left blank")
    fig.update_yaxes(title="Anomaly (°C)")
    fig.update_xaxes(title="Season year", dtick=2)
    return fig


def fig_boxplot_months(matrix: pd.DataFrame) -> go.Figure:
    """Distribution of each calendar month's 25 values."""
    fig = go.Figure()
    for m in matrix.columns:
        fig.add_trace(go.Box(y=matrix[m], name=cfg.MONTH_NAMES[m - 1][:3], marker_color=WARM, line=dict(color=MUTED), boxpoints="all", jitter=0.4, pointpos=0, marker_size=4, showlegend=False))
    _base(fig, "Year-to-year spread of each calendar month", "Each box = 25 annual values of that month")
    fig.update_yaxes(title="T2M (°C)")
    return fig


def fig_amplitude(amp: pd.DataFrame, amp_trends: pd.DataFrame) -> go.Figure:
    """Annual-cycle amplitude over time (range and first harmonic)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=amp["YEAR"], y=amp["RANGE"], mode="lines+markers", name="Range (hottest − coldest month)", line=dict(color=WARM)))
    fig.add_trace(go.Scatter(x=amp["YEAR"], y=amp["HARMONIC"], mode="lines+markers", name="First-harmonic amplitude", line=dict(color=ACCENT)))
    r = amp_trends.iloc[0]
    _base(fig, "Seasonal-cycle amplitude by year", f"Range trend {r['SEN_PER_DECADE']:+.2f} °C/decade (95% CI {r['CI_LOW_PER_DECADE']:+.2f} to {r['CI_HIGH_PER_DECADE']:+.2f}), p = {r['P_MK']:.2f}")
    fig.update_yaxes(title="Amplitude (°C)")
    fig.update_xaxes(title="Year", dtick=2)
    return fig


def fig_vs_annual(vs: pd.DataFrame, title: str) -> go.Figure:
    """Bootstrap difference of each slope from the annual-mean slope."""
    colors = [WARM if d > 0 else COOL for d in vs["DIFF_FROM_ANNUAL_PER_DECADE"]]
    fig = go.Figure(go.Scatter(
        x=vs["SERIES"].str.replace(r" \(.*\)", "", regex=True).str[:11], y=vs["DIFF_FROM_ANNUAL_PER_DECADE"], mode="markers", marker=dict(size=12, color=colors),
        error_y=dict(type="data", symmetric=False, array=vs["CI_HIGH"] - vs["DIFF_FROM_ANNUAL_PER_DECADE"], arrayminus=vs["DIFF_FROM_ANNUAL_PER_DECADE"] - vs["CI_LOW"], color=MUTED),
        customdata=np.c_[vs["P_BOOT"], vs["P_BOOT_FDR"]], hovertemplate="%{x}<br>%{y:+.3f} °C/decade vs annual<br>p boot %{customdata[0]:.3f} | FDR %{customdata[1]:.3f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=MUTED, dash="dot"))
    _base(fig, title, "Paired-year bootstrap (5,000 resamples). Interval excluding 0 is not enough: see FDR-adjusted p in the table.")
    fig.update_yaxes(title="Slope difference (°C per decade)")
    return fig


def fig_acf(acf: pd.DataFrame, series: str) -> go.Figure:
    """Autocorrelation of Sen-detrended ranks with Hamed–Rao 95% bounds."""
    a = acf[acf["SERIES"] == series]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=a["LAG"], y=a["ACF"], marker_color=[WARM if o else NEUTRAL for o in a["OUTSIDE_BOUNDS"]], name="ACF"))
    fig.add_trace(go.Scatter(x=a["LAG"], y=a["UPPER"], mode="lines", line=dict(color=MUTED, dash="dash"), name="95% bounds"))
    fig.add_trace(go.Scatter(x=a["LAG"], y=a["LOWER"], mode="lines", line=dict(color=MUTED, dash="dash"), showlegend=False))
    _base(fig, f"Autocorrelation diagnostics: {series}", "Ranks of the Sen-detrended annual-resolution series (years)", height=340)
    fig.update_xaxes(title="Lag (years)", dtick=1)
    fig.update_yaxes(title="Autocorrelation")
    return fig


def fig_pvalues(mt: pd.DataFrame) -> go.Figure:
    """Raw vs BH-adjusted p-values per month."""
    x = mt["MONTH_NAME"].str[:3]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=mt["P_RAW"], name="Raw p (Mann–Kendall)", marker_color=NEUTRAL))
    fig.add_trace(go.Bar(x=x, y=mt["P_FDR"], name="BH-adjusted p", marker_color=ACCENT))
    fig.add_hline(y=cfg.ALPHA, line=dict(color=WARM, dash="dash"), annotation_text="0.05", annotation_font_color=WARM)
    _base(fig, "Raw versus FDR-adjusted p-values by month", "Benjamini–Hochberg across 12 month-wise tests", height=380)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="p-value", range=[0, 1])
    return fig


def save_figure(fig: go.Figure, name: str, out_dir: Path = cfg.FIG_DIR) -> list[Path]:
    """Save HTML (always) and PNG (if the kaleido renderer is available)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    html = out_dir / f"{name}.html"
    fig.write_html(str(html), include_plotlyjs="cdn")
    saved.append(html)
    try:
        png = out_dir / f"{name}.png"
        fig.write_image(str(png), scale=2)
        saved.append(png)
    except Exception:  # renderer missing: HTML export still succeeded
        pass
    return saved
