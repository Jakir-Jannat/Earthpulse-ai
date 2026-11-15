# EARTHPULSE AI - Scientific Findings Report

*Beyond Average Temperature: Discovering Sylhet's Hidden Seasonal Climate Shifts*

Generated 2026-11-15 17:17 UTC from `POWER_Point_Monthly_20010101_20251231_024d89N_091d87E_LST.csv` (SHA-256 `79f021a95bec99a1...`). All numbers below are computed by the code in this repository from the uploaded file.

## 0. Data and scope
- Source: NASA POWER, NASA/POWER Source Native Resolution Monthly and Annual, parameter **T2M** (MERRA-2 Temperature at 2 Meters, C).
- Location: a **single grid-cell estimate** at 24.8949°N, 91.8687°E; source cell 0.5° x 0.625° (lat x lon), model elevation 103.47 m. It does not describe the whole Sylhet district.
- Period: 01/01/2001 to 12/31/2025 (LST); 300 of 300 monthly values valid; audit status **PASS**.
- Values are reanalysis (MERRA-2) output, not station observations.

## 1. Is the selected Sylhet location warming over 2001-2025?
Annual mean T2M (computed from the 12 monthly values; ANN column not used) shows a positive point estimate (+0.08 °C/decade, 95% CI [-0.06, +0.22]) that is NOT statistically distinguishable from zero (p = 0.216). Mann-Kendall S = 54, Kendall tau = 0.18; Hamed-Rao p = 0.216 (correction factor 1.00). The OLS companion gives +0.09 °C/decade (95% CI [-0.04, +0.22], p = 0.180, R² = 0.08). Using an unweighted annual mean changes the Sen slope to +0.08 °C/decade (p = 0.252).

## 2. Is warming uniform throughout the year?
Month-wise Sen slopes range from -0.23 °C/decade (March) to +0.34 °C/decade (September). A spread in point estimates is expected from sampling noise alone with 25 values per month, so it is tested explicitly:
- Global bootstrap test that all 12 monthly slopes are equal: p = 0.194 (precision-weighted; unweighted p = 0.253).
- Pairwise month differences (66 pairs, paired-year bootstrap, BH-adjusted): **0** supported; smallest adjusted p = 0.422.
- Month slope minus annual slope (12 comparisons): **0** supported after BH; smallest adjusted p = 0.479.

Conclusion: the data do **not** provide statistically supported evidence that monthly trend rates differ from one another or from the annual trend. Uniform warming cannot be rejected, and equally the data are too short to establish that it is uniform.

## 3. Which months show the strongest estimated warming / cooling? (descriptive ranking)
| Rank | Month | Sen slope (°C/decade) | 95% CI | Raw p | FDR p |
|---:|---|---:|---|---:|---:|
| 1 | September | +0.338 | [+0.11, +0.59] | 0.002 | 0.023 |
| 2 | November | +0.320 | [-0.04, +0.65] | 0.059 | 0.247 |
| 3 | January | +0.283 | [-0.25, +0.71] | 0.243 | 0.550 |
| 4 | October | +0.265 | [-0.02, +0.59] | 0.062 | 0.247 |
| 5 | July | +0.197 | [-0.04, +0.40] | 0.102 | 0.305 |
| 6 | December | +0.141 | [-0.18, +0.57] | 0.362 | 0.550 |
| 7 | June | +0.113 | [-0.10, +0.29] | 0.412 | 0.550 |
| 8 | April | -0.048 | [-0.42, +0.34] | 0.852 | 0.852 |
| 9 | August | -0.093 | [-0.36, +0.12] | 0.400 | 0.550 |
| 10 | May | -0.120 | [-0.37, +0.14] | 0.400 | 0.550 |
| 11 | February | -0.125 | [-0.76, +0.33] | 0.607 | 0.729 |
| 12 | March | -0.228 | [-0.70, +0.37] | 0.709 | 0.773 |

Ranking by point estimate is descriptive only; it is not evidence that one month changes faster than another (see Section 2).

## 4. Which monthly trends remain supported after FDR correction?
- **September**: +0.34 °C/decade, 95% CI [+0.11, +0.59], raw p = 0.002, BH-adjusted p = 0.023.

1 month(s) have raw p < 0.05; 1 remain supported after correction. In a joint BH family of all 16 monthly + seasonal tests, adjusted p for September = 0.030.
A single month passing among twelve is a screening result: it merits follow-up with independent data, not a firm conclusion.

## 5. Are seasonal trends different?
| Season | n | Sen (°C/decade) | 95% CI | p (MK) | p (Hamed-Rao) | BH p (4 seasons) | Lag-1 ACF |
|---|---:|---:|---|---:|---:|---:|---:|
| Winter (DJF) | 24 | +0.067 | [-0.28, +0.39] | 0.602 | 0.602 | 0.602 | +0.21 |
| Pre-monsoon (MAM) | 25 | -0.095 | [-0.34, +0.15] | 0.315 | 0.315 | 0.420 | +0.25 |
| Monsoon (JJAS) | 25 | +0.130 | [-0.03, +0.29] | 0.107 | 0.107 | 0.214 | -0.11 |
| Post-monsoon (ON) | 25 | +0.316 | [+0.11, +0.53] | 0.010 | 0.010 | 0.038 | +0.04 |

Seasons with BH-adjusted p < 0.05: Post-monsoon (ON) (joint 16-test BH-adjusted p: Post-monsoon (ON) = 0.076). Global bootstrap test that the four seasonal slopes are equal: p = 0.094. Pairwise seasonal differences supported after BH: 0 of 6 (closest: Pre-monsoon (MAM) vs Post-monsoon (ON), adjusted p = 0.0504). Seasonal and monthly tests use overlapping data, so they are not independent confirmations of each other. DJF is labelled by its January year (Dec Y-1 + Jan-Feb Y), leaving 24 complete winters.

## 6. Is seasonal amplitude changing?
- Annual range (hottest - coldest month): Sen -0.23 °C/decade, 95% CI [-0.62, +0.38], p = 0.350 (not statistically supported).
- First-harmonic amplitude: Sen -0.02 °C/decade, 95% CI [-0.22, +0.16], p = 0.797 (not statistically supported).

## 7. Secondary analyses
- Pettitt change-point test on the annual mean and four seasons: 0 series show a change point supported by raw p, trimmed-ends sensitivity and BH correction. Smallest raw p = 0.0498 (Pre-monsoon (MAM)); its trimmed-ends p = 0.115. No regime shift is claimed.
- Early (2001-2012) vs late (2013-2025) means (midpoint split fixed a priori): 0 of 17 comparisons significant after BH; smallest adjusted p = 0.097.

## 8. Uncertainties and limitations
- One MERRA-2 grid cell (0.5° x 0.625°, roughly 56 x 63 km at this latitude) is not Sylhet district and is not a weather-station record; reanalysis temperatures may carry model biases and inhomogeneities.
- Twenty-five annual values per series give low power: real differences smaller than the confidence intervals above cannot be detected.
- A 25-year window is short relative to multi-decadal variability (e.g. ENSO, monsoon variability); trends here are not claimed to be long-term climate trends.
- Multiple testing: 12 monthly tests were BH-corrected as one family, 4 seasonal tests as another; the joint 16-test sensitivity is reported in Section 4.
- Autocorrelation: lag diagnostics and the Hamed-Rao modified MK (inflate-only variant) are reported; with n = 25 the lag estimates are themselves uncertain.
- Bootstrap comparisons resample years as exchangeable units and may be optimistic if years are serially dependent.
- Statistical association is not attribution: nothing here establishes causes, and no prediction model is implemented or validated.

## 9. What can and cannot be claimed
**Justified by this analysis**
- No statistically detectable linear warming trend in annual mean T2M at this grid cell over 2001-2025 (Sen +0.08 °C/decade, CI [-0.06, +0.22]).
- A statistically supported warming trend in September after FDR correction (screening-level).
- A statistically supported warming trend in Post-monsoon (ON) within the four-season family (+0.32 °C/decade, CI [+0.11, +0.53]); joint 16-test adjusted p = 0.076 (does not meet 0.05 when pooled with monthly tests).
- Point estimates differ between months, but that difference is not statistically established.

**Not justified**
- Claims about Sylhet district as a whole.
- Claims that any month or season warms *faster* than another unless a supported difference is listed above.
- Causal attribution to greenhouse gases, urbanisation, land use or monsoon dynamics.
- Predictions or projections.
- A climate regime shift (no change point passed all checks).
- Treating the monthly ranking or the fingerprint shape as a validated climate index.
