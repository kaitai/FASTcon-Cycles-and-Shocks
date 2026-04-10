"""
Beef Herd UCM Decomposition + Forecast (no shocks)
===================================================
Standalone plot for the forecasting section of the talk.

Fits a clean UCM to the beef cow inventory with NO intervention terms:

    y_t = mu_t  +  psi_t  +  eps_t

    mu_t  = local linear trend  (random-walk level + slope)
    psi_t = stochastic damped cycle  (Harvey 1989)
    eps_t ~ N(0, sigma^2_eps)  irregular

All components are extracted via the Kalman smoother.
A 3-year ahead forecast is generated from the Kalman filter's
one-step-ahead prediction equations, with 80% and 95% confidence bands.

Output: beef_decomposition_forecast.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.statespace.structural import UnobservedComponents

# ── Data ─────────────────────────────────────────────────────────────────────

beef_data = {
    1970: 36.7, 1971: 38.5, 1972: 41.3, 1973: 43.0, 1974: 45.7,
    1975: 45.7, 1976: 43.1, 1977: 40.5, 1978: 38.7, 1979: 37.5,
    1980: 37.1, 1981: 36.6, 1982: 36.8, 1983: 37.4, 1984: 38.5,
    1985: 39.0, 1986: 38.1, 1987: 36.6, 1988: 34.9, 1989: 33.7,
    1990: 33.1, 1991: 32.8, 1992: 32.5, 1993: 33.0, 1994: 33.8,
    1995: 35.2, 1996: 35.4, 1997: 34.8, 1998: 34.2, 1999: 33.7,
    2000: 33.4, 2001: 33.3, 2002: 33.7, 2003: 33.2, 2004: 32.9,
    2005: 32.8, 2006: 32.3, 2007: 31.4, 2008: 31.7, 2009: 31.6,
    2010: 31.3, 2011: 30.9, 2012: 29.3, 2013: 28.9, 2014: 28.2,
    2015: 29.7, 2016: 30.8, 2017: 31.2, 2018: 31.7, 2019: 31.7,
    2020: 31.3, 2021: 30.1, 2022: 28.9, 2023: 28.2,
}

beef = pd.Series(beef_data, name="beef_cows_M")
beef.index = pd.to_datetime([f"{y}-01-01" for y in beef.index])
beef.index.freq = "YS"
years = np.array([d.year for d in beef.index])

# ── Fit shock-free UCM ───────────────────────────────────────────────────────

print("Fitting shock-free UCM (trend + stochastic cycle)...")
model = UnobservedComponents(
    beef,
    level='local linear trend',
    cycle=True,
    stochastic_cycle=True,
    damped_cycle=True,
    cycle_period_bounds=(5, 15),        # cattle cycle: 5-15 year range
)
res = model.fit(disp=False, method='powell', maxiter=2000)

# ── In-sample decomposition via Kalman smoother ──────────────────────────────



beef_state_names = res.model.state_names
states_beef  = res.states.smoothed
states_beef_df = pd.DataFrame(states_beef, columns=beef_state_names)#, index=y_sim.index)

trend_beef   = states_beef['level']   # smoothed trend mu_t
cycle_beef   = states_beef['cycle']   # smoothed cycle psi_t

fitted  = trend_beef + cycle_beef               # model fit (no irregular)
irreg   = beef.values - fitted        # irregular = observed - (trend + cycle)

beef_sim_params = dict(zip(res.param_names, res.params))

freq_beef    = beef_sim_params['frequency.cycle']
period_beef  = 2 * np.pi / freq_beef
fitted_beef  = trend_beef + cycle_beef

rho_est    = beef_sim_params['damping.cycle']

print(f"  Estimated cycle period : {period_beef:.1f} years")
print(f"  Cycle damping rho      : {rho_est:.3f}")
print(f"  AIC: {res.aic:.1f}")
print(f"  Log-likelihood: {res.llf:.1f}")

# Print variance decomposition
var_obs   = beef_sim_params.get('sigma2.irregular', 0.0)
var_level = beef_sim_params.get('sigma2.level', 0.0)
var_slope = beef_sim_params.get('sigma2.trend', 0.0)
var_cycle = beef_sim_params.get('sigma2.cycle', 0.0)
print(f"\n  Variance components:")
print(f"    sigma2 irregular : {var_obs:.4f}")
print(f"    sigma2 level     : {var_level:.4f}")
print(f"    sigma2 slope     : {var_slope:.4f}")
print(f"    sigma2 cycle     : {var_cycle:.4f}")

# ── 3-year forecast ──────────────────────────────────────────────────────────
# get_forecast() propagates the Kalman filter forward, accumulating
# state uncertainty + observation variance into the prediction intervals.

n_fcast    = 3
fcast_obj  = res.get_forecast(steps=n_fcast)
fcast_mean = fcast_obj.predicted_mean
fcast_ci80 = fcast_obj.conf_int(alpha=0.20)   # 80% band
fcast_ci95 = fcast_obj.conf_int(alpha=0.05)   # 95% band

fcast_years = np.array([2024, 2025, 2026])

# Also extract the forecast-horizon trend and cycle from the state forecasts
# (these come from the filtered — not smoothed — states at the end of sample,
#  then propagated forward by the state transition matrix)
fcast_states = fcast_obj.predicted_mean   # observation-level mean

print(f"\n  3-year forecast (mean):")
for yr, val, lo95, hi95 in zip(
        fcast_years,
        fcast_mean.values,
        fcast_ci95.iloc[:, 0].values,
        fcast_ci95.iloc[:, 1].values):
    print(f"    {yr}: {val:.2f} M head  [95% CI: {lo95:.2f} – {hi95:.2f}]")

# ── Plot ─────────────────────────────────────────────────────────────────────

# Colour palette
C_OBS    = '#222222'
C_TREND  = '#1f77b4'
C_CYCLE_POS = '#2ca02c'
C_CYCLE_NEG = '#d62728'
C_IRREG  = '#888888'
C_FCAST  = '#ff7f0e'
C_CI80   = '#ffc47a'
C_CI95   = '#ffe5c0'

fig = plt.figure(figsize=(14, 6))
fig.suptitle(
    "US Beef Cow Inventory — UCM Decomposition & 3-Year Forecast\n"
    f"Local linear trend  +  stochastic cycle (period ≈ {period_beef:.1f} yr,  "
    f"damping ρ ≈ {rho_est:.2f})  +  irregular   [no shock terms]",
    fontsize=12, fontweight='bold', y=0.995
)

gs = gridspec.GridSpec(
    4, 1, figure=fig,
    hspace=0.10,
    height_ratios=[2.8, 1.4, 1.4, 1.1]
)

all_years = np.concatenate([years, fcast_years])

# shared x limits with a little padding
xlim = (years[0] - 1, fcast_years[-1] + 1)

# ── Panel 1: Observed + fitted + forecast ─────────────────────────────────────
ax1 = fig.add_subplot(gs[0])

# 95% CI shading (forecast only)
ax1.fill_between(fcast_years,
                  fcast_ci95.iloc[:, 0], fcast_ci95.iloc[:, 1],
                  color=C_CI95, label='95% forecast band')
# 80% CI shading
ax1.fill_between(fcast_years,
                  fcast_ci80.iloc[:, 0], fcast_ci80.iloc[:, 1],
                  color=C_CI80, label='80% forecast band')
# Observed
ax1.plot(years, beef.values, color=C_OBS, lw=1.8,
          label='Observed (USDA NASS)', zorder=5)
# In-sample fit
ax1.plot(years, fitted, color=C_TREND, lw=1.4, ls='--', alpha=0.85,
          label='UCM in-sample fit (trend + cycle)', zorder=4)
# Forecast mean — connected from last observed point
ax1.plot(
    np.concatenate([[years[-1]], fcast_years]),
    np.concatenate([[beef.values[-1]], fcast_mean.values]),
    color=C_FCAST, lw=2.0, ls='-', marker='o', ms=5,
    label='Forecast mean (2024–2026)', zorder=6
)
# Vertical separator
ax1.axvline(2023.5, color='gray', lw=1.0, ls=':', alpha=0.7)
ax1.text(2023.7, beef.max() - 0.5, 'Forecast →', fontsize=8,
          color='gray', va='top')

ax1.set_ylabel("Million head (Jan 1)", fontsize=10)
ax1.set_xlim(xlim)
ax1.legend(fontsize=8, ncol=2, loc='upper right')
ax1.set_title("Observed, in-sample fit, and 3-year forecast",
               fontsize=10, loc='left', pad=4)
plt.setp(ax1.get_xticklabels(), visible=False)

# ── Panel 2: Trend component ──────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1], sharex=ax1)

ax2.plot(years, trend_beef, color=C_TREND, lw=2.0, label='Trend $\\mu_t$')

# Extend trend into forecast: use the filtered state at T, propagated forward
# by the local-linear-trend transition (mu_{T+h} = mu_T + h * beta_T)
# We approximate this from the difference in smoothed trend at the end.
last_trend = trend_beef[-1]
last_slope = trend_beef[-1] - trend_beef[-2]       # approximate slope from last two obs
trend_fcast = last_trend + last_slope * np.arange(1, n_fcast + 1)

ax2.plot(
    np.concatenate([[years[-1]], fcast_years]),
    np.concatenate([[last_trend], trend_fcast]),
    color=C_TREND, lw=1.6, ls='--', alpha=0.7,
    label='Trend extrapolation'
)
ax2.axvline(2023.5, color='gray', lw=1.0, ls=':', alpha=0.7)
ax2.set_ylabel("M head", fontsize=10)
ax2.legend(fontsize=8, loc='upper right')
ax2.set_title("Trend component $\\mu_t$  (local linear trend via Kalman smoother)",
               fontsize=10, loc='left', pad=4)
plt.setp(ax2.get_xticklabels(), visible=False)

# ── Panel 3: Cycle component ──────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2], sharex=ax1)

ax3.fill_between(years, cycle_beef, 0,
                  where=cycle_beef >= 0,
                  color=C_CYCLE_POS, alpha=0.45, label='Expansion phase')
ax3.fill_between(years, cycle_beef, 0,
                  where=cycle_beef < 0,
                  color=C_CYCLE_NEG, alpha=0.45, label='Contraction phase')
ax3.plot(years, cycle_beef, color='#333', lw=1.2, alpha=0.8)
ax3.axhline(0, color='black', lw=0.8, ls=':')

# Propagate cycle into forecast using Harvey rotation
# psi_{t+1} = rho * [cos(lam)*psi_t + sin(lam)*psi*_t]
# We need both cycle states; approximate psi* ≈ 0 at end of sample
# (a conservative projection — the cycle mean-reverts toward zero)
lam = freq_beef
cos_l, sin_l = np.cos(lam), np.sin(lam)
psi_now  = cycle_beef[-1]
psi_s_now = 0.0       # conjugate state not stored separately; set to zero
rho_use   = rho_est if not np.isnan(rho_est) else 0.85

cycle_fcast = []
ps, pss = psi_now, psi_s_now
for _ in range(n_fcast):
    ps_new  = rho_use * ( cos_l * ps + sin_l * pss)
    pss_new = rho_use * (-sin_l * ps + cos_l * pss)
    ps, pss = ps_new, pss_new
    cycle_fcast.append(ps)

cycle_fcast = np.array(cycle_fcast)

ax3.plot(
    np.concatenate([[years[-1]], fcast_years]),
    np.concatenate([[cycle_beef[-1]], cycle_fcast]),
    color='#333', lw=1.6, ls='--', alpha=0.75,
    label='Cycle projection (damped rotation)'
)
ax3.fill_between(
    np.concatenate([[years[-1]], fcast_years]),
    np.concatenate([[cycle_beef[-1]], cycle_fcast]),
    0,
    alpha=0.15, color='#999',
    label='_nolegend_'
)
ax3.axvline(2023.5, color='gray', lw=1.0, ls=':', alpha=0.7)
ax3.set_ylabel("M head deviation", fontsize=10)
ax3.legend(fontsize=8, loc='upper right', ncol=2)
ax3.set_title(
    f"Cycle component $\\psi_t$  (period ≈ {period_beef:.1f} yr,  "
    f"$\\rho$ ≈ {rho_est:.2f})  —  expansion (green) / contraction (red)",
    fontsize=10, loc='left', pad=4
)
plt.setp(ax3.get_xticklabels(), visible=False)

# ── Panel 4: Irregular component ─────────────────────────────────────────────
ax4 = fig.add_subplot(gs[3], sharex=ax1)

ax4.bar(years, irreg,
         color=[C_CYCLE_POS if v >= 0 else C_CYCLE_NEG for v in irreg],
         alpha=0.6, width=0.7)
ax4.axhline(0, color='black', lw=0.8, ls=':')

# 2-sigma band from estimated observation variance
sigma_irreg = np.sqrt(var_obs) if var_obs > 0 else irreg.std()
ax4.axhline( 2 * sigma_irreg, color='gray', lw=0.8, ls='--', alpha=0.6,
              label=f'±2σ  (σ={sigma_irreg:.2f})')
ax4.axhline(-2 * sigma_irreg, color='gray', lw=0.8, ls='--', alpha=0.6,
              label='_nolegend_')

# Annotate notable outliers (|irregular| > 1.5 sigma)
for yr, val in zip(years, irreg):
    if abs(val) > 1.5 * sigma_irreg:
        ax4.text(yr, val + (0.08 if val > 0 else -0.15),
                  str(yr), fontsize=7, ha='center',
                  color=C_CYCLE_POS if val > 0 else C_CYCLE_NEG)

ax4.axvline(2023.5, color='gray', lw=1.0, ls=':', alpha=0.7)
ax4.set_ylabel("M head", fontsize=10)
ax4.set_xlabel("Year", fontsize=10)
ax4.legend(fontsize=8, loc='upper right')
ax4.set_title(
    "Irregular component $\\varepsilon_t$  (residual after trend + cycle;  "
    "spikes = unmodeled shocks)",
    fontsize=10, loc='left', pad=4
)
ax4.set_ylim(irreg.min(), irreg.max())

# Shared x-axis ticks
ax4.set_xlim(xlim)
ax4.set_xticks(range(1970, 2027, 5))
ax4.tick_params(axis='x', rotation=0)

plt.savefig("beef_decomposition_forecast_wide.png",
            dpi=150, bbox_inches='tight')
print("\nPlot saved: beef_decomposition_forecast.png")
plt.show()
