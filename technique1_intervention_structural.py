"""
Technique 1: Intervention Analysis with Structural Decomposition
================================================================
Shock scenario: The shock dies down and the system returns to normal.

Model: Unobserved Components (UCM) with intervention
------------------------------------------------------
The series is decomposed into three interpretable parts:

    y_t = μ_t  +  ψ_t  +  ω·δ^(t-τ)·I(t≥τ)  +  ε_t

    where:
        μ_t   = trend (local linear: random-walk level + slope)
        ψ_t   = stochastic cycle with period λ and damping ρ
        τ     = shock date; I(t≥τ) = indicator for post-shock periods
        ω     = immediate impact of shock (estimated)
        δ     = decay rate (0 < δ < 1); half-life = log(0.5)/log(δ)
        ε_t   ~ N(0, σ²_ε)  irregular / observation noise

Cycle state equations (Harvey 1989):
    [ψ_t  ]   = ρ [cos λ   sin λ ] [ψ_{t-1}  ] + [κ_t  ]
    [ψ*_t ]       [-sin λ  cos λ ] [ψ*_{t-1} ]   [κ*_t ]

    ρ ∈ (0,1): damping factor (1 = undamped, 0 = no cycle)
    λ = 2π / period: angular frequency
    κ_t, κ*_t ~ N(0, σ²_κ): stochastic cycle innovations

Why UCM over SARIMAX for the cattle cycle?
    - SARIMAX with differencing destroys the multi-year cycle structure
    - UCM explicitly separates trend, cycle, and irregular components
    - The cycle period is estimated (or constrained) — not implicit in AR lags
    - Intervention is added as a regression component on the observation eq.
    - All components are extracted via the Kalman smoother

Real-data example: US beef cow inventory (USDA NASS), annual 1970-2023
    - Cattle cycle ≈ 8-12 years (biological + price feedback)
    - Trend: long-run structural decline from ~46M to ~28M head
    - Shocks: 1974 expansion peak liquidation, 2011-12 drought culling

Install: pip install statsmodels pandas matplotlib numpy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.statespace.structural import UnobservedComponents

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: SIMULATED DATA — mechanism demonstration
# ─────────────────────────────────────────────────────────────────────────────

def simulate_ucm_with_intervention(n=80, mu0=100.0, slope0=0.0,
                                    cycle_period=10.0, cycle_amp=6.0,
                                    rho=0.90, sigma_level=0.4,
                                    sigma_cycle=0.8, sigma_obs=1.0,
                                    omega=-12.0, delta=0.75, shock_t=40):
    """
    Simulate a UCM series: trend + stochastic cycle + decaying intervention.

    Parameters
    ----------
    n            : series length
    mu0, slope0  : initial level and slope of trend
    cycle_period : cycle period in time units (e.g. 10 years)
    cycle_amp    : initial cycle amplitude
    rho          : cycle damping (0 < rho <= 1)
    sigma_level  : std dev of level innovations
    sigma_cycle  : std dev of cycle innovations
    sigma_obs    : observation noise std dev
    omega        : shock impact at tau
    delta        : shock decay rate; half-life = log(0.5)/log(delta)
    shock_t      : shock timing (index)
    """
    lam = 2 * np.pi / cycle_period      # angular frequency

    mu    = np.zeros(n)
    beta  = np.zeros(n)
    psi   = np.zeros(n)
    psi_s = np.zeros(n)                 # conjugate cycle state

    mu[0]    = mu0
    beta[0]  = slope0
    psi[0]   = cycle_amp
    psi_s[0] = 0.0

    cos_l, sin_l = np.cos(lam), np.sin(lam)

    for t in range(1, n):
        mu[t]    = mu[t-1] + beta[t-1] + np.random.normal(0, sigma_level)
        beta[t]  = beta[t-1]             # fixed slope

        kappa    = np.random.normal(0, sigma_cycle)
        kappa_s  = np.random.normal(0, sigma_cycle)
        psi[t]   = rho * ( cos_l * psi[t-1] + sin_l * psi_s[t-1]) + kappa
        psi_s[t] = rho * (-sin_l * psi[t-1] + cos_l * psi_s[t-1]) + kappa_s

    # Intervention: decaying pulse from shock_t onward
    intervention = np.zeros(n)
    for t in range(shock_t, n):
        intervention[t] = omega * (delta ** (t - shock_t))

    obs_noise = np.random.normal(0, sigma_obs, n)
    y = mu + psi + intervention + obs_noise

    return y, mu, psi, intervention


print("=" * 65)
print("PART 1: Simulated UCM — trend + cycle + decaying shock")
print("=" * 65)

n, shock_t = 80, 40
y_sim, true_trend, true_cycle, true_interv = simulate_ucm_with_intervention(
    n=n, cycle_period=10, omega=-12.0, delta=0.75, shock_t=shock_t
)

print("\nDecay rate half-life comparison (for presentation):")
for label, delta in [("Fast (delta=0.55)", 0.55), ("Mid  (delta=0.75)", 0.75),
                      ("Slow (delta=0.90)", 0.90)]:
    hl = np.log(0.5) / np.log(delta)
    print(f"  {label}  ->  half-life = {hl:.1f} periods")

# Build intervention regressor with known delta=0.75
delta_fixed = 0.75
interv_regressor = np.zeros(n)
for t in range(shock_t, n):
    interv_regressor[t] = delta_fixed ** (t - shock_t)

model_sim = UnobservedComponents(
    y_sim,
    level='local linear trend',
    cycle=True,
    stochastic_cycle=True,
    damped_cycle=True,
    cycle_period_bounds=(6, 20),
    exog=interv_regressor.reshape(-1, 1)
)
res_sim = model_sim.fit(disp=False, method='powell', maxiter=2000)

states         = res_sim.states.smoothed
## Check by hand the names
print(res_sim.model.state_names)
state_names = res_sim.model.state_names

states_df = pd.DataFrame(states, columns=state_names)#, index=y_sim.index)

smoothed_trend = states_df['level']
smoothed_cycle = states_df['cycle']
res_sim_params = dict(zip(res_sim.param_names, res_sim.params))
omega_hat_sim = params['beta.x1']

freq_hat       = res_sim_params['frequency.cycle']
period_hat_sim = 2 * np.pi / freq_hat

print(f"\nUCM fit (simulated):")
print(f"  Estimated cycle period : {period_hat_sim:.1f} periods  (true: 10)")
print(f"  Estimated shock impact : {omega_hat_sim:.2f}  (true: -12.0)")
print(f"  AIC: {res_sim.aic:.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 2: REAL DATA — US beef cow inventory, annual 1970-2023
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("PART 2: Real data — US beef cow inventory (USDA NASS)")
print("=" * 65)

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
years   = np.array([d.year for d in beef.index])
n_beef  = len(beef)

idx_1974 = np.where(years == 1974)[0].astype(int)[0]
idx_2012 = np.where(years == 2012)[0].astype(int)[0]

def build_shock_regressor(n, shock_idx, delta):
    reg = np.zeros(n)
    for t in range(shock_idx, n):
        reg[t] = delta ** (t - shock_idx)
    return reg

# --- Grid search over delta to minimise AIC ---
print("\nGrid-searching decay rate delta over [0.30, 0.95]...")
delta_grid = np.arange(0.30, 0.96, 0.05)
aic_grid   = []

for delta_try in delta_grid:
    reg1 = build_shock_regressor(n_beef, idx_1974, delta_try)
    reg2 = build_shock_regressor(n_beef, idx_2012, delta_try)
    exog_try = np.column_stack([reg1, reg2])
    try:
        m = UnobservedComponents(
            beef,
            level='local linear trend',
            cycle=True,
            stochastic_cycle=True,
            damped_cycle=True,
            cycle_period_bounds=(5, 15),
            exog=exog_try
        )
        r = m.fit(disp=False, method='powell', maxiter=1500)
        aic_grid.append(r.aic)
    except Exception:
        aic_grid.append(np.inf)

best_idx   = int(np.argmin(aic_grid))
best_delta = delta_grid[best_idx]
best_hl    = np.log(0.5) / np.log(best_delta)
print(f"Best delta = {best_delta:.2f}  (AIC = {aic_grid[best_idx]:.1f})")
print(f"Implied shock half-life = {best_hl:.1f} years")

# --- Fit final model with best delta ---
reg_1974  = build_shock_regressor(n_beef, idx_1974, best_delta)
reg_2012  = build_shock_regressor(n_beef, idx_2012, best_delta)
exog_beef = np.column_stack([reg_1974, reg_2012])

model_beef = UnobservedComponents(
    beef,
    level='local linear trend',
    cycle=True,
    stochastic_cycle=True,
    damped_cycle=True,
    cycle_period_bounds=(5, 15),
    exog=exog_beef
)
res_beef = model_beef.fit(disp=False, method='powell', maxiter=2000)

beef_state_names = res_beef.model.state_names
states_beef  = res_beef.states.smoothed
states_beef_df = pd.DataFrame(states_beef, columns=beef_state_names)#, index=y_sim.index)

trend_beef   = states_beef['level']
cycle_beef   = states_beef['cycle']
beef_sim_params = dict(zip(res_beef.param_names, res_beef.params))

omega_1974   = beef_sim_params['beta.x1']
omega_2012   = beef_sim_params['beta.x2']
interv_beef  = omega_1974 * reg_1974 + omega_2012 * reg_2012
irreg_beef   = beef.values - trend_beef - cycle_beef - interv_beef

freq_beef    = beef_sim_params['frequency.cycle']
period_beef  = 2 * np.pi / freq_beef
fitted_beef  = trend_beef + cycle_beef + interv_beef
cf_beef      = beef.values - omega_2012 * reg_2012   # counterfactual: no 2012

print(f"\nFinal UCM decomposition (beef herd):")
print(f"  Estimated cattle cycle period : {period_beef:.1f} years")
print(f"  Shock impact 1974 (omega_hat) : {omega_1974:.2f} M head")
print(f"  Shock impact 2012 (omega_hat) : {omega_2012:.2f} M head")
print(f"  AIC: {res_beef.aic:.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(15, 14))
fig.suptitle(
    "Technique 1: Intervention Analysis — Shock Decays, System Returns to Normal\n"
    "UCM decomposition: trend  +  multi-year cycle  +  intervention  +  irregular",
    fontsize=13, fontweight='bold', y=0.995
)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.32)

# ── Panel A: Simulated — observed + true vs smoothed components ───────────────
ax1 = fig.add_subplot(gs[0, :])
t = np.arange(n)
ax1.plot(t, y_sim, color='#aaa', lw=1.0, alpha=0.8, label='Observed y_t', zorder=2)
ax1.plot(t, true_trend + true_cycle, color='#2ca02c', lw=1.4, ls='--',
         label='True trend + cycle', zorder=3)
ax1.plot(t, smoothed_trend + smoothed_cycle, color='tomato', lw=1.6,
         label='UCM smoothed: trend + cycle', zorder=4)
ax1.plot(t, true_trend, color='#1f77b4', lw=1.6,
         label='True trend', zorder=3)
ax1.plot(t, smoothed_trend, color='navy', lw=1.3, ls=':',
         label='UCM smoothed trend', zorder=4)
ax1.axvline(shock_t, color='gray', lw=1.2, ls='--', alpha=0.8)
ax1.text(shock_t + 0.5, y_sim.max() - 1.5, 'Shock tau', fontsize=8, color='gray')
ax1.set_title("A  Simulated: true vs UCM-recovered components  "
              "(10-period cycle, delta=0.75, omega=-12)",
              fontsize=10, loc='left')
ax1.set_xlabel("Period")
ax1.set_ylabel("Level")
ax1.legend(fontsize=8, ncol=3)
ax1.set_xlim(0, n - 1)

# ── Panel B: AIC grid search ──────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
finite_mask = np.isfinite(aic_grid)
ax2.plot(delta_grid[finite_mask], np.array(aic_grid)[finite_mask],
          'o-', color='#1f77b4', lw=1.6, ms=5)
ax2.axvline(best_delta, color='tomato', lw=1.5, ls='--',
             label=f'Best delta={best_delta:.2f}\nHL={best_hl:.1f} yr')
ax2.set_title("B  AIC grid search over shock decay rate delta",
              fontsize=10, loc='left')
ax2.set_xlabel("delta (annual decay rate)")
ax2.set_ylabel("AIC")
ax2.legend(fontsize=9)

# ── Panel C: Impulse response functions ──────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
horizons  = np.arange(0, 16)
omega_ref = -3.0
for delta_c, color in [(0.55, '#1f77b4'),
                        (best_delta, 'tomato'),
                        (0.90, '#ff7f0e')]:
    hl_c = np.log(0.5) / np.log(delta_c)
    irf  = omega_ref * delta_c ** horizons
    ax3.plot(horizons, irf, lw=1.8, color=color,
              label=f'delta={delta_c:.2f}  (HL={hl_c:.1f} yr)')
ax3.axhline(0, color='black', lw=0.8, ls=':')
ax3.set_title("C  Impulse response omega * delta^k  (same omega, three decay rates)",
              fontsize=10, loc='left')
ax3.set_xlabel("Years after shock")
ax3.set_ylabel("Response (M head)")
ax3.legend(fontsize=8)

# ── Panel D: Full UCM decomposition — real beef data (stacked sub-panels) ────
# Split this cell into three stacked axes sharing x-axis
gs_d = gs[2, :].subgridspec(3, 1, hspace=0.08)
ax4a = fig.add_subplot(gs_d[0])   # observed + trend + fitted
ax4b = fig.add_subplot(gs_d[1], sharex=ax4a)   # cycle component
ax4c = fig.add_subplot(gs_d[2], sharex=ax4a)   # intervention + irregular

# D-top: observed, trend, fitted
ax4a.plot(years, beef.values, 'k-', lw=1.4, label='Observed', zorder=4)
ax4a.plot(years, trend_beef,  color='#1f77b4', lw=2.0,
           label='Trend mu_t', zorder=3)
ax4a.plot(years, fitted_beef, color='#2ca02c', lw=1.3, ls='--',
           label=f'Trend + cycle + intervention  (period={period_beef:.1f} yr)',
           alpha=0.9, zorder=3)
ax4a.plot(years, cf_beef,     color='#9467bd', lw=1.2, ls=':',
           label='Counterfactual (no 2012 shock)', alpha=0.85)
for yr, ow in [(1974, omega_1974), (2012, omega_2012)]:
    ax4a.axvline(yr, color='tomato', lw=1.0, ls='--', alpha=0.6)
ax4a.set_ylabel("M head")
ax4a.legend(fontsize=7.5, ncol=2)
ax4a.set_title(
    f"D  Real: US beef cow inventory — UCM decomposition  "
    f"(cycle={period_beef:.1f} yr,  delta={best_delta:.2f},  HL={best_hl:.1f} yr)",
    fontsize=10, loc='left'
)
plt.setp(ax4a.get_xticklabels(), visible=False)

# D-mid: cycle component
ax4b.fill_between(years, cycle_beef, 0,
                   where=cycle_beef >= 0, alpha=0.5, color='#2ca02c',
                   label='Cycle > 0 (expansion)')
ax4b.fill_between(years, cycle_beef, 0,
                   where=cycle_beef < 0, alpha=0.5, color='tomato',
                   label='Cycle < 0 (contraction)')
ax4b.plot(years, cycle_beef, color='#2ca02c', lw=1.2)
ax4b.axhline(0, color='black', lw=0.7, ls=':')
ax4b.set_ylabel("Cycle (M head)")
ax4b.legend(fontsize=7.5, ncol=2, loc='upper right')
plt.setp(ax4b.get_xticklabels(), visible=False)

# D-bottom: intervention + irregular
ax4c.bar(years, interv_beef, color='tomato', alpha=0.6, width=0.6,
          label='Intervention effect')
ax4c.plot(years, irreg_beef, color='#888', lw=1.0, ls='-', alpha=0.7,
           label='Irregular epsilon_t')
ax4c.axhline(0, color='black', lw=0.7, ls=':')
for yr, ow in [(1974, omega_1974), (2012, omega_2012)]:
    ax4c.axvline(yr, color='tomato', lw=1.0, ls='--', alpha=0.5)
    ax4c.text(yr + 0.2, min(interv_beef) * 0.85,
               f"{yr}\nomega={ow:.1f}", fontsize=7, color='tomato')
ax4c.set_ylabel("M head")
ax4c.set_xlabel("Year")
ax4c.legend(fontsize=7.5, ncol=2, loc='upper right')

plt.savefig("technique1_intervention_cyclic.png",
            dpi=150, bbox_inches='tight')
print("\nPlot saved: technique1_intervention.png")
plt.show()
