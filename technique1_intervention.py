"""
Full disclosures: I (Kaisa) worked with Claude Sonnet 4.6 to generate this code. I selected the case study and techniques first, then asked Claude to generate code that would include illustrations of my main points for this audience.


Technique 1: Intervention Analysis (Box-Tiao Transfer Function Models)
=======================================================================
Shock scenario: The shock dies down and the system returns to normal.

Key equation:
    y_t = μ + [ω / (1 - δB)] * I_t + N_t

    where:
        I_t   = pulse indicator (1 at shock date, 0 otherwise)
        ω     = immediate impact of shock
        δ     = decay rate (0 < δ < 1); controls how fast response fades
        B     = backshift operator: B*x_t = x_{t-1}
        N_t   = SARIMA noise process (the underlying cycle)

    The impulse response at lag k is:  ω * δ^k
    Half-life of shock:                k = log(0.5) / log(δ)

Real-data example: US beef cow inventory (USDA NASS)
    - Drought shocks (e.g. 2011-2012 Texas/Oklahoma) cull breeding stock
    - Herd rebuilds over a 5-7 year cattle cycle
    - Intervention captures the shock; SARIMA captures the underlying cycle

Install: pip install statsmodels pandas matplotlib numpy pandas-datareader
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: SIMULATED DATA — mechanism demonstration
# ─────────────────────────────────────────────────────────────────────────────

def simulate_intervention(n=120, mu=100.0, omega=-15.0, delta=0.75,
                           shock_t=60, ar_coef=0.6, sigma=2.0):
    """
    Simulate a time series with a decaying pulse intervention.

    The series follows an AR(1) cycle around mu, with a shock at shock_t
    that decays geometrically at rate delta.

    Parameters
    ----------
    n       : length of series
    mu      : long-run mean (equilibrium)
    omega   : immediate shock size (negative = downward shock)
    delta   : decay rate; higher = slower return to baseline
    shock_t : period of shock
    ar_coef : autoregressive coefficient of underlying cycle
    sigma   : noise std dev
    """
    # Build the intervention effect:  omega * delta^k for k periods after shock
    intervention_effect = np.zeros(n)
    for t in range(shock_t, n):
        k = t - shock_t
        intervention_effect[t] = omega * (delta ** k)

    # Build the AR(1) noise process
    noise = np.zeros(n)
    eps = np.random.normal(0, sigma, n)
    for t in range(1, n):
        noise[t] = ar_coef * noise[t - 1] + eps[t]

    y = mu + noise + intervention_effect
    return y, intervention_effect


print("=" * 65)
print("PART 1: Simulated intervention — decaying pulse shock")
print("=" * 65)

# --- Simulate three scenarios with different decay rates ---
n, shock_t = 120, 60
y_fast, eff_fast = simulate_intervention(n=n, delta=0.60, shock_t=shock_t)
y_mid,  eff_mid  = simulate_intervention(n=n, delta=0.80, shock_t=shock_t)
y_slow, eff_slow = simulate_intervention(n=n, delta=0.92, shock_t=shock_t)

# Half-lives
for label, delta in [("Fast (δ=0.60)", 0.60), ("Mid  (δ=0.80)", 0.80),
                      ("Slow (δ=0.92)", 0.92)]:
    hl = np.log(0.5) / np.log(delta)
    print(f"  {label}  →  half-life = {hl:.1f} periods")

# --- Fit SARIMAX with intervention dummy on the mid-decay series ---
pulse = np.zeros(n)
pulse[shock_t] = 1.0                    # pulse indicator I_t

# SARIMAX(1,0,0) with exogenous pulse — statsmodels handles the transfer fn
# via the ARMA structure on the disturbance + the exogenous regressor
model_sim = SARIMAX(
    y_mid,
    exog=pd.DataFrame({'pulse': pulse}),
    order=(1, 0, 0),                    # AR(1) noise
    trend='c'
)
res_sim = model_sim.fit(disp=False)

omega_hat = res_sim.params['pulse']     # immediate impact
ar_hat    = res_sim.params['ar.L1']     # estimated AR decay (≈ δ in simple case)
print(f"\nEstimated impact ω̂  = {omega_hat:.2f}  (true: -15.0)")
print(f"Estimated AR coef   = {ar_hat:.2f}  (true decay δ: 0.80)")
print(f"AIC: {res_sim.aic:.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 2: REAL DATA — US beef cow inventory
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("PART 2: Real data — US beef cow inventory (USDA NASS)")
print("=" * 65)
print("Note: Using embedded USDA annual data (Jan inventory, millions head)")
print("Source: USDA NASS Cattle report, historical series")

# USDA NASS beef cow inventory, Jan 1 each year (millions head), 1970-2023
# Source: https://usda.library.cornell.edu/concern/publications/h702q636h
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
beef.index.freq = "YS"                  # annual start frequency

# Two documented drought shock years
shock_years = [1974, 2012]              # 1974 expansion peak; 2012 TX drought

# Build intervention dummies (step functions: shock hits and fades via AR)
# For annual data, a pulse at the shock year is simplest; the AR structure
# in the model captures subsequent propagation.
exog_real = pd.DataFrame(index=beef.index)
for yr in shock_years:
    col = np.zeros(len(beef))
    idx = beef.index.get_loc(pd.Timestamp(f"{yr}-01-01"))
    col[idx] = 20.0
    exog_real[f"pulse_{yr}"] = col

model_real = SARIMAX(
    beef,
    exog=exog_real,
    order=(1, 1, 0),                    # AR(1) on first-differences (cattle cycle)
    trend='n'
)
res_real = model_real.fit(disp=False)

print(f"\nImpact of 1974 shock (ω̂): {res_real.params['pulse_1974']:.2f} M head")
print(f"Impact of 2012 shock (ω̂): {res_real.params['pulse_2012']:.2f} M head")
print(f"AR(1) coefficient: {res_real.params['ar.L1']:.3f}")
print(f"AIC: {res_real.aic:.1f}")

# In-sample fit + counterfactual (no 2012 shock)
fitted = res_real.fittedvalues

exog_counter = exog_real.copy()
exog_counter["pulse_2012"] = 0.0       # remove the 2012 shock
counter_forecast = res_real.predict(exog=exog_counter)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 10))
fig.suptitle("Technique 1: Intervention Analysis — Shock Decays, System Returns",
             fontsize=13, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

# Panel A: Three decay rates
ax1 = fig.add_subplot(gs[0, :])
t = np.arange(n)
ax1.axvline(shock_t, color='gray', lw=1.2, ls='--', alpha=0.7, label='Shock')
ax1.axhline(100, color='black', lw=0.8, ls=':', alpha=0.5, label='Equilibrium μ=100')
ax1.plot(t, y_fast, lw=1.4, color='#1f77b4', alpha=0.85, label='δ=0.60 (fast, HL≈1.4)')
ax1.plot(t, y_mid,  lw=1.4, color='#ff7f0e', alpha=0.85, label='δ=0.80 (mid,  HL≈3.1)')
ax1.plot(t, y_slow, lw=1.4, color='#d62728', alpha=0.85, label='δ=0.92 (slow, HL≈8.3)')
ax1.set_title("A  Simulated: same shock (ω=−15), three decay rates",
              fontsize=10, loc='left')
ax1.set_xlabel("Period")
ax1.set_ylabel("Index level")
ax1.legend(fontsize=8, ncol=5)
ax1.set_xlim(0, n - 1)

# Panel B: Impulse response functions (deterministic part only)
ax2 = fig.add_subplot(gs[1, 0])
horizons = np.arange(0, 25)
for delta, label, color in [(0.60, 'δ=0.60', '#1f77b4'),
                              (0.80, 'δ=0.80', '#ff7f0e'),
                              (0.92, 'δ=0.92', '#d62728')]:
    irf = -15 * delta ** horizons       # ω * δ^k
    ax2.plot(horizons, irf, lw=1.8, color=color, label=label)
ax2.axhline(0, color='black', lw=0.8, ls=':')
ax2.set_title("B  Impulse response: ω · δᵏ", fontsize=10, loc='left')
ax2.set_xlabel("Horizon k (periods after shock)")
ax2.set_ylabel("Response")
ax2.legend(fontsize=8)

# Panel C: Real beef cow data + fitted + counterfactual
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(beef.index.year, beef.values, 'k-', lw=1.6, label='Observed', zorder=3)
ax3.plot(beef.index.year, fitted.values, '--', color='#2ca02c', lw=1.3,
         label='Model fit', alpha=0.9)
ax3.plot(beef.index.year, counter_forecast.values, ':', color='red', lw=1.4,
         label='Counterfactual\n(no 2012 shock)', alpha=0.85)
for yr in shock_years:
    ax3.axvline(yr, color='tomato', lw=1.2, ls='--', alpha=0.7)
    ax3.text(yr + 0.3, beef.min() + 0.3, str(yr), fontsize=7, color='tomato')
ax3.set_title("C  Real: US beef cow inventory (USDA NASS)", fontsize=10, loc='left')
ax3.set_xlabel("Year")
ax3.set_ylabel("Million head (Jan 1)")
ax3.set_ylim(20,50)
ax3.legend(fontsize=8)

plt.savefig("technique1_intervention.png",
            dpi=150, bbox_inches='tight')
print("\nPlot saved: technique1_intervention.png")
plt.show()
