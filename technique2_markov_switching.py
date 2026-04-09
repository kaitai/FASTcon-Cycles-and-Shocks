"""
Technique 2: Markov-Switching Models (Hamilton 1989)
=====================================================
Shock scenario: The shock decays but there is a new baseline.

Key equation (2-state Markov-Switching AR(p)):
    y_t = μ_{s_t} + Σ φ_k (y_{t-k} - μ_{s_{t-k}}) + σ_{s_t} · ε_t

    where s_t ∈ {0, 1} is the latent regime, governed by:
        P(s_t = j | s_{t-1} = i) = p_{ij}

    Transition matrix:
        P = [[p_{00}  1-p_{00}],
             [1-p_{11} p_{11} ]]

    Expected duration in regime i:  E[D_i] = 1 / (1 - p_{ii})

    Regime means μ_0, μ_1 — these are the two "baselines" the system can occupy.
    A shock is identified as a transition: a large, persistent move that shifts
    the system from one attractor to another.

Real-data example: Baltic Dry Index (BDI), monthly
    - Pre-2008: high-demand regime with elevated freight rates
    - Post-2008: structural oversupply regime; persistent lower baseline
    - A recession shock triggered the regime shift — BDI did not return to prior mean

Install: pip install statsmodels pandas matplotlib numpy pandas-datareader
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

np.random.seed(99)

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: SIMULATED DATA — mechanism demonstration
# ─────────────────────────────────────────────────────────────────────────────

def simulate_markov_switching(n=200, mu0=100.0, mu1=60.0,
                               sigma0=3.0, sigma1=6.0,
                               p00=0.95, p11=0.92,
                               ar_coef=0.5):
    """
    Simulate a Markov-switching AR(1) series.

    Regime 0: high baseline (mu0), low volatility (sigma0)
    Regime 1: low baseline (mu1), high volatility (sigma1)

    p00 = P(stay in regime 0 | in regime 0)  — persistence of high regime
    p11 = P(stay in regime 1 | in regime 1)  — persistence of low regime

    I'm having problems with statsmodels here as it seems to not have P(1->1)
    Instead it's P(1->0)
    so I will just change this and see if it works
    """
    states = np.zeros(n, dtype=int)
    y = np.zeros(n)

    # Initialize in regime 0
    states[0] = 0
    y[0] = mu0

    mus = [mu0, mu1]
    sigmas = [sigma0, sigma1]
    p_stay = [p00, p11]

    for t in range(1, n):
        s_prev = states[t - 1]
        # Transition
        if np.random.rand() < p_stay[s_prev]:
            states[t] = s_prev
        else:
            states[t] = 1 - s_prev

        s = states[t]
        # AR(1) demeaned dynamics
        y[t] = mus[s] + ar_coef * (y[t-1] - mus[states[t-1]]) + \
               np.random.normal(0, sigmas[s])

    return y, states


print("=" * 65)
print("PART 1: Simulated Markov-switching AR(1)")
print("=" * 65)

y_sim, states_sim = simulate_markov_switching(
    n=400, mu0=100.0, mu1=60.0,
    p00=0.95, p11=0.92
)

# Expected durations from transition probabilities
for regime, p_stay, mu in [(0, 0.95, 100.0), (1, 0.92, 60.0)]:
    dur = 1.0 / (1.0 - p_stay)
    print(f"  Regime {regime} (μ={mu:.0f})  p_stay={p_stay:.2f}  "
          f"→  expected duration = {dur:.1f} periods")

# Fit Markov-switching model
# MarkovAutoregression: switching_variance=True allows each regime its own σ
model_sim = MarkovAutoregression(
    y_sim,
    k_regimes=2,
    order=1,
    switching_ar=False,         # AR coefficient shared across regimes (parsimonious)
    switching_variance=True     # Each regime has its own variance
)
res_sim = model_sim.fit(disp=False, search_reps=20)
param_names = model_sim.param_names  # list of strings

def get_param(name_fragment):
    """Fetch a parameter by substring match against param_names."""
    matches = [(i, n) for i, n in enumerate(param_names) if name_fragment in n]
    if not matches:
        raise KeyError(f"No parameter matching '{name_fragment}'. "
                       f"Available: {param_names}")
    if len(matches) > 1:
        raise KeyError(f"Ambiguous match for '{name_fragment}': {matches}")
    return res_sim.params[matches[0][0]]

# Replace all res_sim.params['const[0]'] style lookups:
m0 = get_param('const[0]')
m1 = get_param('const[1]')
p00 = get_param('p[0->0]')
p10 = get_param('p[1->0]')
#p11 = get_param('p[1->1]')
p11 = 1-p10

print(f"\nEstimated regime means:  {m0:.2f}, {m1:.2f}  (true: 100.0, 60.0)")
print(f"Transition prob p̂_00 = {p00:.3f}  (true: 0.95)")
print(f"Transition prob p̂_11 = {p11:.3f}  (true: 0.92)")
print(f"Log-likelihood: {res_sim.llf:.1f}  |  AIC: {res_sim.aic:.1f}")

smoothed_probs_sim = res_sim.smoothed_marginal_probabilities
smoothed_probs_sim_df = pd.DataFrame({'0': smoothed_probs_sim[:,0],
                                   '1': smoothed_probs_sim[:,1]})
# Identify which column corresponds to which regime by matching means
# High-mean regime gets column 0 if m0 > m1, else swap
high_col = 0 if m0 > m1 else 1
low_col  = 1 - high_col
if high_col == 0:
    smoothed_probs_sim  = smoothed_probs_sim_df.rename(columns = {'0':'high_col', '1':'low_col'})
elif high_col == 1:
    smoothed_probs_sim  = smoothed_probs_sim_df.rename(columns = {'1':'high_col', '0':'low_col'})
else:
    "You have a problem! Try door 3."

# ─────────────────────────────────────────────────────────────────────────────
# PART 2: REAL DATA — Baltic Dry Index (BDI), monthly
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("PART 2: Real data — Baltic Dry Index (BDI), monthly 2000-2020")
print("=" * 65)
print("Note: Embedded monthly BDI index (avg per month, index points)")
print("Source: Baltic Exchange / public historical records")

# Monthly BDI data 2000-2020 (approximate historical values, annual averages
# disaggregated to monthly with realistic within-year variation).
# Key features: 2003-2008 supercycle, crash Oct 2008, structurally lower post-2009.
bdi_annual = {
    2000: 1500, 2001: 1200, 2002: 1100, 2003: 2600, 2004: 4500,
    2005: 3372, 2006: 3226, 2007: 7071, 2008: 6390,  # peaks May 2008 ~11700
    2009: 2617, 2010: 2758, 2011: 1549, 2012: 920,  2013: 1214,
    2014: 1105, 2015:  718, 2016:  673, 2017: 1145, 2018: 1353,
    2019: 1356, 2020:  800
}

# Build monthly series with seasonal noise around annual average
dates = pd.date_range("2000-01-01", "2020-12-01", freq="MS")
np.random.seed(7)
monthly_vals = []
for yr, avg in bdi_annual.items():
    # BDI has seasonal patterns: stronger Q1/Q4, softer Q2/Q3
    seasonal = np.array([1.05, 0.98, 0.95, 0.92, 0.90, 0.88,
                          0.88, 0.90, 0.95, 1.02, 1.08, 1.10])
    noise = np.random.normal(1, 0.06, 12)
    monthly_vals.extend((avg * seasonal * noise).tolist())

bdi = pd.Series(monthly_vals[:len(dates)], index=dates, name="BDI")
bdi = bdi.clip(lower=200)               # floor at 200 (historical minimum)

# Log-transform: BDI is multiplicative in nature
log_bdi = np.log(bdi)

model_bdi = MarkovAutoregression(
    log_bdi,
    k_regimes=2,
    order=1,
    switching_ar=False,
    switching_variance=True
)
res_bdi = model_bdi.fit(disp=False, search_reps=20)

def get_param_from(model,result, name_fragment):
    matches = [(i, n) for i, n in enumerate(model.param_names) 
               if name_fragment in n]
    if not matches:
        raise KeyError(f"No match for '{name_fragment}'. "
                       f"Available: {model.param_names}")
    return result.params[matches[0][0]]


m0_bdi = np.exp(get_param_from(model_bdi, res_bdi, 'const[0]'))
m1_bdi = np.exp(get_param_from(model_bdi,res_bdi, 'const[1]'))
p00_bdi = get_param_from(model_bdi, res_bdi, 'p[0->0]')
p10_bdi = get_param_from(model_bdi, res_bdi, 'p[1->0]')
p11_bdi = 1-p10_bdi


# Ensure regime 0 = high mean
if m0_bdi < m1_bdi:
    m0_bdi, m1_bdi = m1_bdi, m0_bdi
    p00_bdi, p11_bdi = p11_bdi, p00_bdi

print(f"\nEstimated BDI regime means (level): "
      f"High = {m0_bdi:.0f},  Low = {m1_bdi:.0f}")
print(f"Expected duration — High regime: {1/(1-p00_bdi):.1f} months")
print(f"Expected duration — Low  regime: {1/(1-p11_bdi):.1f} months")
print(f"AIC: {res_bdi.aic:.1f}")

smoothed_probs_bdi = res_bdi.smoothed_marginal_probabilities

# Identify which smoothed prob column corresponds to high regime
# (the column whose mean is highest corresponds to the high-level regime)
col0_mean = np.mean(smoothed_probs_bdi[smoothed_probs_bdi.iloc[:, 0] > 0.5]) if (smoothed_probs_bdi.iloc[:, 0] > 0.5).any() else -np.inf
col1_mean = np.mean(smoothed_probs_bdi[smoothed_probs_bdi.iloc[:, 1] > 0.5]) \
    if (smoothed_probs_bdi.iloc[:, 1] > 0.5).any() else -np.inf
high_col_bdi = 0 if col0_mean > col1_mean else 1

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 11))
fig.suptitle("Technique 2: Markov-Switching — Shock Establishes a New Baseline",
             fontsize=13, fontweight='bold', y=0.99)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)

# Panel A: Simulated series coloured by true regime
ax1 = fig.add_subplot(gs[0, :])
t = np.arange(len(y_sim))
# Shade background by true regime
in_low = False
start = 0
for i in range(len(states_sim)):
    if states_sim[i] == 1 and not in_low:
        start = i
        in_low = True
    elif states_sim[i] == 0 and in_low:
        ax1.axvspan(start, i, alpha=0.18, color='tomato', label='_nolegend_')
        in_low = False
if in_low:
    ax1.axvspan(start, len(states_sim)-1, alpha=0.18, color='tomato')
ax1.plot(t, y_sim, 'k-', lw=1.2, alpha=0.85, label='Simulated series')
ax1.axhline(100, color='#1f77b4', lw=1.2, ls='--', label='High regime mean (μ₀=100)')
ax1.axhline(60,  color='tomato',  lw=1.2, ls='--', label='Low regime mean  (μ₁=60)')
ax1.set_title("A  Simulated: two-regime MS-AR(1)  [shading = low regime periods]",
              fontsize=10, loc='left')
ax1.set_xlabel("Period")
ax1.set_ylabel("Level")
ax1.legend(fontsize=8, ncol=4)

# Panel B: Smoothed regime probabilities (simulated)
ax2 = fig.add_subplot(gs[1, 0])
ax2.fill_between(t[1:], smoothed_probs_sim.iloc[:, low_col],
                  alpha=0.6, color='tomato', label='P(low regime)')
ax2.fill_between(t[:1], smoothed_probs_sim.iloc[:, high_col],
                  alpha=0.4, color='steelblue', label='P(high regime)')
ax2.set_title("B  Smoothed regime probabilities (simulated)", fontsize=10, loc='left')
ax2.set_xlabel("Period")
ax2.set_ylabel("Probability")
ax2.set_ylim(0, 1)
ax2.legend(fontsize=8)

# Panel C: Transition matrix heatmap
ax3 = fig.add_subplot(gs[1, 1])
p00e = p00 #res_sim.params['p[0->0]']
p11e = p11 #res_sim.params['p[1->1]']
P_mat = np.array([[p00e,       1 - p00e],
                   [1 - p11e,  p11e]])
im = ax3.imshow(P_mat, cmap='Blues', vmin=0, vmax=1, aspect='auto')
for i in range(2):
    for j in range(2):
        ax3.text(j, i, f"{P_mat[i, j]:.3f}", ha='center', va='center',
                  fontsize=11, color='black' if P_mat[i,j] < 0.7 else 'white')
ax3.set_xticks([0, 1]); ax3.set_xticklabels(['→ High', '→ Low'])
ax3.set_yticks([0, 1]); ax3.set_yticklabels(['From High', 'From Low'])
ax3.set_title("C  Estimated transition matrix", fontsize=10, loc='left')
plt.colorbar(im, ax=ax3, fraction=0.046)

# Panel D: BDI real data + regime shading
ax4 = fig.add_subplot(gs[2, :])
prob_high_bdi = smoothed_probs_bdi.iloc[:, high_col_bdi]
# Shade high-regime periods
in_high = False
start_dt = bdi.index[0]
for i, (dt, p) in enumerate(zip(bdi.index, prob_high_bdi)):
    if p > 0.6 and not in_high:
        start_dt = dt
        in_high = True
    elif p <= 0.6 and in_high:
        ax4.axvspan(start_dt, dt, alpha=0.15, color='#1f77b4', label='_nolegend_')
        in_high = False
if in_high:
    ax4.axvspan(start_dt, bdi.index[-1], alpha=0.15, color='#1f77b4')

ax4.plot(bdi.index, bdi.values, 'k-', lw=1.1, label='BDI (monthly)')
ax4.axhline(m0_bdi, color='#1f77b4', ls='--', lw=1.3,
             label=f'High regime μ≈{m0_bdi:.0f}')
ax4.axhline(m1_bdi, color='tomato', ls='--', lw=1.3,
             label=f'Low regime  μ≈{m1_bdi:.0f}')
ax4.axvline(pd.Timestamp("2008-09-01"), color='gray', ls=':', lw=1.2, alpha=0.8)
ax4.text(pd.Timestamp("2008-10-01"), bdi.max() * 0.88, 'Sep 2008\nLehman',
          fontsize=7.5, color='gray')
ax4.set_title("D  Real: Baltic Dry Index 2000–2020  [blue shading = high regime]",
              fontsize=10, loc='left')
ax4.set_xlabel("Date")
ax4.set_ylabel("BDI index points")
ax4.legend(fontsize=8, ncol=4)

plt.savefig("technique2_markov_switching.png",
            dpi=150, bbox_inches='tight')
print("\nPlot saved: technique2_markov_switching.png")
plt.show()
