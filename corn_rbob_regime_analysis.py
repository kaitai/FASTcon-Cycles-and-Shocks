"""
Corn–RBOB Spread: RFS Regime Change Analysis
=============================================
Spread formula:
    spread_t = Close_rbob_t  -  Close_corn_t / (100 * 2.77)

    Denominator converts corn (cents/bushel) to $/gallon-equivalent:
        100  : cents -> dollars
        2.77 : gallons of ethanol per bushel of corn (energy-equiv. approx.)

    A positive spread means gasoline is expensive relative to corn-ethanol.
    The 2005 Energy Policy Act / 2007 EISA Renewable Fuel Standard (RFS)
    mandated blending volumes that mechanically linked the two prices —
    compressing the spread as corn demand from ethanol surged.

Two analyses:
    1. UCM counterfactual: fit a local-level + seasonal + cycle model on
       the pre-RFS period (1978–2003), forecast through 2024, overlay actual.
       The divergence visualises the regime shift.

    2. Markov-switching AR(1): fit on the full sample to identify regimes,
       estimate regime-specific means and volatilities, plot smoothed
       state probabilities with annotated transition.

Install: pip install statsmodels pandas matplotlib numpy
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & MERGE
# ─────────────────────────────────────────────────────────────────────────────

def load_barchart(path, price_col_name):
    """Read a Barchart CSV (has a header junk row + a footer line)."""
    df = pd.read_csv(path, skiprows=1, parse_dates=['Date Time'])
    # Drop the Barchart footer row (non-parseable date)
    df = df[pd.to_datetime(df['Date Time'], errors='coerce').notna()].copy()
    df['Date Time'] = pd.to_datetime(df['Date Time'])
    df = df[['Date Time', 'Close']].rename(
        columns={'Date Time': 'date', 'Close': price_col_name}
    )
    df = df.set_index('date').sort_index()
    return df

corn = load_barchart(
    'ZCY00_Barchart_Interactive_Chart_Monthly_Nearby_05_27_2024.csv',
    'Close_corn'
)
rbob = load_barchart(
    'RBY00_Barchart_Interactive_Chart_Monthly_Nearby_05_27_2024.csv',
    'Close_rbob'
)

corn_and_gas = corn.join(rbob, how='inner')
corn_and_gas['spread'] = (
    corn_and_gas['Close_rbob'] - corn_and_gas['Close_corn'] / (100 * 2.77)
)
corn_and_gas = corn_and_gas.dropna(subset=['spread'])
corn_and_gas.index.freq = 'MS'

print(f"Merged data: {corn_and_gas.index[0].date()} to "
      f"{corn_and_gas.index[-1].date()}  ({len(corn_and_gas)} months)")
print(f"Spread range: [{corn_and_gas['spread'].min():.3f}, "
      f"{corn_and_gas['spread'].max():.3f}]  $/gal-equiv")
print(f"Pre-RFS mean  (1978–2003): "
      f"{corn_and_gas.loc[:'2003','spread'].mean():.3f}")
print(f"Post-RFS mean (2006–2024): "
      f"{corn_and_gas.loc['2006':,'spread'].mean():.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. UCM COUNTERFACTUAL (pre-RFS training, full-sample forecast)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("UCM counterfactual: train 1978–2003, forecast through 2024")
print("=" * 65)

train = corn_and_gas.loc[:'2003-12-01', 'spread']
full  = corn_and_gas['spread']

# Local level + annual (12-month) seasonal + stochastic cycle
# The seasonal captures within-year refinery/driving patterns.
# cycle_period_bounds: 24–84 months = 2–7 year commodity price cycles.
model_ucm = UnobservedComponents(
    train,
    level='local level',           # random-walk level (no slope: pre-RFS trend flat-ish)
    freq_seasonal=[{'period': 12, 'harmonics': 2}],   # annual seasonality
    cycle=True,
    stochastic_cycle=True,
    damped_cycle=True,
    cycle_period_bounds=(24, 84),
)
res_ucm = model_ucm.fit(disp=False, method='powell', maxiter=3000)

print(f"UCM AIC: {res_ucm.aic:.1f}  |  Log-lik: {res_ucm.llf:.1f}")

# Forecast: steps from end of training to end of full sample
n_fcast = len(full) - len(train)
fcast   = res_ucm.get_forecast(steps=n_fcast)
fcast_mean = fcast.predicted_mean
fcast_ci80 = fcast.conf_int(alpha=0.20)
fcast_ci95 = fcast.conf_int(alpha=0.05)

# In-sample smoothed level for the training period
insample_fit = res_ucm.fittedvalues

# ─────────────────────────────────────────────────────────────────────────────
# 3. MARKOV-SWITCHING AR(1) — full sample
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("Markov-switching AR(1): full sample regime identification")
print("=" * 65)

# k=3 regimes: pre-RFS low spread, transition/volatile, post-RFS compressed
# switching_variance=True: each regime has its own mean AND volatility —
# important here because the post-RFS period is also more volatile.
# switching_ar=False: shared AR coefficient (parsimonious).

def get_param_from(result, name_fragment):
    """Fetch parameter by substring match (handles array-indexed param names)."""
    matches = [(i, n) for i, n in enumerate(result.param_names)
               if name_fragment in n]
    if not matches:
        raise KeyError(f"No match for '{name_fragment}'. "
                       f"Available: {result.param_names}")
    return result.params[matches[0][0]]

# Fit k=2 and k=3; pick lower AIC
results_ms = {}
for k in [2, 3]:
    m = MarkovAutoregression(
        full,
        k_regimes=k,
        order=1,
        switching_ar=False,
        switching_variance=True,
    )
    r = m.fit(disp=False, search_reps=20, maxiter=2000)
    results_ms[k] = r
    print(f"  k={k}: AIC={r.aic:.1f}  log-lik={r.llf:.1f}")

best_k  = min(results_ms, key=lambda k: results_ms[k].aic)
res_ms  = results_ms[best_k]
print(f"\n  Selected k={best_k} regimes (lower AIC)")

# Extract regime parameters
param_names = res_ms.param_names
print(f"  Parameter names: {param_names}")

n_reg = best_k
regime_means  = np.array([get_param_from(res_ms, f'const[{i}]') for i in range(n_reg)])
regime_sigmas = np.array([
    np.sqrt(get_param_from(res_ms, f'sigma2[{i}]')) for i in range(n_reg)
])
ar_coef = get_param_from(res_ms, 'ar.L1')

# Sort regimes by mean (ascending) for consistent labelling
sort_idx     = np.argsort(regime_means)
regime_means  = regime_means[sort_idx]
regime_sigmas = regime_sigmas[sort_idx]

# Transition matrix — statsmodels stores p[i->j] for i < n_reg-1 only.
# Build row-by-row; residual probability fills the missing entry.
trans_mat = np.zeros((n_reg, n_reg))
for i in range(n_reg):
    row_sum = 0.0
    found = []
    for j in range(n_reg):
        try:
            val = get_param_from(res_ms, f'p[{i}->{j}]')
            trans_mat[i, j] = val
            row_sum += val
            found.append(j)
        except KeyError:
            pass
    # One column per row is left as residual
    missing_cols = [j for j in range(n_reg) if j not in found]
    residual = max(0.0, 1.0 - row_sum)
    for j in missing_cols:
        trans_mat[i, j] = residual / max(len(missing_cols), 1)
# Renormalise rows to guard against float noise
row_sums = trans_mat.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
trans_mat = trans_mat / row_sums

# Expected durations
durations = np.array([1.0 / (1.0 - trans_mat[i, i])
                       if trans_mat[i, i] < 1.0 else np.inf
                       for i in range(n_reg)])

print(f"\n  AR(1) coefficient: {ar_coef:.3f}")
print(f"\n  Regime summary (sorted by mean):")
labels = ['Low spread', 'Mid spread', 'High spread'][:n_reg]
for i, (lbl, mu, sig, dur) in enumerate(
        zip(labels, regime_means, regime_sigmas, durations)):
    print(f"    Regime {i} ({lbl}): mean={mu:.3f} $/gal,  "
          f"sigma={sig:.3f},  E[duration]={dur:.1f} mo")

# Smoothed probabilities — reorder columns to match sorted regimes
raw_probs    = res_ms.smoothed_marginal_probabilities
smooth_probs = pd.DataFrame(
    raw_probs.values[:, sort_idx],
    index=full.index,
    columns=[f'regime_{i}' for i in range(n_reg)]
)

# Most likely regime at each date
dominant_regime = smooth_probs.idxmax(axis=1).str[-1].astype(int)

# Monthly seasonality: compute mean spread by calendar month (full sample)
monthly_seasonal = full.groupby(full.index.month).mean()
monthly_seasonal -= monthly_seasonal.mean()     # demean

# ─────────────────────────────────────────────────────────────────────────────
# 4. PLOTS
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    'obs':      '#222222',
    'fit':      '#1f77b4',
    'fcast':    '#ff7f0e',
    'ci80':     '#ffc47a',
    'ci95':     '#ffe5c0',
    'rfs':      '#d62728',
    'regime':   ['#1f77b4', '#2ca02c', '#d62728'],
    'seasonal': '#9467bd',
}

fig = plt.figure(figsize=(15, 18))
fig.suptitle(
    "Corn–RBOB Spread: Renewable Fuel Standard Regime Change\n"
    r"Spread $= P_{RBOB} - P_{corn}/(100 \times 2.77)$   ($/gallon-equivalent)",
    fontsize=13, fontweight='bold', y=0.995
)

gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.48, wspace=0.30,
                        height_ratios=[2.2, 1.8, 1.8, 1.2])

# ── Panel A (top, full width): raw spread + RFS annotation ───────────────────
ax_raw = fig.add_subplot(gs[0, :])
ax_raw.plot(full.index, full.values, color=COLORS['obs'], lw=1.1,
             alpha=0.85, label='Observed spread')
ax_raw.axhline(0, color='gray', lw=0.8, ls=':')

# Pre / post means
pre_mean  = full.loc[:'2004'].mean()
post_mean = full.loc['2008':].mean()
ax_raw.axhline(pre_mean,  color=COLORS['fit'],   lw=1.3, ls='--', alpha=0.7,
                label=f'Pre-RFS mean  ({pre_mean:.2f})')
ax_raw.axhline(post_mean, color=COLORS['fcast'], lw=1.3, ls='--', alpha=0.7,
                label=f'Post-RFS mean ({post_mean:.2f})')

# Shade the transition window
ax_raw.axvspan(pd.Timestamp('2005-01-01'), pd.Timestamp('2008-01-01'),
                alpha=0.12, color=COLORS['rfs'], label='RFS transition (2005–2008)')
ax_raw.axvline(pd.Timestamp('2005-08-01'), color=COLORS['rfs'],
                lw=1.3, ls='--', alpha=0.8)
ax_raw.text(pd.Timestamp('2005-10-01'), full.max() * 0.92,
             'Energy\nPolicy Act\n(Aug 2005)', fontsize=7.5,
             color=COLORS['rfs'], va='top')
ax_raw.axvline(pd.Timestamp('2007-12-01'), color=COLORS['rfs'],
                lw=1.3, ls=':', alpha=0.8)
ax_raw.text(pd.Timestamp('2007-06-01'), full.max() * 0.78,
             'EISA\n(Dec 2007)', fontsize=7.5,
             color=COLORS['rfs'], va='top')

ax_raw.set_title("A  Corn–RBOB spread: full history with regime means",
                  fontsize=10, loc='left')
ax_raw.set_ylabel("$/gallon-equivalent")
ax_raw.legend(fontsize=8, ncol=4)
ax_raw.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax_raw.xaxis.set_major_locator(mdates.YearLocator(4))

# ── Panel B (left): UCM counterfactual forecast ───────────────────────────────
ax_cf = fig.add_subplot(gs[1, :])

# 95% and 80% CI
ax_cf.fill_between(fcast_ci95.index,
                    fcast_ci95.iloc[:, 0], fcast_ci95.iloc[:, 1],
                    color=COLORS['ci95'], label='95% forecast band')
ax_cf.fill_between(fcast_ci80.index,
                    fcast_ci80.iloc[:, 0], fcast_ci80.iloc[:, 1],
                    color=COLORS['ci80'], label='80% forecast band')

# In-sample fit
ax_cf.plot(insample_fit.index, insample_fit.values,
            color=COLORS['fit'], lw=1.3, ls='--', alpha=0.85,
            label='UCM in-sample fit (1978–2003)')

# Forecast mean
ax_cf.plot(fcast_mean.index, fcast_mean.values,
            color=COLORS['fcast'], lw=1.8,
            label='UCM forecast (no RFS world)')

# Actual (full)
ax_cf.plot(full.index, full.values,
            color=COLORS['obs'], lw=1.1, alpha=0.8,
            label='Observed spread')

# Shade divergence region
ax_cf.fill_between(
    fcast_mean.index,
    fcast_mean.values,
    full.loc[fcast_mean.index].values,
    where=(full.loc[fcast_mean.index].values < fcast_mean.values),
    alpha=0.25, color=COLORS['rfs'],
    label='RFS impact (actual < counterfactual)'
)

ax_cf.axvline(pd.Timestamp('2004-01-01'), color='gray', lw=1.0,
               ls=':', alpha=0.8)
ax_cf.text(pd.Timestamp('2004-03-01'),
            full.min() + 0.05,
            'Training\ncutoff', fontsize=7.5, color='gray')
ax_cf.axvspan(pd.Timestamp('2005-01-01'), pd.Timestamp('2008-01-01'),
               alpha=0.08, color=COLORS['rfs'])

ax_cf.set_title(
    "B  UCM counterfactual: model trained 1978–2003 (pre-RFS) forecasted forward\n"
    "   Divergence between orange forecast and black actual = RFS regime impact",
    fontsize=10, loc='left'
)
ax_cf.set_ylabel("$/gallon-equivalent")
ax_cf.legend(fontsize=8, ncol=3)
ax_cf.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax_cf.xaxis.set_major_locator(mdates.YearLocator(4))

# ── Panel C (left): Markov smoothed regime probabilities ─────────────────────
ax_reg = fig.add_subplot(gs[2, :])

for i in range(n_reg):
    ax_reg.fill_between(
        smooth_probs.index,
        smooth_probs[f'regime_{i}'],
        alpha=0.55,
        color=COLORS['regime'][i % len(COLORS['regime'])],
        label=f"Regime {i}: {labels[i]}  (μ={regime_means[i]:.2f}, σ={regime_sigmas[i]:.2f})"
    )

ax_reg.axvline(pd.Timestamp('2005-08-01'), color=COLORS['rfs'],
                lw=1.3, ls='--', alpha=0.8, label='Energy Policy Act (2005)')
ax_reg.axvline(pd.Timestamp('2007-12-01'), color=COLORS['rfs'],
                lw=1.3, ls=':', alpha=0.8, label='EISA / RFS2 (2007)')
ax_reg.set_ylim(0, 1)
ax_reg.set_ylabel("P(regime)")
ax_reg.set_title(
    f"C  Markov-switching AR(1): smoothed regime probabilities  "
    f"(k={best_k} regimes,  AR coef={ar_coef:.3f})",
    fontsize=10, loc='left'
)
ax_reg.legend(fontsize=8, ncol=2)
ax_reg.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax_reg.xaxis.set_major_locator(mdates.YearLocator(4))

# ── Panel D (left): Regime-conditional density + seasonal ────────────────────
ax_dens = fig.add_subplot(gs[3, 0])

spread_vals = full.values
x_range = np.linspace(spread_vals.min() - 0.1, spread_vals.max() + 0.1, 400)

for i in range(n_reg):
    # Kernel density estimate weighted by smoothed probability of this regime
    weights = smooth_probs[f'regime_{i}'].values
    weights = weights / weights.sum()
    mu_i  = (weights * spread_vals).sum()
    var_i = (weights * (spread_vals - mu_i) ** 2).sum()
    sig_i = np.sqrt(var_i)
    kde   = np.exp(-0.5 * ((x_range - mu_i) / sig_i) ** 2) / (sig_i * np.sqrt(2 * np.pi))
    ax_dens.fill_between(x_range, kde, alpha=0.40,
                          color=COLORS['regime'][i % len(COLORS['regime'])])
    ax_dens.plot(x_range, kde, lw=1.5,
                  color=COLORS['regime'][i % len(COLORS['regime'])],
                  label=f"{labels[i]}  μ={mu_i:.2f}")
    ax_dens.axvline(mu_i, color=COLORS['regime'][i % len(COLORS['regime'])],
                     lw=1.0, ls='--', alpha=0.7)

ax_dens.set_title("D  Regime-conditional spread distributions",
                   fontsize=10, loc='left')
ax_dens.set_xlabel("Spread ($/gallon-equivalent)")
ax_dens.set_ylabel("Density")
ax_dens.legend(fontsize=8)

# ── Panel E (right): Monthly seasonality ─────────────────────────────────────
ax_seas = fig.add_subplot(gs[3, 1])

month_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
bar_colors  = [COLORS['regime'][0] if v >= 0
               else COLORS['regime'][2]
               for v in monthly_seasonal.values]
ax_seas.bar(range(1, 13), monthly_seasonal.values,
             color=bar_colors, alpha=0.7, width=0.7)
ax_seas.axhline(0, color='black', lw=0.8, ls=':')
ax_seas.set_xticks(range(1, 13))
ax_seas.set_xticklabels(month_names, fontsize=8)
ax_seas.set_title("E  Monthly seasonal pattern (demeaned, full sample)",
                   fontsize=10, loc='left')
ax_seas.set_ylabel("Seasonal deviation\n($/gallon-equiv.)")
ax_seas.text(0.98, 0.95,
              "Summer driving\nseason: RBOB\npremium",
              transform=ax_seas.transAxes, fontsize=7.5,
              color=COLORS['regime'][0], ha='right', va='top')

plt.savefig('/mnt/user-data/outputs/corn_rbob_regime_analysis.png',
            dpi=150, bbox_inches='tight')
print("\nPlot saved: corn_rbob_regime_analysis.png")
plt.show()
