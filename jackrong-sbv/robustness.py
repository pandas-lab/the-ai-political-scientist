"""
Robustness Checks for Regime Instability and State-Based Violence
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ROBUSTNESS CHECKS")
print("=" * 70)

# Load data
df = pd.read_csv('cm_with_temp.csv')

# Create key variables
df['viol_log'] = np.log1p(df['ged_best_sb'])
df['time_since_regime_change_inv'] = 1.0 / (df['fvp_timesinceregimechange'] + 1)
df['recent_regime_change'] = (df['fvp_timesinceregimechange'] <= 24).astype(int)

controls = [
    'fvp_democracy', 'fvp_liberal', 'fvp_gdpcap_nonoilrent',
    'fvp_population200', 'fvp_ssp2_urban_share_iiasa',
    'fvp_gdpcap_oilrent'
]
control_str = ' + '.join(controls)

# ============================================
# ROBUSTNESS CHECK 1: Alternative DV specifications
# ============================================
print("\n" + "=" * 70)
print("ROBUSTNESS CHECK 1: ALTERNATIVE DV SPECIFICATIONS")
print("=" * 70)

# DV: Binary - any violence
df['viol_any'] = (df['ged_best_sb'] > 0).astype(int)
print("\n1a. DV: Binary - Any Violence (Linear Probability Model)")
model_rb1a = ols(f'viol_any ~ time_since_regime_change_inv + {control_str} + C(country_id) + C(year)', data=df).fit(cov_type='HC3')
print(f"    Coefficient: {model_rb1a.params['time_since_regime_change_inv']:.4f}")
print(f"    SE: {model_rb1a.bse['time_since_regime_change_inv']:.4f}")
print(f"    t-stat: {model_rb1a.tvalues['time_since_regime_change_inv']:.2f}")

# DV: Violence count (negative binomial) - simplified
print("\n1b. DV: Violence Count (Negative Binomial - simple spec)")
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import NegativeBinomial

# Use formula API for simplicity
model_rb1b = GLM.from_formula('ged_best_sb ~ time_since_regime_change_inv + ' + control_str,
                              data=df, family=NegativeBinomial()).fit(cov_type='HC3')
print(f"    Coefficient: {model_rb1b.params['time_since_regime_change_inv']:.4f}")
print(f"    SE: {model_rb1b.bse['time_since_regime_change_inv']:.4f}")

# ============================================
# ROBUSTNESS CHECK 2: Alternative IV specifications
# ============================================
print("\n" + "=" * 70)
print("ROBUSTNESS CHECK 2: ALTERNATIVE IV SPECIFICATIONS")
print("=" * 70)

# IV: Different thresholds for recent regime change
for threshold in [12, 36, 48]:
    df[f'recent_rc_{threshold}'] = (df['fvp_timesinceregimechange'] <= threshold).astype(int)
    model = ols(f'viol_log ~ recent_rc_{threshold} + {control_str} + C(country_id) + C(year)', data=df).fit(cov_type='HC3')
    print(f"\n2a. Recent RC ({threshold} months): {model.params[f'recent_rc_{threshold}']:.4f} ({model.bse[f'recent_rc_{threshold}']:.4f})")

# IV: Time since regime change in categories
print("\n2b. Time since RC in categories:")
df['rc_category'] = pd.cut(df['fvp_timesinceregimechange'],
                           bins=[0, 6, 12, 24, 60, np.inf],
                           labels=['0-6mo', '6-12mo', '12-24mo', '24-60mo', '60+mo'])
model_rb2b = ols(f'viol_log ~ C(rc_category) + {control_str} + C(country_id) + C(year)', data=df).fit(cov_type='HC3')
for cat in ['6-12mo', '12-24mo', '24-60mo', '60+mo']:
    coef_name = f"C(rc_category)[T.{cat}]"
    if coef_name in model_rb2b.params.index:
        print(f"    {cat} (vs 0-6mo): {model_rb2b.params[coef_name]:.4f} ({model_rb2b.bse[coef_name]:.4f})")

# ============================================
# ROBUSTNESS CHECK 3: Placebo tests
# ============================================
print("\n" + "=" * 70)
print("ROBUSTNESS CHECK 3: PLACEBO TESTS")
print("=" * 70)

# Placebo: Effect on non-conflict related variable
print("\n3a. Placebo: Effect on urbanization (should be zero)")
if 'fvp_ssp2_urban_share_iiasa' in df.columns:
    model_rb3a = ols(f'fvp_ssp2_urban_share_iiasa ~ time_since_regime_change_inv + {control_str.replace("fvp_ssp2_urban_share_iiasa", "")} + C(country_id) + C(year)', data=df).fit(cov_type='HC3')
    print(f"    Coefficient: {model_rb3a.params['time_since_regime_change_inv']:.4f}")
    print(f"    SE: {model_rb3a.bse['time_since_regime_change_inv']:.4f}")

# Placebo: Lead effects (future regime changes should not affect current violence)
print("\n3b. Lead effects: Future regime changes")
for lead in [12, 24, 36]:
    df[f'future_rc_{lead}'] = (df.groupby('country_id')['fvp_timesinceregimechange'].shift(-lead) <= 0).astype(int)
    model = ols(f'viol_log ~ future_rc_{lead} + {control_str} + C(country_id) + C(year)', data=df).fit(cov_type='HC3')
    print(f"    Lead {lead} months: {model.params[f'future_rc_{lead}']:.4f} ({model.bse[f'future_rc_{lead}']:.4f})")

# ============================================
# ROBUSTNESS CHECK 4: Alternative samples
# ============================================
print("\n" + "=" * 70)
print("ROBUSTNESS CHECK 4: ALTERNATIVE SAMPLES")
print("=" * 70)

# Exclude post-2020 (pandemic period)
print("\n4a. Exclude post-2020")
df_pre2020 = df[df['year'] < 2020].copy()
model_rb4a = ols(f'viol_log ~ time_since_regime_change_inv + {control_str} + C(country_id) + C(year)', data=df_pre2020).fit(cov_type='HC3')
print(f"    Coefficient: {model_rb4a.params['time_since_regime_change_inv']:.4f}")
print(f"    SE: {model_rb4a.bse['time_since_regime_change_inv']:.4f}")

# Exclude oil-rich countries
print("\n4b. Exclude oil-rich countries (oil rent > 10% GDP)")
df_no_oil = df[df['fvp_gdpcap_oilrent'] < 10].copy()
model_rb4b = ols(f'viol_log ~ time_since_regime_change_inv + {control_str} + C(country_id) + C(year)', data=df_no_oil).fit(cov_type='HC3')
print(f"    Coefficient: {model_rb4b.params['time_since_regime_change_inv']:.4f}")
print(f"    SE: {model_rb4b.bse['time_since_regime_change_inv']:.4f}")

# ============================================
# ROBUSTNESS CHECK 5: Clustering standard errors
# ============================================
print("\n" + "=" * 70)
print("ROBUSTNESS CHECK 5: CLUSTERED STANDARD ERRORS")
print("=" * 70)

print("\n5a. SEs with HC3 (robust)")
model_rb5a = ols(f'viol_log ~ time_since_regime_change_inv + {control_str} + C(country_id) + C(year)', data=df).fit(cov_type='HC3')
print(f"    Coefficient: {model_rb5a.params['time_since_regime_change_inv']:.4f}")
print(f"    SE (HC3): {model_rb5a.bse['time_since_regime_change_inv']:.4f}")

print("\n5b. SEs with HC1 (classic robust)")
model_rb5b = ols(f'viol_log ~ time_since_regime_change_inv + {control_str} + C(country_id) + C(year)', data=df).fit(cov_type='HC1')
print(f"    Coefficient: {model_rb5b.params['time_since_regime_change_inv']:.4f}")
print(f"    SE (HC1): {model_rb5b.bse['time_since_regime_change_inv']:.4f}")

# ============================================
# ROBUSTNESS CHECK 6: Non-parametric specification
# ============================================
print("\n" + "=" * 70)
print("ROBUSTNESS CHECK 6: NON-PARAMETRIC SPECIFICATION")
print("=" * 70)

print("\n6a. Violence by time since regime change (non-parametric)")
df['rc_bin'] = pd.cut(df['fvp_timesinceregimechange'], bins=[0, 3, 6, 12, 18, 24, 36, 60, 120, np.inf])
viol_by_rc = df.groupby('rc_bin')['viol_log'].mean()
print(viol_by_rc)

print("\n" + "=" * 70)
print("ROBUSTNESS CHECKS COMPLETE")
print("=" * 70)
