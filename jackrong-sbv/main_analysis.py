"""
Main Econometric Analysis: Regime Instability and State-Based Violence

Hypothesis: Countries that have recently experienced regime change experience
higher levels of state-based violence as new governments consolidate power
and deal with opposition.

Identification Strategy: Use time since regime change as the key independent
variable, with country and year-month fixed effects to control for time-invariant
country characteristics and global time trends.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("MAIN ECONOMETRIC ANALYSIS")
print("Regime Instability and State-Based Violence")
print("=" * 70)

# Load data
df = pd.read_csv('cm_with_temp.csv')

# Create key variables
print("\n1. PREPARING VARIABLES")

# Dependent variable: log violence (handles zero-inflation and skew)
df['viol_log'] = np.log1p(df['ged_best_sb'])

# Key independent variable: time since regime change (in months)
# Inverse relationship expected - more recent = more violence
df['time_since_regime_change_inv'] = 1.0 / (df['fvp_timesinceregimechange'] + 1)

# Alternative: dummy for recent regime change (within last 24 months)
df['recent_regime_change'] = (df['fvp_timesinceregimechange'] <= 24).astype(int)

# Controls
controls = [
    'fvp_democracy', 'fvp_liberal', 'fvp_gdpcap_nonoilrent',
    'fvp_population200', 'fvp_ssp2_urban_share_iiasa',
    'fvp_gdpcap_oilrent'
]

print(f"   Observations: {len(df):,}")
print(f"   Countries: {df['country_id'].nunique()}")
print(f"   Years: {df['year'].nunique()}")

# Summary statistics
print("\n2. SUMMARY STATISTICS")
summary_vars = ['ged_best_sb', 'viol_log', 'fvp_timesinceregimechange',
                'time_since_regime_change_inv', 'recent_regime_change']
for var in summary_vars:
    if var in df.columns:
        s = df[var].dropna()
        print(f"   {var}: mean={s.mean():.3f}, sd={s.std():.3f}")

# Model 1: Simple bivariate regression
print("\n3. MODEL 1: BIVARIATE REGRESSION")
print("   viol_log ~ time_since_regime_change_inv")

model1 = ols('viol_log ~ time_since_regime_change_inv', data=df).fit(cov_type='HC3')
print(f"   Coefficient: {model1.params['time_since_regime_change_inv']:.4f}")
print(f"   SE: {model1.bse['time_since_regime_change_inv']:.4f}")
print(f"   t-stat: {model1.tvalues['time_since_regime_change_inv']:.2f}")

# Model 2: With controls
print("\n4. MODEL 2: WITH CONTROLS")
print("   viol_log ~ time_since_regime_change_inv + controls")

control_str = ' + '.join(controls)
model2 = ols(f'viol_log ~ time_since_regime_change_inv + {control_str}', data=df).fit(cov_type='HC3')
print(f"   Coefficient: {model2.params['time_since_regime_change_inv']:.4f}")
print(f"   SE: {model2.bse['time_since_regime_change_inv']:.4f}")
print(f"   t-stat: {model2.tvalues['time_since_regime_change_inv']:.2f}")

# Model 3: With country fixed effects
print("\n5. MODEL 3: WITH COUNTRY FIXED EFFECTS")
print("   viol_log ~ time_since_regime_change_inv + controls + country FE")

model3 = ols(f'viol_log ~ time_since_regime_change_inv + {control_str} + C(country_id)', data=df).fit(cov_type='HC3')
print(f"   Coefficient: {model3.params['time_since_regime_change_inv']:.4f}")
print(f"   SE: {model3.bse['time_since_regime_change_inv']:.4f}")
print(f"   t-stat: {model3.tvalues['time_since_regime_change_inv']:.2f}")

# Model 4: With country and year fixed effects
print("\n6. MODEL 4: WITH COUNTRY AND YEAR FIXED EFFECTS")
print("   viol_log ~ time_since_regime_change_inv + controls + country FE + year FE")

model4 = ols(f'viol_log ~ time_since_regime_change_inv + {control_str} + C(country_id) + C(year)', data=df).fit(cov_type='HC3')
print(f"   Coefficient: {model4.params['time_since_regime_change_inv']:.4f}")
print(f"   SE: {model4.bse['time_since_regime_change_inv']:.4f}")
print(f"   t-stat: {model4.tvalues['time_since_regime_change_inv']:.2f}")

# Model 5: Using recent regime change dummy
print("\n7. MODEL 5: RECENT REGIME CHANGE DUMMY")
print("   viol_log ~ recent_regime_change + controls + country FE + year FE")

model5 = ols(f'viol_log ~ recent_regime_change + {control_str} + C(country_id) + C(year)', data=df).fit(cov_type='HC3')
print(f"   Coefficient: {model5.params['recent_regime_change']:.4f}")
print(f"   SE: {model5.bse['recent_regime_change']:.4f}")
print(f"   t-stat: {model5.tvalues['recent_regime_change']:.2f}")

# Model 6: Non-linear specification (squared term)
print("\n8. MODEL 6: NON-LINEAR SPECIFICATION")
print("   viol_log ~ time_since_regime_change_inv + time_since_regime_change_inv^2 + controls + FEs")

df['time_since_regime_change_inv_sq'] = df['time_since_regime_change_inv'] ** 2

model6 = ols(f'viol_log ~ time_since_regime_change_inv + time_since_regime_change_inv_sq + {control_str} + C(country_id) + C(year)', data=df).fit(cov_type='HC3')
print(f"   Linear coeff: {model6.params['time_since_regime_change_inv']:.4f}")
print(f"   Quadratic coeff: {model6.params['time_since_regime_change_inv_sq']:.4f}")

# Create summary table
print("\n9. SUMMARY TABLE")
print("=" * 70)

models_dict = {
    'Bivariate': model1,
    '+Controls': model2,
    '+Country FE': model3,
    '+Year FE': model4,
    'Recent RC': model5,
    'Non-linear': model6
}

# Extract coefficients for summary
def get_coef(model, var_name):
    if var_name in model.params.index:
        return f"{model.params[var_name]:.4f}"
    return ""

def get_se(model, var_name):
    if var_name in model.bse.index:
        return f"({model.bse[var_name]:.4f})"
    return ""

print("\nDependent Variable: Log(1 + State-Based Violence Deaths)")
print("\n{:<15} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}".format(
    "Specification", "(1)", "(2)", "(3)", "(4)", "(5)", "(6)"))
print("-" * 70)

# Time since regime change inverse
row1 = "{:<15}".format("Time since RC (inv)")
for i, (name, model) in enumerate(models_dict.items()):
    if i < 4 or i == 5:
        coef = get_coef(model, 'time_since_regime_change_inv')
        se = get_se(model, 'time_since_regime_change_inv')
        row1 += f"{coef:>12}\n{se:>12}" if i == 0 else f" {coef:>10}\n {se:>10}"
    else:
        row1 += " " * 12
print(row1)

# Recent regime change
row2 = "{:<15}".format("Recent RC")
for i, (name, model) in enumerate(models_dict.items()):
    if i == 4:
        coef = get_coef(model, 'recent_regime_change')
        se = get_se(model, 'recent_regime_change')
        row2 += f" {coef:>10}\n {se:>10}"
    else:
        row2 += " " * 12
print(row2)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
