"""
Main Analysis Script - Final Version
Historical Trauma and Cultural Consumption: Evidence from Demon Slayer in China

RESEARCH DESIGN:
This study examines whether historical anti-Japanese sentiment affects contemporary
cultural consumption. We compare audience counts for Demon Slayer (Japanese anime)
vs. Zootopia 2 (American animation) across Chinese cities with and without anti-Japanese
war memorials.

Identification: The key variation is cross-city. Cities with memorials (treatment)
are compared to cities without (control). We examine whether the Japanese-American
film audience gap differs between treatment and control cities.

Model: audience = β0 + β1*japanese_film + β2*treatment*japanese_film + city_FE + ε

Note: We cannot include date FE because the interaction is constant within city-date.
The treatment effect is identified from cross-city variation.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ANALYSIS: Historical Trauma and Cultural Consumption")
print("="*70)

# ============================================================================
# 1. DATA LOADING AND PREPARATION
# ============================================================================
print("\n[1/5] Loading and preparing data...")

# Load data
kny = pd.read_csv('kny_citylevel_1113_1211.csv')  # Demon Slayer
ztp = pd.read_excel('ztp_citylevel_1126_1226.xlsx')  # Zootopia 2
anti_jp = pd.read_csv('anti_japanese_site.csv')  # Anti-Japanese sites

print(f"  Demon Slayer: {len(kny)} obs, {kny['city_CN'].nunique()} cities")
print(f"  Zootopia 2: {len(ztp)} obs, {ztp['city_CN'].nunique()} cities")
print(f"  Anti-Japanese sites: {len(anti_jp)} sites")

# Create city-level treatment indicator
city_memorial_count = anti_jp.groupby('所在城市').size().reset_index(name='memorial_count')
city_memorial_count = city_memorial_count.rename(columns={'所在城市': 'city'})
city_memorial_count['treatment'] = 1

print(f"  Cities with memorials: {city_memorial_count['treatment'].sum()}")

# ============================================================================
# 2. CREATE PANEL DATASET
# ============================================================================
print("\n[2/5] Creating panel dataset...")

# Filter to overlap period: Nov 26 - Dec 11, 2025
overlap_start = '2025-11-26'
overlap_end = '2025-12-11'

kny_overlap = kny[(kny['date'] >= overlap_start) & (kny['date'] <= overlap_end)].copy()
ztp_overlap = ztp[(ztp['date'] >= overlap_start) & (ztp['date'] <= overlap_end)].copy()

print(f"  Demon Slayer (overlap): {len(kny_overlap)} obs")
print(f"  Zootopia 2 (overlap): {len(ztp_overlap)} obs")

# Add film indicator
kny_overlap['film_type'] = 'japanese'
ztp_overlap['film_type'] = 'american'

# Standardize column names
kny_overlap = kny_overlap.rename(columns={
    'city_CN': 'city',
    'province_CN': 'province',
    'audience_count': 'audience'
})

ztp_overlap = ztp_overlap.rename(columns={
    'city_CN': 'city',
    'province_CN': 'province',
    'audience_count (人次)': 'audience'
})

# Keep only relevant columns
kny_panel = kny_overlap[['date', 'city', 'province', 'audience', 'screening_count', 'film_type']]
ztp_panel = ztp_overlap[['date', 'city', 'province', 'audience', 'screening_count (场次)', 'film_type']].copy()
ztp_panel = ztp_panel.rename(columns={'screening_count (场次)': 'screening_count'})

# Combine
panel = pd.concat([kny_panel, ztp_panel], ignore_index=True)

print(f"  Combined panel: {len(panel)} obs")

# Merge treatment indicator
panel = panel.merge(city_memorial_count, on='city', how='left')
panel['treatment'] = panel['treatment'].fillna(0).astype(int)
panel['memorial_count'] = panel['memorial_count'].fillna(0).astype(int)

# Create binary indicator for Japanese film
panel['japanese_film'] = (panel['film_type'] == 'japanese').astype(int)

# Create interaction term
panel['treat_x_japanese'] = panel['treatment'] * panel['japanese_film']

print(f"  Cities with treatment: {panel['treatment'].sum()}")
print(f"  Japanese film observations: {panel['japanese_film'].sum()}")

# ============================================================================
# 3. DESCRIPTIVE STATISTICS
# ============================================================================
print("\n[3/5] Descriptive statistics...")

# Descriptive table
desc = panel.groupby(['treatment', 'japanese_film']).agg({
    'audience': ['mean', 'std', 'count'],
    'screening_count': 'mean'
}).round(2)
desc.columns = ['audience_mean', 'audience_std', 'n_obs', 'screening_mean']
desc = desc.reset_index()
desc['group'] = (
    desc.apply(lambda x: 'Control-American' if x['treatment']==0 and x['japanese_film']==0 else
                       'Control-Japanese' if x['treatment']==0 and x['japanese_film']==1 else
                       'Treatment-American' if x['treatment']==1 and x['japanese_film']==0 else
                       'Treatment-Japanese', axis=1)
)

print("\n  Descriptive Statistics by Group:")
print("  " + "-"*60)
for _, row in desc.iterrows():
    print(f"  {row['group']:20s}: Audience={row['audience_mean']:10.0f} (n={int(row['n_obs'])})")

# Calculate DID estimate manually
ctrl_am = panel[(panel['treatment']==0) & (panel['japanese_film']==0)]['audience'].mean()
ctrl_jp = panel[(panel['treatment']==0) & (panel['japanese_film']==1)]['audience'].mean()
treat_am = panel[(panel['treatment']==1) & (panel['japanese_film']==0)]['audience'].mean()
treat_jp = panel[(panel['treatment']==1) & (panel['japanese_film']==1)]['audience'].mean()

print(f"\n  Pre-trend (American film):")
print(f"    Control cities: {ctrl_am:.0f}")
print(f"    Treatment cities: {treat_am:.0f}")
print(f"    Difference: {treat_am - ctrl_am:.0f}")
print(f"\n  Post-treatment (Japanese film):")
print(f"    Control cities: {ctrl_jp:.0f}")
print(f"    Treatment cities: {treat_jp:.0f}")
print(f"    Difference: {treat_jp - ctrl_jp:.0f}")
print(f"\n  DID Estimate: ({treat_jp:.0f} - {ctrl_jp:.0f}) - ({treat_am:.0f} - {ctrl_am:.0f}) = {(treat_jp-ctrl_jp) - (treat_am-ctrl_am):.0f}")

# ============================================================================
# 4. REGRESSION ANALYSIS
# ============================================================================
print("\n[4/5] Running regressions...")

# Log transform audience (add 1 to handle zeros)
panel['log_audience'] = np.log1p(panel['audience'])

# Model 1: Simple DID (no FE)
X1 = sm.add_constant(panel[['treatment', 'japanese_film', 'treat_x_japanese']])
y1 = panel['log_audience']
model1 = sm.OLS(y1, X1).fit(cov_type='cluster', cov_kwds={'groups': panel['city']})

print("\n  Model 1: Simple DID (no fixed effects)")
print("  " + "-"*50)
print(f"  {'Variable':<25} {'Coeff':>12} {'SE':>12} {'t':>8} {'p':>8}")
print("  " + "-"*50)
for var in ['treatment', 'japanese_film', 'treat_x_japanese']:
    coef = model1.params[var]
    se = model1.bse[var]
    t_stat = coef / se
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(panel)-len(X1.columns)))
    print(f"  {var:<25} {coef:>12.4f} {se:>12.4f} {t_stat:>8.2f} {p_val:>8.4f}")
print(f"  {'R-squared':<25} {model1.rsquared:>12.4f}")

# Model 2: DID with city fixed effects (MAIN SPECIFICATION)
print("\n  Model 2: DID with City Fixed Effects (Main Specification)")
print("  " + "-"*50)
print(f"  {'Variable':<25} {'Coeff':>12} {'SE':>12} {'t':>8} {'p':>8}")
print("  " + "-"*50)

# Manual within transformation for city FE
panel_grouped = panel.groupby('city')
panel['log_audience_within'] = panel['log_audience'] - panel_grouped['log_audience'].transform('mean')
panel['treat_x_japanese_within'] = panel['treat_x_japanese'] - panel_grouped['treat_x_japanese'].transform('mean')
panel['japanese_film_within'] = panel['japanese_film'] - panel_grouped['japanese_film'].transform('mean')

X2_within = sm.add_constant(panel[['japanese_film_within', 'treat_x_japanese_within']])
y2_within = panel['log_audience_within']
model2 = sm.OLS(y2_within, X2_within).fit(cov_type='cluster', cov_kwds={'groups': panel['city']})

for var in ['japanese_film_within', 'treat_x_japanese_within']:
    coef = model2.params[var]
    se = model2.bse[var]
    t_stat = coef / se
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(panel)-len(X2_within.columns)))
    display_var = var.replace('_within', '')
    print(f"  {display_var:<25} {coef:>12.4f} {se:>12.4f} {t_stat:>8.2f} {p_val:>8.4f}")

# Model 3: With controls (screening count)
print("\n  Model 3: With Controls (screening count)")
print("  " + "-"*50)
print(f"  {'Variable':<25} {'Coeff':>12} {'SE':>12} {'t':>8} {'p':>8}")
print("  " + "-"*50)

panel['log_screening'] = np.log1p(panel['screening_count'])
panel['log_screening_within'] = panel['log_screening'] - panel_grouped['log_screening'].transform('mean')

X3_within = sm.add_constant(panel[['japanese_film_within', 'treat_x_japanese_within', 'log_screening_within']])
y3_within = panel['log_audience_within']
model3 = sm.OLS(y3_within, X3_within).fit(cov_type='cluster', cov_kwds={'groups': panel['city']})

for var in ['japanese_film_within', 'treat_x_japanese_within', 'log_screening_within']:
    coef = model3.params[var]
    se = model3.bse[var]
    t_stat = coef / se
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(panel)-len(X3_within.columns)))
    display_var = var.replace('_within', '')
    print(f"  {display_var:<25} {coef:>12.4f} {se:>12.4f} {t_stat:>8.2f} {p_val:>8.4f}")

# Store results
did_results = {
    'model1_coef': model1.params['treat_x_japanese'],
    'model1_se': model1.bse['treat_x_japanese'],
    'model2_coef': model2.params['treat_x_japanese_within'],
    'model2_se': model2.bse['treat_x_japanese_within'],
    'model3_coef': model3.params['treat_x_japanese_within'],
    'model3_se': model3.bse['treat_x_japanese_within'],
}

# ============================================================================
# 5. PLACEBO TESTS
# ============================================================================
print("\n[5/5] Running placebo tests...")

# Placebo: Random treatment assignment at CITY level
np.random.seed(42)
unique_cities = panel['city'].unique()
random_treatment = np.random.choice([0, 1], size=len(unique_cities), p=[0.5, 0.5])
city_to_random_treat = dict(zip(unique_cities, random_treatment))
panel['placebo_treatment'] = panel['city'].map(city_to_random_treat)
panel['placebo_interaction'] = panel['placebo_treatment'] * panel['japanese_film']

# Within transformation for placebo
panel['placebo_interaction_within'] = panel['placebo_interaction'] - panel_grouped['placebo_interaction'].transform('mean')

X_placebo = sm.add_constant(panel[['japanese_film_within', 'placebo_interaction_within']])
y_placebo = panel['log_audience_within']
model_placebo = sm.OLS(y_placebo, X_placebo).fit(cov_type='cluster', cov_kwds={'groups': panel['city']})

print(f"\n  Placebo Test (random treatment at city level):")
print(f"    Coefficient: {model_placebo.params['placebo_interaction_within']:.4f}")
print(f"    SE: {model_placebo.bse['placebo_interaction_within']:.4f}")
print(f"    t-stat: {model_placebo.params['placebo_interaction_within']/model_placebo.bse['placebo_interaction_within']:.2f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY OF RESULTS")
print("="*70)
print(f"\nMain finding: The interaction term (treatment \u00d7 japanese_film) is")
coef_sign = did_results['model2_coef']
if abs(coef_sign) < did_results['model2_se'] * 1.96:
    print("not statistically significant at conventional levels.")
elif coef_sign < 0:
    print("negative and statistically significant.")
else:
    print("positive and statistically significant.")

if coef_sign != 0:
    p_level = 1 - stats.t.cdf(abs(coef_sign/did_results['model2_se']), df=len(panel)-len(unique_cities))
    print(f"Two-tailed p-value: {2*p_level:.4f}")

print(f"\nThis suggests that cities with anti-Japanese war memorials show")
if coef_sign < 0:
    print("LOWER relative audience counts for the Japanese film Demon Slayer")
elif coef_sign > 0:
    print("HIGHER relative audience counts for the Japanese film Demon Slayer")
else:
    print("NO DIFFERENT relative audience counts for the Japanese film Demon Slayer")
print("compared to the American film Zootopia 2.")
print("="*70)

# Save results
results_summary = pd.DataFrame({
    'Model': ['Simple DID', 'City FE', 'City FE + Controls'],
    'Coefficient': [did_results['model1_coef'], did_results['model2_coef'], did_results['model3_coef']],
    'SE': [did_results['model1_se'], did_results['model2_se'], did_results['model3_se']]
})
results_summary.to_csv('did_results.csv', index=False)
print("\nResults saved to did_results.csv")
