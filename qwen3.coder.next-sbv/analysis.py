#!/usr/bin/env python3
"""
Research: Democratic Transitions and State-Based Violence
Author: AI Political Scientist
Date: 2026-04-04

Hypothesis: Democratic transitions cause an increase in state-based violence due to
institutional uncertainty, weakened state capacity, and increased opposition challenges.

Methodology: Difference-in-Differences (DiD) using democracy transition as treatment
event. Countries that experience transitions form the treatment group; those that don't
form the control group.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load data
print("=" * 70)
print("LOADING DATA")
print("=" * 70)
df = pd.read_csv('cm.csv')
print(f"Original dataset: {len(df)} rows, {len(df.columns)} columns")
print(f"Country count: {df['country_id'].nunique()}")
print(f"Time period: {df['month_id'].min()} to {df['month_id'].max()}")

# Define dependent variable
dependent_var = 'ged_best_sb'

print("\n" + "=" * 70)
print("DATA PREPARATION")
print("=" * 70)

# Step 1: Identify FIRST month of each democracy transition
transition_rows = df[df['vdem_e_democracy_trans'] == 1]
first_transitions = transition_rows.groupby('country_id')['month_id'].min().reset_index()
first_transitions.columns = ['country_id', 'transition_month']

# Merge back to main dataframe
df = df.merge(first_transitions, on='country_id', how='left')

# Step 2: Create treatment indicators
df['treatment_group'] = df['transition_month'].notna().astype(int)

# Create post_treatment only for treatment group (months >= first transition month)
df['post_treatment'] = 0
df.loc[(df['treatment_group'] == 1) & (df['month_id'] >= df['transition_month']), 'post_treatment'] = 1

# DiD interaction
df['did_interaction'] = df['treatment_group'] * df['post_treatment']

# Step 3: Create event study variables
df['months_since_transition'] = np.nan
df.loc[df['treatment_group'] == 1, 'months_since_transition'] = (
    df.loc[df['treatment_group'] == 1, 'month_id'] -
    df.loc[df['treatment_group'] == 1, 'transition_month']
)

print(f"Countries with democracy transitions: {len(first_transitions)}")

# Descriptive statistics
print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

print(f"\nDependent variable ({dependent_var}):")
print(f"  Mean: {df[dependent_var].mean():.2f}")
print(f"  Median: {df[dependent_var].median():.2f}")
print(f"  Std: {df[dependent_var].std():.2f}")
print(f"  Min: {df[dependent_var].min():.2f}")
print(f"  Max: {df[dependent_var].max():.2f}")
print(f"  Zero count: {(df[dependent_var] == 0).sum()} ({(df[dependent_var] == 0).mean()*100:.1f}%)")
print(f"  NaN count: {df[dependent_var].isna().sum()} ({df[dependent_var].isna().mean()*100:.1f}%)")

# Violations by treatment status
print(f"\nViolations by treatment status:")
treat_violence = df[df['treatment_group'] == 1][dependent_var]
control_violence = df[df['treatment_group'] == 0][dependent_var]
print(f"  Treatment group: Mean={treat_violence.mean():.2f}, N={len(treat_violence)}")
print(f"  Control group: Mean={control_violence.mean():.2f}, N={len(control_violence)}")

# Pre-treatment balance check
print("\n" + "=" * 70)
print("PRE-TREATMENT BALANCE CHECK")
print("=" * 70)

# Get pre-transition period for balance check
pre_transition = df[(df['treatment_group'] == 1) & (df['months_since_transition'] < 0)]

covariates = ['fvp_demo', 'fvp_gdppc200', 'fvp_population200', 'fvp_regime3c', 'fvp_auto']
print("\nPre-transition covariate means (Treatment only - before transition):")
for var in covariates:
    if var in pre_transition.columns:
        mean_val = pre_transition[var].mean()
        n = pre_transition[var].count()
        print(f"  {var}: Mean={mean_val:.3f} (n={n})")

# Compare treatment vs control in pre-period
pre_all = df[(df['months_since_transition'] < 0) | (df['months_since_transition'].isna())]
print("\nPre-period comparison (Treatment vs Control):")
for var in covariates:
    if var in pre_all.columns:
        treat_mean = pre_all[pre_all['treatment_group'] == 1][var].mean()
        control_mean = pre_all[pre_all['treatment_group'] == 0][var].mean()
        print(f"  {var}: Treatment={treat_mean:.3f}, Control={control_mean:.3f}")

# Regression analysis
print("\n" + "=" * 70)
print("REGRESSION ANALYSIS")
print("=" * 70)

# Prepare data for regression - drop NaN values in dependent variable
df_clean = df.copy()

# For DiD, we need both pre and post observations for treatment group
countries_with_full_pre = df[df['months_since_transition'] < 0]['country_id'].unique()
df_clean['has_full_pre'] = df_clean['country_id'].isin(countries_with_full_pre).astype(int)

# Filter to countries with at least one pre-transition observation AND non-NaN dependent variable
reg_df = df_clean[df_clean['has_full_pre'] == 1].copy()
# Drop rows with NaN in dependent variable
reg_df = reg_df[reg_df[dependent_var].notna()].copy()
reg_df = reg_df.reset_index(drop=True)

print(f"Regression sample size (countries with full pre/post): {len(reg_df)}")

# Create numeric country index as plain numpy array
country_codes = reg_df['country_id'].astype('category')
country_codes_num = country_codes.cat.codes.to_numpy(dtype=int)

# Extract variables as numpy arrays
y = reg_df[dependent_var].to_numpy(dtype=float)
treatment = reg_df['did_interaction'].to_numpy(dtype=float)
treatment_group = reg_df['treatment_group'].to_numpy(dtype=float)
post_treatment = reg_df['post_treatment'].to_numpy(dtype=float)

# Control variables
fvp_demo = reg_df['fvp_demo'].to_numpy(dtype=float)
fvp_gdppc200 = reg_df['fvp_gdppc200'].to_numpy(dtype=float)
fvp_population200 = reg_df['fvp_population200'].to_numpy(dtype=float)

# Build design matrix with country dummies
country_dummies = pd.get_dummies(reg_df['country_id'], prefix='country').to_numpy(dtype=float)
X = np.column_stack([treatment, fvp_demo, fvp_gdppc200, fvp_population200, country_dummies])
X = sm.add_constant(X)

print(f"Design matrix shape: {X.shape}")
print(f"Groups (countries): {len(np.unique(country_codes_num))}")

# Fit OLS with cluster-robust SE
print("\n--- Difference-in-Differences Regression (Main) ---")
model = sm.OLS(y, X)
result = model.fit(cov_type='cluster', cov_kwds={'groups': country_codes_num})

# Print key results
did_coef_idx = 1  # did_interaction is second column after constant
did_coef = result.params[did_coef_idx]
did_se = result.bse[did_coef_idx]
did_pval = result.pvalues[did_coef_idx]

print(f"DiD coefficient: {did_coef:.4f}")
print(f"  Standard Error: {did_se:.4f}")
print(f"  p-value: {did_pval:.4f}")

# Confidence interval
conf_int = result.conf_int()[did_coef_idx]
print(f"  95% CI: [{conf_int[0]:.4f}, {conf_int[1]:.4f}]")

# Alternative specification with interaction only
print("\n--- Alternative DiD Specification (Interaction Only) ---")
X2 = np.column_stack([treatment_group, post_treatment, treatment, fvp_demo, fvp_gdppc200, fvp_population200, country_dummies])
X2 = sm.add_constant(X2)

model2 = sm.OLS(y, X2)
result2 = model2.fit(cov_type='cluster', cov_kwds={'groups': country_codes_num})

# Interaction term is at index 3 (after const, treatment_group, post_treatment)
interaction_coef = result2.params[3]
interaction_se = result2.bse[3]
interaction_pval = result2.pvalues[3]
interaction_ci = result2.conf_int()[3]

print(f"Interaction coefficient: {interaction_coef:.4f}")
print(f"  Standard Error: {interaction_se:.4f}")
print(f"  p-value: {interaction_pval:.4f}")
print(f"  95% CI: [{interaction_ci[0]:.4f}, {interaction_ci[1]:.4f}]")

# Event study analysis
print("\n" + "=" * 70)
print("EVENT STUDY ANALYSIS")
print("=" * 70)

# Create event study dataset
event_df = reg_df[reg_df['months_since_transition'].notna()].copy()
event_df = event_df.reset_index(drop=True)
event_df['event_time'] = event_df['months_since_transition'].astype(int)

# Create leads and lags
event_df['lead_1'] = (event_df['event_time'] == -1).astype(int)
event_df['lead_2'] = (event_df['event_time'] == -2).astype(int)
event_df['lead_3'] = (event_df['event_time'] == -3).astype(int)
event_df['lag_1'] = (event_df['event_time'] == 1).astype(int)
event_df['lag_2'] = (event_df['event_time'] == 2).astype(int)
event_df['lag_3'] = (event_df['event_time'] == 3).astype(int)

# Drop rows at t=0
event_df = event_df[event_df['event_time'] != 0]
event_df = event_df.reset_index(drop=True)

print(f"Event study sample size: {len(event_df)}")

# Build event study design matrix
event_y = event_df[dependent_var].to_numpy(dtype=float)
lead_1 = event_df['lead_1'].to_numpy(dtype=float)
lead_2 = event_df['lead_2'].to_numpy(dtype=float)
lead_3 = event_df['lead_3'].to_numpy(dtype=float)
lag_1 = event_df['lag_1'].to_numpy(dtype=float)
lag_2 = event_df['lag_2'].to_numpy(dtype=float)
lag_3 = event_df['lag_3'].to_numpy(dtype=float)

# Get country dummies for event study
event_country_codes = event_df['country_id'].astype('category')
event_country_codes_num = event_country_codes.cat.codes.to_numpy(dtype=int)
event_country_dummies = pd.get_dummies(event_df['country_id'], prefix='country').to_numpy(dtype=float)

X_event = np.column_stack([lead_1, lead_2, lead_3, lag_1, lag_2, lag_3,
                           fvp_demo[:len(event_y)],
                           fvp_gdppc200[:len(event_y)],
                           fvp_population200[:len(event_y)],
                           event_country_dummies])
X_event = sm.add_constant(X_event)

print(f"Event study design matrix shape: {X_event.shape}")

model_event = sm.OLS(event_y, X_event)
result_event = model_event.fit(cov_type='cluster', cov_kwds={'groups': event_country_codes_num})

# Print event study coefficients
print("\nEvent Study Coefficients:")
coef_names = ['lead_1', 'lead_2', 'lead_3', 'lag_1', 'lag_2', 'lag_3']
for i, name in enumerate(coef_names):
    coef = result_event.params[i + 1]  # +1 for constant
    se = result_event.bse[i + 1]
    pval = result_event.pvalues[i + 1]
    sig = '*' if pval < 0.05 else ('**' if pval < 0.10 else '')
    print(f"  {name}: {coef:.4f} (SE={se:.4f}, p={pval:.4f}){sig}")

# Calculate and save results
print("\n" + "=" * 70)
print("CALCULATING RESULTS")
print("=" * 70)

mean_y = df[dependent_var].mean()
elasticity = interaction_coef / mean_y * 100
print(f"\nDiD coefficient (treatment*post): {interaction_coef:.4f}")
print(f"  95% CI: [{interaction_ci[0]:.4f}, {interaction_ci[1]:.4f}]")
print(f"  p-value: {interaction_pval:.4f}")
print(f"  Elasticity: {elasticity:.4f}%")

# Prepare results for LaTeX
results = {
    'did_coefficient': float(interaction_coef),
    'did_std_err': float(interaction_se),
    'did_pvalue': float(interaction_pval),
    'did_ci_lower': float(interaction_ci[0]),
    'did_ci_upper': float(interaction_ci[1]),
    'n_obs': int(len(reg_df)),
    'n_countries': int(reg_df['country_id'].nunique()),
    'mean_violence': float(df[dependent_var].mean()),
    'transition_countries': int(len(first_transitions))
}

# Save results to CSV
results_df = pd.DataFrame({
    'Variable': ['DiD_Interaction', 'Treatment_Group', 'Post_Treatment', 'Control_1'],
    'Coefficient': [float(interaction_coef), float(result2.params[1]), float(result2.params[2]), float(result2.params[4])],
    'Std_Err': [float(interaction_se), float(result2.bse[1]), float(result2.bse[2]), float(result2.bse[4])],
    'P_value': [float(interaction_pval), float(result2.pvalues[1]), float(result2.pvalues[2]), float(result2.pvalues[4])]
})
results_df.to_csv('regression_results.csv', index=False)
print("\nResults saved to regression_results.csv")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
