"""
Analysis: Anti-Japanese Sentiment and Demon Slayer Box Office in China
Difference-in-Differences design using anti-Japanese memorial sites as treatment.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster
import warnings
warnings.filterwarnings('ignore')

WORK_DIR = '/home/bjr113/github/the-ai-political-scientist/adviser-muh/'

# ── 1. Load data ──────────────────────────────────────────────────────────────
kny = pd.read_csv(WORK_DIR + 'kny_citylevel_1113_1211.csv')
ztp = pd.read_excel(WORK_DIR + 'ztp_citylevel_1126_1226.xlsx')
anti = pd.read_csv(WORK_DIR + 'anti_japanese_site.csv')

# Standardize ZTP column names to match KNY
ztp.columns = ['date', 'province_CN', 'province_EN', 'city_CN', 'city_EN',
               'daily_box_office_yuan', 'daily_box_office_wan', 'box_office_share',
               'screening_share', 'avg_attendance_per_screening', 'cumulative_box_office',
               'seat_occupancy_rate', 'prime_time_screening_share', 'audience_count',
               'screening_count']

# ── 2. Build city-level anti-Japanese memorial count ─────────────────────────
anti_count = anti.groupby('所在城市').size().reset_index(name='n_anti_sites')
anti_count.columns = ['city_CN', 'n_anti_sites']

print("Anti-Japanese memorial site distribution:")
print(anti_count['n_anti_sites'].describe())
print(f"Cities with at least 1 site: {(anti_count['n_anti_sites'] > 0).sum()} / {len(anti_count)}")

# ── 3. Restrict to overlapping date window: Nov 26 – Dec 11, 2025 ────────────
overlap_start = '2025-11-26'
overlap_end   = '2025-12-11'

kny['date'] = pd.to_datetime(kny['date'])
ztp['date'] = pd.to_datetime(ztp['date'])

kny_ov = kny[(kny['date'] >= overlap_start) & (kny['date'] <= overlap_end)].copy()
ztp_ov = ztp[(ztp['date'] >= overlap_start) & (ztp['date'] <= overlap_end)].copy()

print(f"\nKNY overlap obs: {len(kny_ov)}, ZTP overlap obs: {len(ztp_ov)}")

# ── 4. Stack into one panel ───────────────────────────────────────────────────
kny_ov['is_kny'] = 1
ztp_ov['is_kny'] = 0

panel = pd.concat([kny_ov, ztp_ov], ignore_index=True)

# ── 5. Merge anti-Japanese site counts ───────────────────────────────────────
panel = panel.merge(anti_count, on='city_CN', how='left')
panel['n_anti_sites'] = panel['n_anti_sites'].fillna(0)

# Restrict to cities that appear in BOTH films (balanced panel)
kny_cities = set(kny_ov['city_CN'].unique())
ztp_cities = set(ztp_ov['city_CN'].unique())
common_cities = kny_cities & ztp_cities
panel = panel[panel['city_CN'].isin(common_cities)].copy()
print(f"Common cities: {len(common_cities)}")
print(f"Panel obs after restricting to common cities: {len(panel)}")

# ── 6. Construct regression variables ────────────────────────────────────────
panel['log_audience'] = np.log(panel['audience_count'] + 1)
panel['log_screening'] = np.log(panel['screening_count'] + 1)

# Binary treatment: any anti-Japanese site in city
panel['has_anti_site'] = (panel['n_anti_sites'] > 0).astype(int)

# Standardize continuous treatment
panel['anti_sites_std'] = (panel['n_anti_sites'] - panel['n_anti_sites'].mean()) / panel['n_anti_sites'].std()

# Date and city fixed effects as categorical
panel['city_fe'] = panel['city_CN'].astype('category')
panel['date_fe'] = panel['date'].astype(str).astype('category')

# Interaction terms
panel['anti_x_kny'] = panel['n_anti_sites'] * panel['is_kny']
panel['anti_std_x_kny'] = panel['anti_sites_std'] * panel['is_kny']
panel['has_anti_x_kny'] = panel['has_anti_site'] * panel['is_kny']

print("\nTreatment summary:")
print(panel[panel['is_kny']==1].groupby('has_anti_site')['city_CN'].nunique().rename('n_cities'))

# ── 7. Regressions ────────────────────────────────────────────────────────────

def run_ols_clustered(formula, data, cluster_var):
    """OLS with cluster-robust SEs."""
    model = smf.ols(formula, data=data).fit()
    cluster_groups = data[cluster_var].values
    model_robust = model.get_robustcov_results(cov_type='cluster',
                                               groups=cluster_groups)
    return model_robust

print("\n" + "="*70)
print("REGRESSIONS")
print("="*70)

# Model 1: Basic DID — no fixed effects
m1 = run_ols_clustered(
    'log_audience ~ anti_x_kny + n_anti_sites + is_kny',
    panel, 'city_CN'
)
print("\nModel 1: Basic DID (no FE)")
print(m1.summary2().tables[1][['Coef.', 'Std.Err.', 't', 'P>|t|']])

# Model 2: City + Date FE
m2 = run_ols_clustered(
    'log_audience ~ anti_x_kny + C(city_CN) + C(date_fe)',
    panel, 'city_CN'
)
def print_coefs(model, keywords):
    """Print rows from summary table matching any keyword."""
    tbl = model.summary2().tables[1]
    # Ensure index is a proper string Index
    tbl.index = pd.Index([str(i) for i in tbl.index])
    mask = tbl.index.str.contains('|'.join(keywords))
    print(tbl.loc[mask, ['Coef.', 'Std.Err.', 't', 'P>|t|']])

print("\nModel 2: City + Date FE (with is_kny main effect)")
# is_kny must be included as main effect; n_anti_sites absorbed by city FE
m2 = run_ols_clustered(
    'log_audience ~ anti_x_kny + is_kny + C(city_CN) + C(date_fe)',
    panel, 'city_CN'
)
print_coefs(m2, ['anti_x_kny', 'is_kny'])

# Model 3: City + Date FE + screening_count control (log)
m3 = run_ols_clustered(
    'log_audience ~ anti_x_kny + is_kny + log_screening + C(city_CN) + C(date_fe)',
    panel, 'city_CN'
)
print("\nModel 3: City + Date FE + log(screenings)")
print_coefs(m3, ['anti_x_kny', 'is_kny', 'log_screening'])

# Model 4: Binary treatment
m4 = run_ols_clustered(
    'log_audience ~ has_anti_x_kny + is_kny + C(city_CN) + C(date_fe)',
    panel, 'city_CN'
)
print("\nModel 4: Binary treatment (any anti-Japanese site)")
print_coefs(m4, ['has_anti_x_kny', 'is_kny'])

# Model 5: Standardized continuous treatment + controls
m5 = run_ols_clustered(
    'log_audience ~ anti_std_x_kny + is_kny + log_screening + C(city_CN) + C(date_fe)',
    panel, 'city_CN'
)
print("\nModel 5: Standardized treatment + log(screenings)")
print_coefs(m5, ['anti_std_x_kny', 'is_kny', 'log_screening'])

# Model 6: DV = log(avg_attendance_per_screening) — avoids endogenous screening control
# but captures demand-side conditional on supply
panel['log_avg_att'] = np.log(panel['avg_attendance_per_screening'] + 1)
m6 = run_ols_clustered(
    'log_avg_att ~ anti_x_kny + is_kny + C(city_CN) + C(date_fe)',
    panel, 'city_CN'
)
print("\nModel 6: DV = log(avg attendance per screening)")
print_coefs(m6, ['anti_x_kny', 'is_kny'])

# Model 7: DV = seat_occupancy_rate (convert % string to float)
panel['seat_occ_num'] = pd.to_numeric(
    panel['seat_occupancy_rate'].astype(str).str.replace('%','', regex=False),
    errors='coerce'
)
m7 = run_ols_clustered(
    'seat_occ_num ~ anti_x_kny + is_kny + C(city_CN) + C(date_fe)',
    panel, 'city_CN'
)
print("\nModel 7: DV = seat occupancy rate (%)")
print_coefs(m7, ['anti_x_kny', 'is_kny'])

# ── 8. Save key results ───────────────────────────────────────────────────────
def extract_result(model, coef_name, model_name):
    # params may have numpy array or pandas Index
    param_names = list(model.model.exog_names)
    idx = param_names.index(coef_name)
    coef = float(model.params[idx])
    se   = float(model.bse[idx])
    pval = float(model.pvalues[idx])
    n    = int(model.nobs)
    return {'Model': model_name, 'Coefficient': round(coef, 4),
            'SE': round(se, 4), 'P-value': round(pval, 4), 'N': n}

results = pd.DataFrame([
    extract_result(m1, 'anti_x_kny',     'No FE'),
    extract_result(m2, 'anti_x_kny',     'City + Date FE'),
    extract_result(m3, 'anti_x_kny',     'City + Date FE + Screenings'),
    extract_result(m4, 'has_anti_x_kny', 'Binary Treatment'),
    extract_result(m5, 'anti_std_x_kny', 'Standardized Treatment'),
    extract_result(m6, 'anti_x_kny',     'Avg Attendance per Screening'),
    extract_result(m7, 'anti_x_kny',     'Seat Occupancy Rate'),
])

results.to_csv(WORK_DIR + 'did_results.csv', index=False)
print("\n\nFinal DID results saved:")
print(results.to_string(index=False))

# ── 9. Summary statistics for the paper ──────────────────────────────────────
print("\n\nSummary statistics:")
sumstats = panel[['audience_count', 'log_audience', 'screening_count',
                  'n_anti_sites', 'has_anti_site', 'is_kny']].describe().round(2)
print(sumstats)
sumstats.to_csv(WORK_DIR + 'summary_stats.csv')

# Save key scalars for LaTeX
scalars = {
    'n_obs_panel':      len(panel),
    'n_cities':         panel['city_CN'].nunique(),
    'n_days':           panel['date'].nunique(),
    'mean_anti_kny':    round(panel[panel['is_kny']==1]['n_anti_sites'].mean(), 2),
    'pct_treated':      round(panel['has_anti_site'].mean() * 100, 1),
    'coef_main':        results.loc[results['Model']=='City + Date FE + Screenings','Coefficient'].iloc[0],
    'se_main':          results.loc[results['Model']=='City + Date FE + Screenings','SE'].iloc[0],
    'pval_main':        results.loc[results['Model']=='City + Date FE + Screenings','P-value'].iloc[0],
}
pd.Series(scalars).to_csv(WORK_DIR + 'scalars.csv', header=False)
print("\nScalars:", scalars)

print("\nDone.")
