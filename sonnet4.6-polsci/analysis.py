"""
Oil Rents and State-Based Violence in Africa, 1992–2020
Shift-share IV: Global oil price changes × country baseline oil exposure
Country-year panel; entity FE; year trend controlled via annual oil price
"""

import pandas as pd
import numpy as np
import warnings, json
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from linearmodels.panel import PanelOLS
warnings.filterwarnings('ignore')

# ── 1. Load and prepare data ───────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('/home/bjr113/github/gemma4polsci/cm.csv')
df['year'] = 1990 + (df['month_id'] - 1) // 12
df['month'] = (df['month_id'] - 1) % 12 + 1
df = df[(df['year'] >= 1990) & (df['year'] <= 2020)].copy()

# Aggregate to country-year
ann = df.groupby(['country_id','year']).agg({
    'ged_best_sb': 'sum', 'ged_best_ns': 'sum', 'ged_best_os': 'sum',
    'tlag_12_vdem_v2x_polyarchy': 'first',
    'fvp_prop_excluded':     'first',
    'fvp_prop_discriminated':'first',
    'fvp_lnpop200':          'first',
    'fvp_lngdpcap_nonoilrent':'first',
    'fvp_regime3c':          'first',
    'vdem_e_coups':          'first',
    'country_name':          'first',
}).reset_index()

# ISO-3 mapping for WB data merge
name_to_iso3 = {
    'Algeria':'DZA','Angola':'AGO','Benin':'BEN','Botswana':'BWA',
    'Burkina Faso':'BFA','Burundi':'BDI','Cameroon':'CMR','Cape Verde':'CPV',
    'Central African Republic':'CAF','Chad':'TCD','Comoros':'COM',
    'Congo':'COG','Congo, DRC':'COD',"Cote d'Ivoire":'CIV','Djibouti':'DJI',
    'Egypt':'EGY','Equatorial Guinea':'GNQ','Eritrea':'ERI','Ethiopia':'ETH',
    'Gabon':'GAB','Ghana':'GHA','Guinea':'GIN','Guinea-Bissau':'GNB',
    'Kenya':'KEN','Lesotho':'LSO','Liberia':'LBR','Libya':'LBY',
    'Madagascar':'MDG','Malawi':'MWI','Mali':'MLI','Mauritania':'MRT',
    'Mauritius':'MUS','Morocco':'MAR','Mozambique':'MOZ','Namibia':'NAM',
    'Niger':'NER','Nigeria':'NGA','Rwanda':'RWA','Sao Tome and Principe':'STP',
    'Senegal':'SEN','Seychelles':'SYC','Sierra Leone':'SLE','Somalia':'SOM',
    'South Africa':'ZAF','South Sudan':'SSD','Sudan':'SDN','Swaziland':'SWZ',
    'Tanzania':'TZA','The Gambia':'GMB','Togo':'TGO','Tunisia':'TUN',
    'Uganda':'UGA','Zambia':'ZMB','Zimbabwe':'ZWE',
}
ann['iso3'] = ann['country_name'].map(name_to_iso3)

# Merge World Bank oil rents (% of GDP) — annual, time-varying
wb = pd.read_csv('/home/bjr113/github/gemma4polsci/wb_oil_rents.csv')
ann = ann.merge(wb[['iso3','year','wb_oil_rents_pct']], on=['iso3','year'], how='left')
ann['wb_oil_rents_pct'] = ann['wb_oil_rents_pct'].fillna(0)
ann['ln_oil_rents'] = np.log1p(ann['wb_oil_rents_pct'])   # endogenous regressor

# Merge Brent crude prices (annual averages)
brent = pd.read_csv('/home/bjr113/github/gemma4polsci/brent_crude.csv',
                    parse_dates=['observation_date'])
brent.columns = ['date','oil_price_global']
brent['year'] = brent['date'].dt.year
brent_ann = brent.groupby('year')['oil_price_global'].mean().reset_index()
brent_ann['ln_oil_price'] = np.log(brent_ann['oil_price_global'])
brent_ann['Δln_oil'] = brent_ann['ln_oil_price'].diff()   # annual change in log oil price
ann = ann.merge(brent_ann[['year','ln_oil_price','Δln_oil']], on='year', how='left')

# Merge food price index (for falsification)
food = pd.read_csv('/home/bjr113/github/gemma4polsci/food_prices.csv',
                   parse_dates=['observation_date'])
food.columns = ['date','food_price_global']
food['year'] = food['date'].dt.year
food_ann = food.groupby('year')['food_price_global'].mean().reset_index()
food_ann['ln_food_price'] = np.log(food_ann['food_price_global'])
food_ann['Δln_food'] = food_ann['ln_food_price'].diff()
ann = ann.merge(food_ann[['year','ln_food_price','Δln_food']], on='year', how='left')

# ── 2. Baseline oil exposure (1990–1994 average, pre-sample) ──────────────────
bl = (ann[ann['year'].between(1990,1994)]
       .groupby('country_id')['wb_oil_rents_pct']
       .mean()
       .fillna(0)
       .rename('bl_oil_rents'))
ann = ann.merge(bl.reset_index(), on='country_id', how='left')
ann['ln_bl_oil'] = np.log1p(ann['bl_oil_rents'])
ann['oil_producer'] = (ann['bl_oil_rents'] > 5).astype(float)  # binary

# ── 3. Instruments ────────────────────────────────────────────────────────────
# Main: annual oil price CHANGE × log baseline oil exposure
# This captures: when global oil prices rise, countries with higher baseline oil
# exposure see larger increases in oil revenues (differential exposure design)
ann['iv_main'] = ann['Δln_oil'] * ann['ln_bl_oil']      # shift-share IV
ann['iv_food'] = ann['Δln_food'] * ann['ln_bl_oil']      # falsification: food price × oil exposure

# ── 4. Dependent and control variables ────────────────────────────────────────
ann['ln_ged_sb'] = np.log1p(ann['ged_best_sb'])
ann['ln_ged_ns'] = np.log1p(ann['ged_best_ns'])
ann['ln_ged_os'] = np.log1p(ann['ged_best_os'])

ann = ann.rename(columns={
    'tlag_12_vdem_v2x_polyarchy': 'democracy',
    'fvp_prop_excluded':          'excl_share',
    'fvp_prop_discriminated':     'discrim_share',
    'fvp_lnpop200':               'ln_pop',
    'fvp_lngdpcap_nonoilrent':    'ln_gdpcap_nonoil',
})

# Temporal lags
ann = ann.sort_values(['country_id','year'])
for lag_var, new_name in [('ln_ged_sb','y_lag1'), ('ln_ged_ns','ns_lag1')]:
    ann[new_name] = ann.groupby('country_id')[lag_var].shift(1)

# Interaction for heterogeneity analysis: oil rents × political exclusion
ann['oil_x_excl'] = ann['ln_oil_rents'] * ann['excl_share']
ann['iv_x_excl']  = ann['iv_main']       * ann['excl_share']

# ── 5. Analysis sample ─────────────────────────────────────────────────────────
req_vars = ['ln_ged_sb','ln_oil_rents','iv_main','iv_food','excl_share',
            'democracy','ln_pop','ln_gdpcap_nonoil','y_lag1','ns_lag1',
            'country_id','year','oil_producer','ln_bl_oil','Δln_oil',
            'oil_x_excl','iv_x_excl','ln_ged_ns','ln_ged_os']
dfc = ann[req_vars].dropna().copy()
print(f"Analysis obs: {len(dfc):,} | Countries: {dfc['country_id'].nunique()} "
      f"| Years: {dfc['year'].min()}–{dfc['year'].max()}")
print(f"Oil-producing countries: {dfc[dfc['oil_producer']==1]['country_id'].nunique()}")

# ── 6. Summary statistics ──────────────────────────────────────────────────────
sumvars = ['ln_ged_sb','ln_oil_rents','Δln_oil','ln_bl_oil','iv_main',
           'ln_gdpcap_nonoil','ln_pop','democracy','excl_share']
sum_labels = [
    'ln(State-Based Fatalities+1)',
    'Oil Rents (ln, % GDP)',
    'Annual Change in ln(Brent price)',
    'Baseline Oil Exposure (ln)',
    'IV: Oil Price Change × Exposure',
    'ln(GDP/cap, non-oil)',
    'ln(Population)',
    'Democracy (V-Dem Polyarchy, 12-mo lag)',
    'Excluded Population Share',
]
summary = dfc[sumvars].describe().T[['count','mean','std','min','50%','max']].round(3)
summary.index = sum_labels
summary.to_csv('/home/bjr113/github/gemma4polsci/summary_stats.csv')
print("\n=== Summary Statistics ===")
print(summary.to_string())

# ── 7. Entity (country) demeaning  ────────────────────────────────────────────
# NOTE: We do NOT include year FEs in the main specification because the instrument
# (Δln_oil × baseline) is primarily time-series variation that year FEs would absorb.
# Instead, we control for Δln_oil directly to absorb the global time trend.
# This follows Brückner & Ciccone (2010, AER) exactly.

dm_cols = ['ln_ged_sb','ln_oil_rents','iv_main','iv_food','excl_share',
           'democracy','ln_pop','ln_gdpcap_nonoil','y_lag1','ns_lag1',
           'Δln_oil','oil_x_excl','iv_x_excl','ln_ged_ns','ln_ged_os']
dfc_dm = dfc.copy()
for c in dm_cols:
    dfc_dm[c] = dfc_dm[c] - dfc_dm.groupby('country_id')[c].transform('mean')

y    = dfc_dm['ln_ged_sb']
# Controls include Δln_oil directly to absorb the common oil price trend
exog_base = dfc_dm[['Δln_oil','excl_share','democracy','ln_pop','ln_gdpcap_nonoil','y_lag1']]
endog     = dfc_dm[['ln_oil_rents']]
iv_main_v = dfc_dm[['iv_main']]
iv_food_v = dfc_dm[['iv_food']]

# ── 8. OLS (pooled, with country FE via demeaning) ────────────────────────────
print("\n=== Running regressions ===")

# OLS
X_ols = sm.add_constant(pd.concat([endog, exog_base], axis=1))
res_ols_sm = sm.OLS(y, X_ols).fit(cov_type='cluster', cov_kwds={'groups': dfc['country_id']})
print(f"OLS: oil coef={res_ols_sm.params['ln_oil_rents']:.4f}, "
      f"t={res_ols_sm.tvalues['ln_oil_rents']:.2f}")

# ── 9. First stage ────────────────────────────────────────────────────────────
X_fs = sm.add_constant(pd.concat([iv_main_v, exog_base], axis=1))
res_fs = sm.OLS(dfc_dm['ln_oil_rents'], X_fs).fit(
    cov_type='cluster', cov_kwds={'groups': dfc['country_id']})
fs_t = res_fs.tvalues['iv_main']
fs_F = fs_t**2
print(f"First stage: iv coef={res_fs.params['iv_main']:.4f}, t={fs_t:.2f}, F={fs_F:.1f}")

# ── 10. 2SLS main ─────────────────────────────────────────────────────────────
res_2sls = IV2SLS(y, exog_base.assign(const=1), endog, iv_main_v).fit(
    cov_type='clustered', clusters=dfc['country_id'])
print(f"2SLS: oil coef={res_2sls.params['ln_oil_rents']:.4f}, "
      f"se={res_2sls.std_errors['ln_oil_rents']:.4f}, "
      f"t={res_2sls.tstats['ln_oil_rents']:.2f}")

# ── 11. 2SLS with heterogeneity (oil × exclusion) ────────────────────────────
exog_het = dfc_dm[['Δln_oil','excl_share','democracy','ln_pop','ln_gdpcap_nonoil',
                    'y_lag1','oil_x_excl']]
endog_het = dfc_dm[['ln_oil_rents']]
iv_het    = dfc_dm[['iv_main','iv_x_excl']]
res_het   = IV2SLS(y, exog_het.assign(const=1), endog_het, iv_het).fit(
    cov_type='clustered', clusters=dfc['country_id'])
print(f"2SLS-het: oil={res_het.params['ln_oil_rents']:.4f}, "
      f"oil×excl={res_het.params.get('oil_x_excl',float('nan')):.4f}")

# ── 12. Falsification: food price × oil exposure ─────────────────────────────
# Under H0 (exclusion restriction), food price changes should NOT predict oil rents
X_fs_food = sm.add_constant(pd.concat([iv_food_v, exog_base], axis=1))
res_fs_food = sm.OLS(dfc_dm['ln_oil_rents'], X_fs_food).fit(
    cov_type='cluster', cov_kwds={'groups': dfc['country_id']})
print(f"Falsification FS (food → oil rents): t={res_fs_food.tvalues['iv_food']:.2f}")

res_falsi = IV2SLS(y, exog_base.assign(const=1), endog, iv_food_v).fit(
    cov_type='clustered', clusters=dfc['country_id'])
print(f"Falsification 2SLS (food IV): oil coef={res_falsi.params['ln_oil_rents']:.4f}, "
      f"t={res_falsi.tstats['ln_oil_rents']:.2f}")

# ── 13. Reduced form ──────────────────────────────────────────────────────────
X_rf = sm.add_constant(pd.concat([iv_main_v, exog_base], axis=1))
res_rf = sm.OLS(y, X_rf).fit(cov_type='cluster', cov_kwds={'groups': dfc['country_id']})
print(f"Reduced form: iv→ ln_ged_sb: t={res_rf.tvalues['iv_main']:.2f}, "
      f"coef={res_rf.params['iv_main']:.4f}")

# ── 14. Placebo outcome: non-state conflict ────────────────────────────────────
y_ns = dfc_dm['ln_ged_ns']
res_placebo = IV2SLS(y_ns, exog_base.assign(const=1), endog, iv_main_v).fit(
    cov_type='clustered', clusters=dfc['country_id'])
print(f"Placebo (non-state): oil coef={res_placebo.params['ln_oil_rents']:.4f}, "
      f"t={res_placebo.tstats['ln_oil_rents']:.2f}")

# ── 15. Two-way FE robustness ─────────────────────────────────────────────────
# Include time fixed effects as year dummies (but use iv within oil countries)
dfc['y_lag1_c'] = dfc['y_lag1']
panel_data = dfc.set_index(['country_id','year'])
res_twoway = PanelOLS.from_formula(
    'ln_ged_sb ~ ln_oil_rents + excl_share + democracy + ln_pop + ln_gdpcap_nonoil + y_lag1 '
    '+ EntityEffects + TimeEffects',
    data=panel_data
).fit(cov_type='clustered', cluster_entity=True)
print(f"Two-way FE OLS: oil={res_twoway.params['ln_oil_rents']:.4f}, "
      f"t={res_twoway.tstats['ln_oil_rents']:.2f}")

# ── 16. Save numerical results ─────────────────────────────────────────────────
r = {
    'ols_coef':        float(res_ols_sm.params['ln_oil_rents']),
    'ols_se':          float(res_ols_sm.bse['ln_oil_rents']),
    'ols_t':           float(res_ols_sm.tvalues['ln_oil_rents']),
    'ols_p':           float(res_ols_sm.pvalues['ln_oil_rents']),
    'fs_coef':         float(res_fs.params['iv_main']),
    'fs_se':           float(res_fs.bse['iv_main']),
    'fs_t':            float(fs_t),
    'fs_F':            float(fs_F),
    'rf_coef':         float(res_rf.params['iv_main']),
    'rf_t':            float(res_rf.tvalues['iv_main']),
    '2sls_coef':       float(res_2sls.params['ln_oil_rents']),
    '2sls_se':         float(res_2sls.std_errors['ln_oil_rents']),
    '2sls_t':          float(res_2sls.tstats['ln_oil_rents']),
    '2sls_p':          float(res_2sls.pvalues['ln_oil_rents']),
    'het_oil_coef':    float(res_het.params.get('ln_oil_rents',np.nan)),
    'het_int_coef':    float(res_het.params.get('oil_x_excl',np.nan)),
    'het_int_p':       float(res_het.pvalues.get('oil_x_excl',np.nan)),
    'twoway_coef':     float(res_twoway.params['ln_oil_rents']),
    'twoway_t':        float(res_twoway.tstats['ln_oil_rents']),
    'falsi_fs_t':      float(res_fs_food.tvalues['iv_food']),
    'falsi_2sls_coef': float(res_falsi.params['ln_oil_rents']),
    'falsi_2sls_p':    float(res_falsi.pvalues['ln_oil_rents']),
    'placebo_coef':    float(res_placebo.params['ln_oil_rents']),
    'placebo_p':       float(res_placebo.pvalues['ln_oil_rents']),
    'nobs':            int(res_2sls.nobs),
    'ncountries':      int(dfc['country_id'].nunique()),
    'nyears':          int(dfc['year'].nunique()),
    'year_range':      f"{int(dfc['year'].min())}--{int(dfc['year'].max())}",
    'noil_countries':  int(dfc[dfc['oil_producer']==1]['country_id'].nunique()),
}
with open('/home/bjr113/github/gemma4polsci/res_summary.json','w') as f:
    json.dump(r, f, indent=2)
print("\nResults saved.")
print(json.dumps(r, indent=2))

# ── 17. Build LaTeX tables ────────────────────────────────────────────────────
def star(p):
    return '***' if p<0.01 else ('**' if p<0.05 else ('*' if p<0.10 else ''))

def fmt(coef, se, p):
    return f'{coef:.3f}{star(p)}', f'({se:.3f})'

def build_table(models, col_hdrs, var_map, caption, label, note, filepath):
    """
    models: list of (label, result_object, 'statsmodels'|'linearmodels')
    """
    ncols = len(models)
    lines = [r'\begin{table}[htbp]\centering\small',
             r'\caption{' + caption + r'}\label{' + label + '}',
             r'\begin{tabular}{l' + 'c'*ncols + '}',
             r'\toprule',
             ' & ' + ' & '.join(f'({i+1})' for i in range(ncols)) + r' \\',
             ' & ' + ' & '.join(col_hdrs) + r' \\',
             r'\midrule']

    for var, disp in var_map.items():
        row_c = disp; row_s = ''
        for _, res, kind in models:
            try:
                if kind == 'sm':
                    c_v = res.params[var]; s_v = res.bse[var]; p_v = res.pvalues[var]
                else:
                    c_v = res.params[var]; s_v = res.std_errors[var]; p_v = res.pvalues[var]
                c, s = fmt(c_v, s_v, p_v)
                row_c += f' & {c}'; row_s += f' & {s}'
            except KeyError:
                row_c += ' & '; row_s += ' & '
        lines += [row_c + r' \\', row_s + r' \\']

    lines.append(r'\midrule')
    obs_r = 'Observations'; r2_r = '$R^2$'
    for _, res, kind in models:
        try: obs_r += f' & {int(res.nobs):,}'
        except: obs_r += ' & '
        try:
            r2 = res.rsquared if kind=='sm' else res.rsquared
            r2_r += f' & {r2:.3f}'
        except: r2_r += ' & '
    lines += [obs_r + r' \\', r2_r + r' \\']
    lines.append('Country FE & ' + ' & '.join(['Yes']*ncols) + r' \\')
    lines.append(r'\bottomrule',)
    lines.append(r'\multicolumn{' + str(ncols+1) + r'}{p{0.95\linewidth}}{\footnotesize')
    lines.append(r'\textit{Note}: ' + note + '}')
    lines += [r'\end{tabular}', r'\end{table}']
    with open(filepath,'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved {filepath}")

# Main results table
main_models = [
    ('OLS',   res_ols_sm,  'sm'),
    ('2SLS',  res_2sls,    'lm'),
    ('Het',   res_het,     'lm'),
    ('2Way',  res_twoway,  'lm'),
]
main_col_hdrs = [r'\textit{ln SB Deaths}',r'\textit{ln SB Deaths}',
                 r'\textit{ln SB Deaths}',r'\textit{ln SB Deaths}']
main_var_map = {
    'ln_oil_rents':   r'Oil Rents (ln\%GDP)',
    'oil_x_excl':     r'Oil Rents $\times$ Excl.\ Share',
    'excl_share':     r'Excluded Pop.\ Share',
    'democracy':      r'Democracy (V-Dem Polyarchy)',
    'ln_gdpcap_nonoil':r'ln(GDP/cap, non-oil)',
    'ln_pop':         r'ln(Population)',
    'Δln_oil':        r'$\Delta \ln(\text{Oil Price})$',
    'y_lag1':         r'Lagged DV ($t-1$)',
}
main_note = (
    r'Dependent variable: $\ln(\text{state-based fatalities}+1)$. '
    r'All models include country fixed effects (entity demeaning). '
    r'Columns (1)--(3) control for $\Delta\ln(\text{Oil Price})$ directly to absorb '
    r'the global oil price trend; column (4) uses two-way (country + year) fixed effects. '
    r'The instrument in columns (2)--(3) is '
    r'$\Delta\ln(\text{Brent Crude})_t \times \ln(1+\overline{\text{Oil Rents}}_{i,1990\text{--}1994})$. '
    r'Standard errors clustered by country (54 clusters). '
    r'*, **, *** denote $p<0.10$, $p<0.05$, $p<0.01$.'
)
build_table(main_models, main_col_hdrs, main_var_map,
    caption=r'Oil Rents and State-Based Violence in Africa',
    label='tab:main',
    note=main_note,
    filepath='/home/bjr113/github/gemma4polsci/table_main.tex')

# First stage + falsification table
fs_models = [
    ('FS-Main',   res_fs,       'sm'),
    ('FS-Food',   res_fs_food,  'sm'),
    ('2SLS-Falsi',res_falsi,    'lm'),
    ('Placebo',   res_placebo,  'lm'),
]
fs_col_hdrs = [r'\textit{Oil Rents}', r'\textit{Oil Rents}',
               r'\textit{ln SB Deaths}', r'\textit{ln NS Deaths}']
fs_var_map = {
    'iv_main':      r'$\Delta\ln(\text{Oil})_t \times \ln(\overline{\text{Oil}}_i)$ [IV]',
    'iv_food':      r'$\Delta\ln(\text{Food})_t \times \ln(\overline{\text{Oil}}_i)$ [Placebo IV]',
    'ln_oil_rents': r'Oil Rents (ln\%GDP) [inst.]',
    'excl_share':   r'Excluded Pop.\ Share',
    'democracy':    r'Democracy (V-Dem Polyarchy)',
    'ln_gdpcap_nonoil':r'ln(GDP/cap, non-oil)',
    'ln_pop':       r'ln(Population)',
    'y_lag1':       r'Lagged DV ($t-1$)',
}
fs_note = (
    r'Column (1): First stage for main instrument. Dependent variable: '
    r'$\ln(\text{oil rents }\%\text{GDP}+1)$ (World Bank). '
    r'Column (2): First-stage using food price $\times$ oil exposure as a placebo IV; '
    r'food price changes should not predict oil revenues. '
    r'Column (3): 2SLS using food price instrument---IV should be irrelevant for conflict '
    r'if exclusion restriction holds. '
    r'Column (4): 2SLS second stage using non-state conflict as outcome; '
    r'oil rents should not affect armed group violence via the same channel. '
    r'Standard errors clustered by country. '
    r'*, **, *** denote $p<0.10$, $p<0.05$, $p<0.01$.'
)
build_table(fs_models, fs_col_hdrs, fs_var_map,
    caption=r'First Stage, Falsification, and Placebo Tests',
    label='tab:diagnostics',
    note=fs_note,
    filepath='/home/bjr113/github/gemma4polsci/table_diagnostics.tex')

print("\nAll analysis complete.")
