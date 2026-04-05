"""
Extended EDA: Country-specific temperature anomalies and alternative measures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')

# Load data
df = pd.read_csv('cm_with_temp.csv')

print("=" * 60)
print("EXTENDED EDA: Country-Specific Analysis")
print("=" * 60)

# Create country-specific temperature anomaly (deviation from country's own mean)
print("\n1. CREATING COUNTRY-SPECIFIC TEMPERATURE ANOMALIES")

# The global anomaly is the same for all countries in a given month
# We need to create country-specific deviations

# First, calculate the global mean for each month
global_monthly_mean = df.groupby(['year', 'month'])['anomaly'].first().reset_index()
global_monthly_mean.columns = ['year', 'month', 'global_anomaly']

# Merge back
df = df.merge(global_monthly_mean, on=['year', 'month'], how='left')

# For country-specific analysis, we need sub-national temperature data
# Since we only have global data, let's use the global anomaly but
# interact with country characteristics

# Create lagged temperature variables
print("\n2. CREATING LAGGED TEMPERATURE VARIABLES")
for lag in [1, 3, 6, 12]:
    df[f'anomaly_lag{lag}'] = df.groupby('country_id')['anomaly'].shift(lag)

# Create rolling averages
df['anomaly_rolling_12'] = df.groupby('country_id')['anomaly'].transform(
    lambda x: x.rolling(window=12, min_periods=1).mean()
)

# Create extreme heat indicator (above 90th percentile)
heat_threshold = df['anomaly'].quantile(0.90)
df['extreme_heat'] = (df['anomaly'] > heat_threshold).astype(int)
df['extreme_cold'] = (df['anomaly'] < df['anomaly'].quantile(0.10)).astype(int)

print(f"   Heat threshold (90th pct): {heat_threshold:.3f}°C")
print(f"   Cold threshold (10th pct): {df['anomaly'].quantile(0.10):.3f}°C")

# Check correlation with lags
print("\n3. CORRELATION WITH LAGGED TEMPERATURE")
for lag in [0, 1, 3, 6, 12]:
    var_name = 'anomaly' if lag == 0 else f'anomaly_lag{lag}'
    corr = df[var_name].corr(df['ged_best_sb'])
    print(f"   Lag {lag}: {corr:.4f}")

# Check correlation with extreme heat
print("\n4. CORRELATION WITH EXTREME WEATHER INDICATORS")
print(f"   Extreme heat: {df['extreme_heat'].corr(df['ged_best_sb']):.4f}")
print(f"   Extreme cold: {df['extreme_cold'].corr(df['ged_best_sb']):.4f}")

# Violence by extreme heat indicator
print("\n5. VIOLENCE BY EXTREME WEATHER")
viol_by_heat = df.groupby('extreme_heat')['ged_best_sb'].agg(['mean', 'sum', 'count'])
print(viol_by_heat)

# Check other potential predictors in the data
print("\n6. EXPLORING OTHER POTENTIAL PREDICTORS")

# List of potentially interesting variables
potential_predictors = [
    'fvp_democracy', 'fvp_liberal', 'fvp_gdpcap_nonoilrent', 'fvp_population200',
    'fvp_timesinceregimechange', 'fvp_ssp2_urban_share_iiasa',
    'icgcw_alerts', 'icgcw_deteriorated', 'icgcw_improved'
]

print("\n   Correlation with ged_best_sb:")
for var in potential_predictors:
    if var in df.columns:
        corr = df[var].corr(df['ged_best_sb'])
        if not np.isnan(corr):
            print(f"   {var}: {corr:.4f}")

# Check ICGCW variables (International Crisis Group)
print("\n7. ICGCW VARIABLES")
for var in ['icgcw_alerts', 'icgcw_deteriorated', 'icgcw_improved']:
    if var in df.columns:
        print(f"\n   {var}:")
        print(f"      Unique values: {df[var].nunique()}")
        print(f"      Non-null: {df[var].notna().sum()}")
        if df[var].notna().sum() > 0:
            print(f"      Correlation with violence: {df[var].corr(df['ged_best_sb']):.4f}")

# Create a violence intensity measure
print("\n8. CREATING VIOLENCE INTENSITY MEASURES")
df['viol_log'] = np.log1p(df['ged_best_sb'])
df['viol_any'] = (df['ged_best_sb'] > 0).astype(int)
df['viol_high'] = (df['ged_best_sb'] > df['ged_best_sb'].quantile(0.90)).astype(int)

print(f"   Log violence mean: {df['viol_log'].mean():.3f}")
print(f"   Any violence pct: {df['viol_any'].mean()*100:.1f}%")
print(f"   High violence pct: {df['viol_high'].mean()*100:.1f}%")

# Generate updated figures
print("\n9. GENERATING UPDATED FIGURES...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Extreme heat vs violence
ax = axes[0, 0]
sns.boxplot(data=df, x='extreme_heat', y='viol_log', ax=ax, palette='Set2')
ax.set_xlabel('Extreme Heat Month')
ax.set_ylabel('Log(1 + Violence Deaths)')
ax.set_title('Violence by Extreme Heat Indicator')
ax.set_xticklabels(['Normal', 'Extreme Heat'])

# Panel B: Temperature trend with violence overlay
ax = axes[0, 1]
temp_trend = df.groupby('year')['anomaly'].mean()
viol_trend = df.groupby('year')['ged_best_sb'].sum()
ax.plot(temp_trend.index, temp_trend.values, 'b-', label='Temp Anomaly', linewidth=2)
ax.set_xlabel('Year')
ax.set_ylabel('Temperature Anomaly (°C)')
ax.set_title('Temperature Trend 1980-2026')
ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.plot(viol_trend.index, viol_trend.values, 'r-', alpha=0.5, label='Violence')
ax2.set_ylabel('Total Violence Deaths')
ax2.set_yscale('log')

# Panel C: Distribution of violence (log scale)
ax = axes[1, 0]
df[df['ged_best_sb'] > 0]['viol_log'].hist(bins=50, ax=ax, edgecolor='black', alpha=0.7)
ax.set_xlabel('Log(1 + Violence Deaths)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Violence (Positive Values, Log Scale)')

# Panel D: Violence by democracy level
ax = axes[1, 1]
if 'fvp_democracy' in df.columns and df['fvp_democracy'].nunique() > 3:
    df['democracy_cat'] = pd.qcut(df['fvp_democracy'].dropna(), q=4, labels=['Lowest', 'Low', 'High', 'Highest'])
    sns.boxplot(data=df.dropna(subset=['democracy_cat']), x='democracy_cat', y='viol_log', ax=ax)
    ax.set_xlabel('Democracy Level')
    ax.set_ylabel('Log(1 + Violence Deaths)')
    ax.set_title('Violence by Democracy Level')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('extended_eda.png', dpi=300, bbox_inches='tight')
print("   Saved: extended_eda.png")

print("\n" + "=" * 60)
print("EXTENDED EDA COMPLETE")
print("=" * 60)
