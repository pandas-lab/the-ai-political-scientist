"""
Exploratory Data Analysis for Climate-Violence Research
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# Load data
df = pd.read_csv('cm_with_temp.csv')

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Basic data summary
print("\n1. DATA SUMMARY")
print(f"   Observations: {len(df):,}")
print(f"   Variables: {len(df.columns)}")

# Check for missing values
print("\n2. MISSING VALUES")
missing = pd.DataFrame({
    'Variable': ['ged_best_sb', 'anomaly', 'year', 'month'],
    'Missing': [df['ged_best_sb'].isna().sum(),
                df['anomaly'].isna().sum(),
                df['year'].isna().sum(),
                df['month'].isna().sum()],
    'Pct Missing': [df['ged_best_sb'].isna().mean()*100,
                    df['anomaly'].isna().mean()*100,
                    df['year'].isna().mean()*100,
                    df['month'].isna().mean()*100]
})
print(missing.to_string(index=False))

# DV Summary
print("\n3. DEPENDENT VARIABLE: ged_best_sb (State-Based Violence Deaths)")
dv_summary = df['ged_best_sb'].dropna()
print(f"   Mean: {dv_summary.mean():.2f}")
print(f"   Std Dev: {dv_summary.std():.2f}")
print(f"   Min: {dv_summary.min():.0f}")
print(f"   Max: {dv_summary.max():.0f}")
print(f"   Zero observations: {(df['ged_best_sb'] == 0).sum():,} ({(df['ged_best_sb'] == 0).mean()*100:.1f}%)")
print(f"   Non-zero observations: {(df['ged_best_sb'] > 0).sum():,} ({(df['ged_best_sb'] > 0).mean()*100:.1f}%)")

# IV Summary
print("\n4. INDEPENDENT VARIABLE: Temperature Anomaly")
temp_summary = df['anomaly'].dropna()
print(f"   Mean: {temp_summary.mean():.3f}°C")
print(f"   Std Dev: {temp_summary.std():.3f}°C")
print(f"   Min: {temp_summary.min():.3f}°C")
print(f"   Max: {temp_summary.max():.3f}°C")

# Create extreme temperature indicator
extreme_threshold = temp_summary.quantile(0.90)
print(f"   90th percentile: {extreme_threshold:.3f}°C")
df['extreme_heat'] = (df['anomaly'] > extreme_threshold).astype(int)
print(f"   Extreme heat months: {df['extreme_heat'].sum():,} ({df['extreme_heat'].mean()*100:.1f}%)")

# Time range
print("\n5. TIME RANGE")
print(f"   Years: {df['year'].min()} - {df['year'].max()}")
print(f"   Months covered: {(df['year'].max() - df['year'].min()) * 12}")

# Correlation analysis
print("\n6. CORRELATION: Temperature Anomaly vs. State-Based Violence")
corr_data = df[['anomaly', 'ged_best_sb']].dropna()
corr = corr_data['anomaly'].corr(corr_data['ged_best_sb'])
print(f"   Pearson correlation: {corr:.4f}")

# Correlation with extreme heat indicator
corr_extreme = df[['extreme_heat', 'ged_best_sb']].dropna()
corr_ext = corr_extreme['extreme_heat'].corr(corr_extreme['ged_best_sb'])
print(f"   Correlation with extreme heat indicator: {corr_ext:.4f}")

# Violence by temperature terciles
print("\n7. VIOLENCE BY TEMPERATURE TERCILES")
df_clean = df.dropna(subset=['anomaly', 'ged_best_sb'])
terciles = pd.qcut(df_clean['anomaly'], q=3, labels=['Cold', 'Moderate', 'Hot'], duplicates='drop')
tercile_summary = df_clean.assign(tercile=terciles).groupby('tercile')['ged_best_sb'].agg(['mean', 'sum', 'count'])
print(tercile_summary)

# Create figures
print("\n8. GENERATING FIGURES...")

# Figure 1: Temperature trend over time
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Temperature trend
ax = axes[0, 0]
temp_by_year = df.groupby('year')['anomaly'].mean()
ax.plot(temp_by_year.index, temp_by_year.values, 'b-', linewidth=2)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('Temperature Anomaly (°C)')
ax.set_title('Global Temperature Anomaly Over Time')
ax.set_ylim(-1, 2)

# Panel B: Violence trend
ax = axes[0, 1]
viol_by_year = df.groupby('year')['ged_best_sb'].sum()
ax.plot(viol_by_year.index, viol_by_year.values, 'r-', linewidth=2)
ax.set_xlabel('Year')
ax.set_ylabel('Total State-Based Violence Deaths')
ax.set_title('Global State-Based Violence Over Time')
ax.set_yscale('log')

# Panel C: Distribution of temperature anomalies
ax = axes[1, 0]
df_clean['anomaly'].hist(bins=50, ax=ax, edgecolor='black', alpha=0.7)
ax.set_xlabel('Temperature Anomaly (°C)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Temperature Anomalies')

# Panel D: Distribution of violence
ax = axes[1, 1]
viol_positive = df_clean[df_clean['ged_best_sb'] > 0]['ged_best_sb']
viol_positive.hist(bins=50, ax=ax, edgecolor='black', alpha=0.7, color='red')
ax.set_xlabel('State-Based Violence Deaths')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Violence Deaths (Positive Values)')

plt.tight_layout()
plt.savefig('eda_figures.png', dpi=300, bbox_inches='tight')
print("   Saved: eda_figures.png")

# Figure 2: Violence by temperature tercile (box plot)
fig, ax = plt.subplots(figsize=(10, 6))
df_box = df_clean.copy()
df_box['tercile_name'] = pd.qcut(df_box['anomaly'], q=3, labels=['Cold (Low)', 'Moderate', 'Hot (High)'], duplicates='drop')

# Log transform for visualization
df_box['viol_log'] = np.log1p(df_box['ged_best_sb'])

sns.boxplot(data=df_box, x='tercile_name', y='viol_log', ax=ax, palette='viridis')
ax.set_xlabel('Temperature Anomaly Tercile')
ax.set_ylabel('Log(1 + State-Based Violence Deaths)')
ax.set_title('State-Based Violence by Temperature Tercile')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('violence_by_temp_tercile.png', dpi=300, bbox_inches='tight')
print("   Saved: violence_by_temp_tercile.png")

# Figure 3: Scatter plot with trend line
fig, ax = plt.subplots(figsize=(10, 6))
sample = df_clean.sample(min(10000, len(df_clean)), random_state=42)
sns.scatterplot(data=sample, x='anomaly', y='ged_best_sb', ax=ax, alpha=0.3, s=10)

# Add trend line
z = np.polyfit(sample['anomaly'], sample['ged_best_sb'], 1)
p = np.poly1d(z)
ax.plot(sample['anomaly'].min(), p(sample['anomaly'].min()),
        sample['anomaly'].max(), p(sample['anomaly'].max()),
        'r-', linewidth=2, label=f'Trend (slope={z[0]:.1f})')

ax.set_xlabel('Temperature Anomaly (°C)')
ax.set_ylabel('State-Based Violence Deaths')
ax.set_title('Temperature Anomaly vs. State-Based Violence')
ax.legend()
plt.tight_layout()
plt.savefig('scatter_temp_violence.png', dpi=300, bbox_inches='tight')
print("   Saved: scatter_temp_violence.png")

print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)
