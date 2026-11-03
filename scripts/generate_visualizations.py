import os
import sys

# Ensure local writable cache for matplotlib
os.environ['MPLCONFIGDIR'] = os.path.abspath('.matplotlib_cache')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH = "data/raw/transformer_hourly_festival_with_temperature.csv"
VIS_DIR = "visualizations"
os.makedirs(VIS_DIR, exist_ok=True)

# Set clean aesthetic styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

print(f"Loading data from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)
df['datetime'] = pd.to_datetime(df['datetime_IST'])
df['hour'] = df['datetime'].dt.hour
df['month_year'] = df['datetime'].dt.to_period('M')
df['date'] = df['datetime'].dt.date
df['is_festival'] = df['festival'] != 'Normal day'

colors = {'T1': '#2563eb', 'T2': '#dc2626', 'T3': '#16a34a'}

print("1. Generating Hourly Load Trends...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for i, t in enumerate(['t1', 't2', 't3']):
    t_name = t.upper()
    hourly_stats = df.groupby('hour')[f'{t}_power_W'].agg(['mean', 'median', lambda x: np.percentile(x.dropna(), 25), lambda x: np.percentile(x.dropna(), 75)]).reset_index()
    hourly_stats.columns = ['hour', 'mean', 'median', 'q25', 'q75']
    
    ax = axes[i]
    ax.plot(hourly_stats['hour'], hourly_stats['mean'] / 1000.0, label='Mean Load (kW)', color=colors[t_name], lw=2.5)
    ax.plot(hourly_stats['hour'], hourly_stats['median'] / 1000.0, label='Median Load (kW)', color=colors[t_name], ls='--', lw=1.8)
    ax.fill_between(hourly_stats['hour'], hourly_stats['q25'] / 1000.0, hourly_stats['q75'] / 1000.0, color=colors[t_name], alpha=0.18, label='IQR (25th-75th %ile)')
    ax.set_title(f"Transformer {t_name} Hourly Diurnal Load Profile", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Hour of Day (0-23 IST)", fontsize=11)
    if i == 0:
        ax.set_ylabel("Power Demand (kW)", fontsize=11)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, ls=':', alpha=0.6)
    ax.legend(loc='upper left', frameon=True)

plt.tight_layout()
fig.savefig(os.path.join(VIS_DIR, '01_hourly_load_trends.png'), dpi=200)
plt.close(fig)

print("2. Generating Daily and Monthly Load Trends...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))

# Monthly mean power
monthly_load = df.groupby('month_year')[['t1_power_W', 't2_power_W', 't3_power_W']].mean() / 1000.0
monthly_load.index = monthly_load.index.astype(str)

monthly_load.plot(ax=ax1, color=[colors['T1'], colors['T2'], colors['T3']], lw=2.2, marker='o', markersize=4)
ax1.set_title("Monthly Mean Load Trend (Nov 2013 - Dec 2017)", fontsize=13, fontweight='bold', pad=10)
ax1.set_ylabel("Mean Active Power (kW)", fontsize=11)
ax1.set_xlabel("")
ax1.legend(['T1 (315 kVA implied)', 'T2 (315 kVA implied)', 'T3 (250 kVA implied)'], loc='upper right', frameon=True)
ax1.grid(True, ls=':', alpha=0.6)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Daily 30-day rolling average power to show long-term seasonality
daily_power = df.groupby('date')[['t1_power_W', 't2_power_W', 't3_power_W']].mean() / 1000.0
daily_rolling = daily_power.rolling(window=30, min_periods=7).mean()
daily_rolling.plot(ax=ax2, color=[colors['T1'], colors['T2'], colors['T3']], lw=2)
ax2.set_title("30-Day Moving Average Active Power Demand", fontsize=13, fontweight='bold', pad=10)
ax2.set_ylabel("Smoothed Power (kW)", fontsize=11)
ax2.set_xlabel("Date", fontsize=11)
ax2.legend(['T1 (Offline mid-2015 to mid-2016)', 'T2', 'T3'], loc='upper right', frameon=True)
ax2.grid(True, ls=':', alpha=0.6)

plt.tight_layout()
fig.savefig(os.path.join(VIS_DIR, '02_daily_monthly_load_trends.png'), dpi=200)
plt.close(fig)

print("3. Generating Festival vs Non-Festival Comparison...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
for i, t in enumerate(['t1', 't2', 't3']):
    t_name = t.upper()
    ax = axes[i]
    norm_hourly = df[~df['is_festival']].groupby('hour')[f'{t}_power_W'].mean() / 1000.0
    fest_hourly = df[df['is_festival']].groupby('hour')[f'{t}_power_W'].mean() / 1000.0
    
    ax.plot(norm_hourly.index, norm_hourly.values, label='Normal Day Mean (kW)', color='#64748b', lw=2.2, ls='--')
    ax.plot(fest_hourly.index, fest_hourly.values, label='Festival Day Mean (kW)', color=colors[t_name], lw=2.8, marker='s', markersize=4)
    
    ax.set_title(f"Transformer {t_name}: Festival vs Normal Day Profile", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Hour of Day (IST)", fontsize=11)
    if i == 0:
        ax.set_ylabel("Power Demand (kW)", fontsize=11)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, ls=':', alpha=0.6)
    ax.legend(loc='upper left', frameon=True)

plt.tight_layout()
fig.savefig(os.path.join(VIS_DIR, '03_festival_vs_non_festival.png'), dpi=200)
plt.close(fig)

print("4. Generating Transformer-wise Comparison...")
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# (a) Load Pct Boxplot
load_data = [
    df['t1_load_pct'].dropna(),
    df['t2_load_pct'].dropna(),
    df['t3_load_pct'].dropna()
]
bplot1 = axes[0, 0].boxplot(load_data, patch_artist=True, tick_labels=['Transformer T1', 'Transformer T2', 'Transformer T3'],
                            medianprops=dict(color='black', lw=2),
                            boxprops=dict(lw=1.5))
for patch, color in zip(bplot1['boxes'], [colors['T1'], colors['T2'], colors['T3']]):
    patch.set_facecolor(color)
    patch.set_alpha(0.65)
axes[0, 0].axhline(100.0, color='red', ls='--', lw=1.8, label='Rated Capacity (100% Load)')
axes[0, 0].axhline(120.0, color='darkred', ls=':', lw=1.8, label='Overload Threshold (120%)')
axes[0, 0].set_title("Load Percentage Distribution & Overload Exposure", fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel("Load Percentage (%)", fontsize=11)
axes[0, 0].legend(loc='upper right', frameon=True)
axes[0, 0].grid(True, ls=':', alpha=0.6)

# (b) Oil Temperature Boxplot
temp_data = [
    df['t1_oil_temp_C'].dropna(),
    df['t2_oil_temp_C'].dropna(),
    df['t3_oil_temp_C'].dropna()
]
bplot2 = axes[0, 1].boxplot(temp_data, patch_artist=True, tick_labels=['Transformer T1', 'Transformer T2', 'Transformer T3'],
                            medianprops=dict(color='black', lw=2),
                            boxprops=dict(lw=1.5))
for patch, color in zip(bplot2['boxes'], [colors['T1'], colors['T2'], colors['T3']]):
    patch.set_facecolor(color)
    patch.set_alpha(0.65)
axes[0, 1].axhline(90.0, color='red', ls='--', lw=1.8, label='Warning Temp (90°C)')
axes[0, 1].set_title("Top-Oil Temperature Distribution (°C)", fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel("Oil Temperature (°C)", fontsize=11)
axes[0, 1].legend(loc='upper right', frameon=True)
axes[0, 1].grid(True, ls=':', alpha=0.6)

# (c) Active Power Distribution KDE / Hist
for t in ['t1', 't2', 't3']:
    t_name = t.upper()
    sns.kdeplot(df[f'{t}_power_W'].dropna() / 1000.0, ax=axes[1, 0], label=f'{t_name} Active Power', color=colors[t_name], lw=2.2)
axes[1, 0].set_title("Active Power Probability Density Function", fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel("Active Power (kW)", fontsize=11)
axes[1, 0].set_ylabel("Density", fontsize=11)
axes[1, 0].legend(loc='upper right', frameon=True)
axes[1, 0].grid(True, ls=':', alpha=0.6)

# (d) Oil Temp vs Power Scatter Sample
sample_df = df.sample(n=2500, random_state=42)
for t in ['t1', 't2', 't3']:
    t_name = t.upper()
    axes[1, 1].scatter(sample_df[f'{t}_power_W'] / 1000.0, sample_df[f'{t}_oil_temp_C'],
                       label=t_name, color=colors[t_name], alpha=0.35, s=16)
axes[1, 1].set_title("Oil Temperature vs Active Power (2,500 Sampled Hours)", fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel("Active Power (kW)", fontsize=11)
axes[1, 1].set_ylabel("Oil Temperature (°C)", fontsize=11)
axes[1, 1].legend(loc='upper left', frameon=True)
axes[1, 1].grid(True, ls=':', alpha=0.6)

plt.tight_layout()
fig.savefig(os.path.join(VIS_DIR, '04_transformer_wise_comparison.png'), dpi=200)
plt.close(fig)

print("5. Generating Missing-Value Visualization Timeline...")
fig, ax = plt.subplots(figsize=(16, 4.5))

# Create daily missing rate matrix
df_daily_missing = df.set_index('datetime')[['t1_power_W', 't2_power_W', 't3_power_W', 'ambient_temperature_C']].resample('D').apply(lambda s: s.isnull().mean())
df_daily_missing.columns = ['Transformer T1', 'Transformer T2', 'Transformer T3', 'Ambient Temperature']

# Plot timeline heatmap
sns.heatmap(df_daily_missing.T, ax=ax, cmap='YlOrRd', cbar_kws={'label': 'Missing Fraction (0 = Complete, 1 = Outage)'}, vmin=0, vmax=1)
ax.set_title("Temporal Missing Data Profile (Daily Sensor Outages 2013-2017)", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Timeline (Daily Sample Frequency)", fontsize=11)
ax.set_ylabel("Subsystem Telemetry", fontsize=11)

# Format x ticks to show every 6 months
xtick_locs = range(0, len(df_daily_missing), 180)
xtick_labels = [df_daily_missing.index[i].strftime('%Y-%m') for i in xtick_locs]
ax.set_xticks(xtick_locs)
ax.set_xticklabels(xtick_labels, rotation=0)

plt.tight_layout()
fig.savefig(os.path.join(VIS_DIR, '05_missing_values_timeline.png'), dpi=200)
plt.close(fig)

print("6. Generating Correlation Heatmaps...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, t in enumerate(['t1', 't2', 't3']):
    t_name = t.upper()
    corr_cols = [f'{t}_power_W', f'{t}_current_A', f'{t}_load_pct', f'{t}_oil_temp_C', 'ambient_temperature_C']
    sub_corr = df[corr_cols].dropna().corr()
    sub_corr.columns = ['Power (W)', 'Current (A)', 'Load (%)', 'Oil Temp (°C)', 'Ambient (°C)']
    sub_corr.index = ['Power (W)', 'Current (A)', 'Load (%)', 'Oil Temp (°C)', 'Ambient (°C)']
    
    sns.heatmap(sub_corr, ax=axes[i], annot=True, fmt='.2f', cmap='coolwarm', vmin=-0.2, vmax=1.0, cbar=(i == 2))
    axes[i].set_title(f"Correlation Matrix: {t_name}", fontsize=12, fontweight='bold', pad=10)

plt.tight_layout()
fig.savefig(os.path.join(VIS_DIR, '06_correlation_heatmaps.png'), dpi=200)
plt.close(fig)

print("All visualizations generated successfully!")

