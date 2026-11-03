import pandas as pd
import numpy as np
import json
import os

CSV_PATH = "data/raw/transformer_hourly_festival_with_temperature.csv"

def analyze():
    print(f"Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    print("\n=== 1. DATASET OVERVIEW ===")
    n_rows, n_cols = df.shape
    print(f"Total Rows: {n_rows}")
    print(f"Total Columns: {n_cols}")
    print("Columns:", list(df.columns))
    print("\nData Types:\n", df.dtypes)
    
    # DateTime analysis
    df['datetime_parsed'] = pd.to_datetime(df['datetime_IST'], errors='coerce')
    min_date = df['datetime_parsed'].min()
    max_date = df['datetime_parsed'].max()
    print(f"\nDate Range: {min_date} to {max_date}")
    
    # Duration and expected rows
    diffs = df['datetime_parsed'].diff()
    freq_counts = diffs.value_counts()
    print(f"Sampling Frequency Step Diffs:\n{freq_counts.head(5)}")
    
    # Duplicate records
    n_dupes = df.duplicated().sum()
    n_time_dupes = df.duplicated(subset=['datetime_IST']).sum()
    print(f"Full duplicate rows: {n_dupes}")
    print(f"Duplicate datetime_IST entries: {n_time_dupes}")
    
    # Transformers present
    # Looking at columns: t1_*, t2_*, t3_*
    t_prefixes = ['t1', 't2', 't3']
    print("\n=== 2. DATA QUALITY ===")
    # Missing values column-wise
    missing_col = df.isnull().sum()
    missing_pct = (missing_col / n_rows) * 100
    missing_summary = pd.DataFrame({'Missing_Count': missing_col, 'Missing_Pct': missing_pct})
    print("\nMissing values (columns with >0 missing):\n", missing_summary[missing_summary['Missing_Count'] > 0])
    print(f"Total missing cells across entire dataset: {df.isnull().sum().sum()}")
    
    # Check invalid / unusual readings
    # Check voltage, current, power factor, frequency, load_pct, oil_temp
    print("\n--- Unusual / Invalid Value Checks ---")
    for t in t_prefixes:
        print(f"\nTransformer {t.upper()} Checks:")
        for col in [f"{t}_voltage_V", f"{t}_current_A", f"{t}_power_W", f"{t}_pf", f"{t}_frequency_Hz", f"{t}_load_pct", f"{t}_oil_temp_C"]:
            if col in df.columns:
                s = df[col]
                print(f"  {col}: min={s.min():.2f}, max={s.max():.2f}, mean={s.mean():.2f}, std={s.std():.2f}, negative_count={(s < 0).sum()}, zero_count={(s == 0).sum()}")

    print("\nAmbient Temperature:")
    s = df['ambient_temperature_C']
    print(f"  ambient_temperature_C: min={s.min():.2f}, max={s.max():.2f}, mean={s.mean():.2f}, std={s.std():.2f}")
    
    print("\nUplift pct & Festival:")
    print("  Festival unique values:\n", df['festival'].value_counts())
    print(f"  uplift_pct: min={df['uplift_pct'].min()}, max={df['uplift_pct'].max()}, non_zero={(df['uplift_pct'] != 0).sum()}")

    # Check observed_W vs power_W
    for t in t_prefixes:
        diff = (df[f'{t}_observed_W'] - df[f'{t}_power_W']).abs()
        print(f"  {t.upper()} max abs difference between observed_W and power_W: {diff.max()}")

    # Time continuity check
    df_sorted = df.sort_values('datetime_parsed')
    time_gaps = df_sorted['datetime_parsed'].diff()
    abnormal_gaps = df_sorted[time_gaps > pd.Timedelta(hours=1)]
    print(f"\nNumber of time gaps > 1 hour: {len(abnormal_gaps)}")
    if len(abnormal_gaps) > 0:
        print("Sample abnormal gaps:\n", abnormal_gaps[['datetime_IST', 'timestamp']].head(5))

    print("\n=== 3. TRANSFORMER ANALYSIS ===")
    for t in t_prefixes:
        print(f"\n--- {t.upper()} Summary ---")
        cols = [c for c in df.columns if c.startswith(f"{t}_")]
        print(f"Available parameters ({len(cols)}): {cols}")
        print(df[cols].describe().T[['mean', 'std', 'min', '50%', 'max']])

    print("\n=== 4. FESTIVAL ANALYSIS ===")
    # Compare festival and non-festival load patterns
    is_festival = df['festival'] != 'Normal day'
    print(f"Festival hours count: {is_festival.sum()} ({is_festival.mean()*100:.2f}%)")
    print(f"Normal day hours count: {(~is_festival).sum()} ({(~is_festival).mean()*100:.2f}%)")
    
    for t in t_prefixes:
        p_norm = df.loc[~is_festival, f'{t}_power_W']
        p_fest = df.loc[is_festival, f'{t}_power_W']
        print(f"\n{t.upper()} Power (W):")
        print(f"  Normal:   mean={p_norm.mean():.2f}, median={p_norm.median():.2f}, max={p_norm.max():.2f}")
        print(f"  Festival: mean={p_fest.mean():.2f}, median={p_fest.median():.2f}, max={p_fest.max():.2f}")
        print(f"  Uplift in mean power: {((p_fest.mean() - p_norm.mean()) / p_norm.mean()) * 100:.2f}%")

if __name__ == "__main__":
    analyze()

