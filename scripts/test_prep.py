import pandas as pd
import numpy as np

CSV_PATH = "data/raw/transformer_hourly_festival_with_temperature.csv"

def test():
    df = pd.read_csv(CSV_PATH)
    df['datetime'] = pd.to_datetime(df['datetime_IST'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Check regular index
    full_range = pd.date_range(start=df['datetime'].min(), end=df['datetime'].max(), freq='1h')
    print("Full date range equals df datetime exactly:", (df['datetime'] == full_range).all())
    
    for t in ['t1', 't2', 't3']:
        p_col = f"{t}_power_W"
        oil_col = f"{t}_oil_temp_C"
        load_col = f"{t}_load_pct"
        
        # Build individual transformer dataframe
        t_df = pd.DataFrame({
            'datetime': df['datetime'],
            'power_W': df[p_col],
            'oil_temp_C': df[oil_col],
            'load_pct': df[load_col],
            'ambient_temp_C': df['ambient_temperature_C'],
            'festival': df['festival']
        })
        
        # Target: next hour power
        t_df['target_next_hour_power_W'] = t_df['power_W'].shift(-1)
        t_df['target_next_hour_load_pct'] = t_df['load_pct'].shift(-1)
        t_df['target_next_hour_overload'] = (t_df['target_next_hour_load_pct'] > 100.0).astype(float)
        # If target power is NaN, set target overload to NaN as well
        t_df.loc[t_df['target_next_hour_power_W'].isnull(), 'target_next_hour_overload'] = np.nan
        
        # Lags
        t_df['load_lag_1h'] = t_df['power_W'].shift(1)
        t_df['load_lag_24h'] = t_df['power_W'].shift(24)
        t_df['load_lag_168h'] = t_df['power_W'].shift(168)
        
        # Rolling on past observations
        past_power = t_df['power_W'].shift(1)
        t_df['load_rolling_mean_24h'] = past_power.rolling(window=24, min_periods=24).mean()
        t_df['load_rolling_std_24h'] = past_power.rolling(window=24, min_periods=24).std()
        t_df['load_rolling_mean_168h'] = past_power.rolling(window=168, min_periods=168).mean()
        t_df['load_rolling_std_168h'] = past_power.rolling(window=168, min_periods=168).std()
        
        # Complete rows check
        required_cols = [
            'target_next_hour_power_W', 'power_W', 'load_lag_1h', 'load_lag_24h', 'load_lag_168h',
            'load_rolling_mean_24h', 'load_rolling_std_24h', 'load_rolling_mean_168h', 'load_rolling_std_168h'
        ]
        valid_mask = t_df[required_cols].notnull().all(axis=1)
        
        train_valid = valid_mask & (t_df['datetime'] < '2017-01-01')
        test_valid = valid_mask & (t_df['datetime'] >= '2017-01-01')
        
        print(f"\n=== Transformer {t.upper()} ===")
        print(f"Total raw rows: {len(t_df)}")
        print(f"Total usable records with all lags & target: {valid_mask.sum()} ({valid_mask.mean()*100:.2f}%)")
        print(f"Train usable records (2013-2016): {train_valid.sum()}")
        print(f"Test usable records (2017): {test_valid.sum()}")

if __name__ == "__main__":
    test()

