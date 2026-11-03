"""
Feature Engineering Module for Transformer Telemetry System.
Constructs leakage-free calendar, lag, and rolling features strictly from past observations.
Strictly excludes uplift_pct and invariant signals.
"""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Constructs ML input features and next-hour prediction targets for transformer load.
    Ensures strict separation of past observations and future targets.
    """

    FESTIVAL_LIST = [
        'Christmas',
        'Diwali',
        'Dussehra',
        'Eid-ul-Fitr',
        'Holi',
        'New Year',
        'Pongal/Sankranti'
    ]

    def __init__(self, min_periods_24: int = 18, min_periods_168: int = 84):
        self.min_periods_24 = min_periods_24
        self.min_periods_168 = min_periods_168

    def add_calendar_features(self, df: pd.DataFrame, dt_col: str = 'datetime') -> pd.DataFrame:
        """Adds cyclical temporal features and festival indicators."""
        df = df.copy()
        dt = df[dt_col]

        # Basic calendar units
        df['hour'] = dt.dt.hour
        df['day_of_week'] = dt.dt.dayofweek
        df['month'] = dt.dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Cyclical sine/cosine encodings (smooth continuity across 23->0, 6->0, 12->1)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
        df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12.0)
        df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12.0)

        # Festival binary indicator and one-hot encoding
        df['is_festival'] = (df['festival'] != 'Normal day').astype(int)
        for fest in self.FESTIVAL_LIST:
            clean_name = fest.replace('/', '_').replace(' ', '_').lower()
            df[f'fest_{clean_name}'] = (df['festival'] == fest).astype(int)

        return df

    def add_autoregressive_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates autoregressive lags for load, oil temp, and ambient temp.
        Timestamp continuity is enforced: since dataframe is regular 1-hour grid,
        shift(k) is strictly k hours in the past.
        """
        df = df.copy()

        # Load lags (1h, 24h, 168h)
        df['load_lag_1h'] = df['power_W'].shift(1)
        df['load_lag_24h'] = df['power_W'].shift(24)
        df['load_lag_168h'] = df['power_W'].shift(168)

        # Oil temperature lags (1h, 24h)
        df['oil_temp_lag_1h'] = df['oil_temp_C'].shift(1)
        df['oil_temp_lag_24h'] = df['oil_temp_C'].shift(24)

        # Ambient temperature lags (1h, 24h)
        df['ambient_temp_lag_1h'] = df['ambient_temp_C'].shift(1)
        df['ambient_temp_lag_24h'] = df['ambient_temp_C'].shift(24)

        # Thermal gradient at time t
        df['temp_gradient'] = df['oil_temp_C'] - df['ambient_temp_C']
        df['temp_gradient_lag_1h'] = df['oil_temp_lag_1h'] - df['ambient_temp_lag_1h']

        return df

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates moving window statistics strictly over past observations.
        Uses shift(1) before rolling to prevent current observation leakage.
        """
        df = df.copy()

        # Strictly past power series (excludes time t)
        past_power = df['power_W'].shift(1)
        df['load_rolling_mean_24h'] = past_power.rolling(window=24, min_periods=self.min_periods_24).mean()
        df['load_rolling_std_24h'] = past_power.rolling(window=24, min_periods=self.min_periods_24).std()
        df['load_rolling_min_24h'] = past_power.rolling(window=24, min_periods=self.min_periods_24).min()
        df['load_rolling_max_24h'] = past_power.rolling(window=24, min_periods=self.min_periods_24).max()

        df['load_rolling_mean_168h'] = past_power.rolling(window=168, min_periods=self.min_periods_168).mean()
        df['load_rolling_std_168h'] = past_power.rolling(window=168, min_periods=self.min_periods_168).std()

        # Rolling ambient temperature (past)
        past_ambient = df['ambient_temp_C'].shift(1)
        df['ambient_rolling_mean_24h'] = past_ambient.rolling(window=24, min_periods=self.min_periods_24).mean()

        # Rolling oil temperature (past)
        past_oil = df['oil_temp_C'].shift(1)
        df['oil_temp_rolling_mean_24h'] = past_oil.rolling(window=24, min_periods=self.min_periods_24).mean()

        return df

    def add_prediction_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Constructs next-hour targets (t+1):
        1. target_next_hour_power_W (Regression)
        2. target_next_hour_load_pct (Auxiliary regression)
        3. target_next_hour_overload (Binary classification: load_pct > 100%)
        """
        df = df.copy()

        df['target_next_hour_power_W'] = df['power_W'].shift(-1)
        df['target_next_hour_load_pct'] = df['load_pct'].shift(-1)
        
        # Binary overload label
        df['target_next_hour_overload'] = (df['target_next_hour_load_pct'] > 100.0).astype(float)
        # If target power is NaN, set target overload to NaN as well
        df.loc[df['target_next_hour_power_W'].isnull(), 'target_next_hour_overload'] = np.nan

        return df

