"""
Transformer Data Preprocessor Module.
Loads raw telemetry, isolates transformers, applies leakage-free feature engineering,
splits chronologically into Train (2013-2016) and Test (2017), and exports to data/processed/.
"""

import json
import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.preprocessing.feature_engineering import FeatureEngineer


class TransformerDataPreprocessor:
    """
    Orchestrates end-to-end preprocessing and feature generation for T1, T2, and T3.
    """

    EXCLUDED_FEATURES_RATIONALE = {
        'uplift_pct': 'User rule: uplift_pct is a synthetic boost percentage, not an observable telemetry feature. Model must learn holiday patterns from historical telemetry.',
        'voltage_V': 'Zero variance: static constant 415.0V (std=0.0) across all records. Provides no predictive variance.',
        'pf': 'Zero variance: static constant 0.90 (std=0.0) across all records. Provides no predictive variance.',
        'frequency_Hz': 'Zero variance: static constant 50.00Hz (std=0.0) across all records. Provides no predictive variance.',
        'watts_per_frequency': 'Exact multicollinearity: exactly equal to power_W / 50.00.',
        'current_A': 'Exact multicollinearity: since voltage (415V) and pf (0.90) are constant, current is perfectly collinear with active power (I = P / 646.9042).',
        'observed_W': 'Redundancy & ambiguity: equals power_W on normal days and represents pre-uplift synthetic baseline during festivals. Excluded from feature inputs.'
    }

    def __init__(
        self,
        raw_csv_path: str = "data/raw/transformer_hourly_festival_with_temperature.csv",
        processed_dir: str = "data/processed",
        train_end_date: str = "2017-01-01 00:00:00",
        min_periods_24: int = 18,
        min_periods_168: int = 84
    ):
        self.raw_csv_path = raw_csv_path
        self.processed_dir = processed_dir
        self.train_end_date = train_end_date
        self.feature_engineer = FeatureEngineer(
            min_periods_24=min_periods_24,
            min_periods_168=min_periods_168
        )
        os.makedirs(self.processed_dir, exist_ok=True)

    def load_and_validate_raw_data(self) -> pd.DataFrame:
        """Loads raw dataset and validates timestamp continuity."""
        if not os.path.exists(self.raw_csv_path):
            raise FileNotFoundError(f"Raw dataset not found at {self.raw_csv_path}")

        df = pd.read_csv(self.raw_csv_path)
        df['datetime'] = pd.to_datetime(df['datetime_IST'])
        df = df.sort_values('datetime').reset_index(drop=True)

        # Continuity check: ensure 1-hour regular grid
        time_diffs = df['datetime'].diff().dropna()
        is_continuous = (time_diffs == pd.Timedelta(hours=1)).all()
        if not is_continuous:
            raise ValueError("Raw dataset timestamps contain irregular step intervals!")

        return df

    def extract_transformer_data(self, df: pd.DataFrame, transformer_id: str) -> pd.DataFrame:
        """Isolates telemetry for a single transformer, dropping invariant/redundant channels."""
        t = transformer_id.lower()
        sub_df = pd.DataFrame({
            'datetime_IST': df['datetime_IST'],
            'datetime': df['datetime'],
            'power_W': df[f'{t}_power_W'],
            'oil_temp_C': df[f'{t}_oil_temp_C'],
            'load_pct': df[f'{t}_load_pct'],
            'ambient_temp_C': df['ambient_temperature_C'],
            'festival': df['festival']
        })
        return sub_df

    def process_transformer(self, df: pd.DataFrame, transformer_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Executes full preprocessing and chronological split for one transformer.
        Returns (train_df, test_df, metadata_dict).
        """
        t = transformer_id.lower()
        sub_df = self.extract_transformer_data(df, t)

        # 1. Feature Engineering (applied across full timeline before splitting to preserve lag context)
        feat_df = self.feature_engineer.add_calendar_features(sub_df, dt_col='datetime')
        feat_df = self.feature_engineer.add_autoregressive_lags(feat_df)
        feat_df = self.feature_engineer.add_rolling_features(feat_df)
        feat_df = self.feature_engineer.add_prediction_targets(feat_df)

        # 2. Strict validation: Identify required ML features
        target_col = 'target_next_hour_power_W'
        feature_cols = [
            c for c in feat_df.columns
            if c not in ['datetime_IST', 'datetime', 'festival', target_col, 'target_next_hour_load_pct', 'target_next_hour_overload']
        ]

        # Verify uplift_pct is absent
        assert 'uplift_pct' not in feat_df.columns, "CRITICAL ERROR: uplift_pct found in engineered features!"

        # 3. Drop rows with NaNs in required features or target
        # (Preserves true outages as non-trainable, zero artificial interpolation)
        clean_mask = feat_df[feature_cols + [target_col]].notnull().all(axis=1)
        valid_df = feat_df[clean_mask].copy()

        # 4. Chronological Split (Train: 2013-2016, Test: 2017)
        # Boundary condition: target timestamp (datetime + 1 hour) must also be strictly before 2017-01-01 00:00:00
        # This ensures no training record utilizes a target landing inside the test evaluation period.
        train_mask = (valid_df['datetime'] + pd.Timedelta(hours=1)) < self.train_end_date
        test_mask = valid_df['datetime'] >= self.train_end_date

        train_df = valid_df[train_mask].reset_index(drop=True)
        test_df = valid_df[test_mask].reset_index(drop=True)

        meta = {
            'transformer': transformer_id.upper(),
            'total_raw_rows': len(sub_df),
            'usable_rows_total': len(valid_df),
            'dropped_outage_rows': len(sub_df) - len(valid_df),
            'train_rows': len(train_df),
            'train_date_range': [str(train_df['datetime'].min()), str(train_df['datetime'].max())],
            'test_rows': len(test_df),
            'test_date_range': [str(test_df['datetime'].min()), str(test_df['datetime'].max())],
            'feature_count': len(feature_cols),
            'feature_list': feature_cols,
            'target_variables': ['target_next_hour_power_W', 'target_next_hour_load_pct', 'target_next_hour_overload'],
            'overload_samples_train': int((train_df['target_next_hour_overload'] == 1.0).sum()),
            'overload_samples_test': int((test_df['target_next_hour_overload'] == 1.0).sum())
        }

        return train_df, test_df, meta

    def run(self) -> Dict:
        """Processes T1, T2, and T3, saving processed CSVs and summary JSON."""
        print(f"Loading raw data from {self.raw_csv_path}...")
        df = self.load_and_validate_raw_data()
        summary = {
            'transformers': {},
            'excluded_features_rationale': self.EXCLUDED_FEATURES_RATIONALE
        }

        for t_id in ['t1', 't2', 't3']:
            print(f"\nProcessing Transformer {t_id.upper()}...")
            train_df, test_df, meta = self.process_transformer(df, t_id)
            summary['transformers'][t_id.upper()] = meta

            # Export to data/processed
            train_path = os.path.join(self.processed_dir, f"{t_id}_train.csv")
            test_path = os.path.join(self.processed_dir, f"{t_id}_test.csv")

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            print(f"  Saved {train_path} ({len(train_df)} rows)")
            print(f"  Saved {test_path} ({len(test_df)} rows)")

        summary_path = os.path.join(self.processed_dir, "dataset_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved metadata summary to {summary_path}")

        return summary

