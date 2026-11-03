#!/usr/bin/env python
"""
Executable script for Milestone 2 - Data Preprocessing & Feature Engineering.
Runs the complete preprocessing pipeline for T1, T2, and T3.
"""

import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.preprocessor import TransformerDataPreprocessor

def main():
    print("="*70)
    print("MILESTONE 2: DATA PREPROCESSING & FEATURE ENGINEERING PIPELINE")
    print("="*70)
    preprocessor = TransformerDataPreprocessor(
        raw_csv_path="data/raw/transformer_hourly_festival_with_temperature.csv",
        processed_dir="data/processed",
        train_end_date="2017-01-01 00:00:00",
        min_periods_24=18,
        min_periods_168=84
    )
    summary = preprocessor.run()
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*70)
    for t_id, meta in summary['transformers'].items():
        print(f"\nTransformer {t_id}:")
        print(f"  Total Raw Rows:        {meta['total_raw_rows']}")
        print(f"  Total Usable Rows:     {meta['usable_rows_total']} ({meta['usable_rows_total']/meta['total_raw_rows']*100:.2f}%)")
        print(f"  Dropped Outage Rows:   {meta['dropped_outage_rows']}")
        print(f"  Train Set (2013-2016): {meta['train_rows']} rows ({meta['train_date_range'][0]} to {meta['train_date_range'][1]})")
        print(f"  Test Set (2017):       {meta['test_rows']} rows ({meta['test_date_range'][0]} to {meta['test_date_range'][1]})")
        print(f"  Input Features:        {meta['feature_count']} features")
        print(f"  Train Overload Events: {meta['overload_samples_train']}")
        print(f"  Test Overload Events:  {meta['overload_samples_test']}")

if __name__ == '__main__':
    main()

