#!/usr/bin/env python
"""
Milestone 3: Model Training Script.
Trains 6 Random Forest Regression models (2 per transformer: active power and load percentage)
using only the 2013-2016 training datasets.
"""

import os
import sys
import time
import json
import pandas as pd

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.random_forest import TransformerRandomForest

MODELS_DIR = "models"
PROCESSED_DIR = "data/processed"
FEATURE_SUMMARY_PATH = os.path.join(PROCESSED_DIR, "dataset_summary.json")

os.makedirs(MODELS_DIR, exist_ok=True)

def train_all():
    print("=" * 70)
    print("MILESTONE 3: RANDOM FOREST MODEL TRAINING PIPELINE")
    print("=" * 70)

    with open(FEATURE_SUMMARY_PATH, 'r') as f:
        meta = json.load(f)

    transformers = ['t1', 't2', 't3']
    targets = [
        ('power', 'target_next_hour_power_W'),
        ('load', 'target_next_hour_load_pct')
    ]

    training_results = {}

    for t in transformers:
        t_upper = t.upper()
        train_path = os.path.join(PROCESSED_DIR, f"{t}_train.csv")
        print(f"\n[{t_upper}] Loading training dataset from {train_path}...")
        df_train = pd.read_csv(train_path)

        # Get approved 39 features
        feature_cols = meta['transformers'][t_upper]['feature_list']
        assert len(feature_cols) == 39, f"Expected 39 features, got {len(feature_cols)}"
        assert 'uplift_pct' not in feature_cols, "CRITICAL: uplift_pct found in feature list!"
        assert 'target_next_hour_power_W' not in feature_cols, "CRITICAL: target found in features!"
        assert 'target_next_hour_load_pct' not in feature_cols, "CRITICAL: target found in features!"
        assert 'datetime' not in feature_cols, "CRITICAL: datetime found in features!"

        X_train = df_train[feature_cols]
        print(f"[{t_upper}] X_train shape: {X_train.shape} (Rows: {len(X_train)}, Features: {len(feature_cols)})")

        training_results[t_upper] = {}

        for target_label, target_col in targets:
            print(f"\n--- Training {t_upper} Random Forest: {target_label.upper()} ({target_col}) ---")
            y_train = df_train[target_col]

            # Model hyperparameter configuration
            rf_wrapper = TransformerRandomForest(
                transformer_id=t_upper,
                target_name=target_col,
                n_estimators=100,
                max_depth=20,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )

            start_time = time.time()
            rf_wrapper.train(X_train, y_train)
            elapsed = time.time() - start_time

            # In-sample metrics for sanity check
            train_preds = rf_wrapper.predict(X_train)
            train_r2 = float(rf_wrapper.model.score(X_train, y_train))

            model_filename = f"{t}_rf_{target_label}.joblib"
            model_filepath = os.path.join(MODELS_DIR, model_filename)
            rf_wrapper.save(model_filepath)

            print(f"  Trained in {elapsed:.2f} seconds | In-Sample Train R²: {train_r2:.4f}")
            print(f"  Saved model artifact to: {model_filepath}")

            # Top 5 features
            top_feats = rf_wrapper.get_feature_importances().head(5).to_dict(orient='records')

            training_results[t_upper][target_label] = {
                'model_file': model_filepath,
                'target_column': target_col,
                'train_rows': len(X_train),
                'train_seconds': round(elapsed, 2),
                'train_r2': round(train_r2, 4),
                'parameters': rf_wrapper.params,
                'top_5_features': top_feats
            }

    summary_file = os.path.join(MODELS_DIR, "training_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(training_results, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL 6 RANDOM FOREST MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print(f"Summary metadata written to {summary_file}")
    print("=" * 70)

if __name__ == '__main__':
    train_all()

