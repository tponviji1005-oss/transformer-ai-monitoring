"""
Training Script for Milestone 4 Overload Detection & Anomaly System.
Trains:
1. Transformer T2: Supervised Random Forest Classifier on historical training data
   with Cross-Validation threshold selection prioritizing recall.
   
   CROSS-VALIDATION & THRESHOLD SELECTION DOCUMENTATION:
   - The production baseline threshold (0.3382) was selected using 5-fold StratifiedKFold
     because the overload class is rare (only 170 overloads out of 24,238 training rows = 0.701%).
     Stratification ensures that every validation fold contains a representative number of overload samples.
   - Note on Temporal Mixing: Shuffled StratifiedKFold mixes observations across different
     years (2013–2016) within the training set. However, the 2017 test set was strictly held out.
   - Configurable Chronological Validation: A `cv_strategy` parameter is provided ('stratified' or
     'chronological') to allow future retraining using TimeSeriesSplit to prevent temporal mixing.
     
2. Transformer T1: Unsupervised Isolation Forest Anomaly Detector on normal operating data.
Persists model bundles to models/overload/ and logs metadata to models/overload/overload_training_summary.json.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict
import joblib
import pandas as pd

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detection.overload_classifier import TransformerOverloadClassifier
from src.detection.anomaly_detector import TransformerAnomalyDetector


def load_feature_list() -> list:
    """Loads approved 39 feature names from existing Milestone 3 model."""
    ref_model_path = os.path.join('models', 't2_rf_power.joblib')
    if not os.path.exists(ref_model_path):
        raise FileNotFoundError(f"Reference model not found at {ref_model_path}")
    bundle = joblib.load(ref_model_path)
    return bundle['feature_names']


def main(cv_strategy: str = "stratified") -> None:
    print("=" * 70)
    print("MILESTONE 4: TRAINING OVERLOAD DETECTION & EARLY WARNING MODELS")
    print(f"Validation Strategy: {cv_strategy.upper()} Cross-Validation")
    print("=" * 70)

    os.makedirs(os.path.join('models', 'overload'), exist_ok=True)
    features = load_feature_list()
    print(f"Loaded {len(features)} approved features from Milestone 3.")

    training_summary: Dict[str, Any] = {}

    # -------------------------------------------------------------
    # 1. Train T2 Supervised Overload Classifier
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print(f"STEP 1: Training Transformer T2 Overload Classifier ({cv_strategy} CV)")
    print("-" * 50)

    t2_train_path = os.path.join('data', 'processed', 't2_train.csv')
    df_t2_train = pd.read_csv(t2_train_path)
    X_t2_train = df_t2_train[features]
    y_t2_train = df_t2_train['target_next_hour_overload'].astype(int)

    overload_count = int(y_t2_train.sum())
    total_count = len(df_t2_train)
    print(f"T2 Train Data: {total_count:,} rows | Overload events: {overload_count} ({overload_count/total_count*100:.3f}%)")

    t2_classifier = TransformerOverloadClassifier(
        transformer_id="T2",
        n_estimators=100,
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        default_threshold=0.5,
    )

    t0 = time.time()
    t2_train_res = t2_classifier.train(
        X_train=X_t2_train,
        y_train=y_t2_train,
        tune_threshold=True,
        cv_folds=5,
        target_recall=0.85,
        cv_strategy=cv_strategy,
    )
    t2_duration = time.time() - t0

    t2_model_path = os.path.join('models', 'overload', 't2_overload_classifier.joblib')
    t2_classifier.save(t2_model_path)
    print(f"T2 Classifier trained in {t2_duration:.2f}s and saved to {t2_model_path}")
    print(f"  OOF ROC-AUC:            {t2_classifier.cv_metrics_['oof_roc_auc']:.4f}")
    print(f"  OOF PR-AUC:             {t2_classifier.cv_metrics_['oof_pr_auc']:.4f}")
    print(f"  Selected Threshold:     {t2_classifier.selected_threshold:.4f}")
    print(f"  OOF Recall (Selected):  {t2_classifier.cv_metrics_['oof_recall_selected']*100:.2f}% (vs Default: {t2_classifier.cv_metrics_['oof_recall_default']*100:.2f}%)")
    print(f"  OOF Precision (Selected):{t2_classifier.cv_metrics_['oof_precision_selected']*100:.2f}%")

    top_feat = t2_classifier.get_feature_importances().head(10).to_dict(orient='records')

    training_summary['T2_classifier'] = {
        'model_file': t2_model_path,
        'train_samples': total_count,
        'train_overloads': overload_count,
        'training_time_seconds': round(t2_duration, 2),
        'selected_threshold': round(t2_classifier.selected_threshold, 4),
        'default_threshold': 0.5,
        'cv_metrics': t2_classifier.cv_metrics_,
        'top_10_features': top_feat,
    }

    # -------------------------------------------------------------
    # 2. Train T1 Unsupervised Anomaly Detector
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("STEP 2: Training Transformer T1 Anomaly Detector (Isolation Forest)")
    print("-" * 50)

    t1_train_path = os.path.join('data', 'processed', 't1_train.csv')
    df_t1_train = pd.read_csv(t1_train_path)
    X_t1_train = df_t1_train[features]
    print(f"T1 Train Data: {len(df_t1_train):,} rows (strictly 0 overloads in 2013-2016)")

    t1_detector = TransformerAnomalyDetector(
        transformer_id="T1",
        n_estimators=150,
        contamination=0.01,
        random_state=42,
        n_jobs=-1,
    )

    t0 = time.time()
    t1_train_res = t1_detector.train(X_t1_train)
    t1_duration = time.time() - t0

    t1_model_path = os.path.join('models', 'overload', 't1_anomaly_detector.joblib')
    t1_detector.save(t1_model_path)
    print(f"T1 Anomaly Detector trained in {t1_duration:.2f}s and saved to {t1_model_path}")
    print(f"  Contamination: {t1_detector.contamination}")
    print(f"  Estimators:    {t1_detector.params['n_estimators']}")

    training_summary['T1_anomaly_detector'] = {
        'model_file': t1_model_path,
        'train_samples': len(df_t1_train),
        'training_time_seconds': round(t1_duration, 2),
        'contamination': t1_detector.contamination,
        'n_estimators': t1_detector.params['n_estimators'],
        'train_score_min': t1_train_res['train_score_min'],
        'train_score_max': t1_train_res['train_score_max'],
    }

    # Save summary json
    summary_path = os.path.join('models', 'overload', 'overload_training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(training_summary, f, indent=2)
    print(f"\nTraining summary saved to {summary_path}")
    print("=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Overload Detection and Anomaly Models.")
    parser.add_argument(
        '--cv_strategy',
        type=str,
        default='stratified',
        choices=['stratified', 'chronological'],
        help="Cross-validation strategy for threshold tuning: 'stratified' (default) or 'chronological' (TimeSeriesSplit)."
    )
    args = parser.parse_args()
    main(cv_strategy=args.cv_strategy)

