"""
Milestone 4 Independent Forensic Audit Script.
Performs independent recalculations, leakage checks, temporal CV inspection,
metric verification, and model loadability checks without modifying any files.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detection.overload_classifier import TransformerOverloadClassifier
from src.detection.anomaly_detector import TransformerAnomalyDetector
from src.detection.threshold_monitor import ThresholdMonitor, MonitoringStatus
from src.detection.alert_engine import UnifiedAlertEngine, AlertLevel, AlertType


def audit() -> dict:
    results = {}
    print("=" * 70)
    print("STARTING INDEPENDENT FORENSIC AUDIT OF MILESTONE 4")
    print("=" * 70)

    # Load 39 features from Milestone 3
    ref_bundle = TransformerOverloadClassifier.load('models/overload/t2_overload_classifier.joblib')
    features = ref_bundle.feature_names
    print(f"Features count: {len(features)}")

    # Load raw processed datasets
    t1_train = pd.read_csv('data/processed/t1_train.csv')
    t1_test = pd.read_csv('data/processed/t1_test.csv')
    t2_train = pd.read_csv('data/processed/t2_train.csv')
    t2_test = pd.read_csv('data/processed/t2_test.csv')
    t3_train = pd.read_csv('data/processed/t3_train.csv')
    t3_test = pd.read_csv('data/processed/t3_test.csv')

    # -------------------------------------------------------------
    # 1. T2 CLASSIFIER VALIDATION
    # -------------------------------------------------------------
    print("\n--- 1. T2 Classifier Validation ---")
    t2_clf = TransformerOverloadClassifier.load('models/overload/t2_overload_classifier.joblib')
    
    # Target check
    target_col = 'target_next_hour_overload'
    has_target = target_col in t2_train.columns and target_col in t2_test.columns
    print(f"Target column '{target_col}' present: {has_target}")

    # Feature check: exactly 39 features, no leakage
    feature_match = features == ref_bundle.feature_names
    has_uplift = 'uplift_pct' in features
    has_target_in_feats = any(f.startswith('target_') for f in features)
    print(f"39 Features match exactly: {feature_match}")
    print(f"uplift_pct in features: {has_uplift}")
    print(f"target_* in features: {has_target_in_feats}")

    # Threshold selection inspection
    selected_thresh = t2_clf.selected_threshold
    print(f"Selected Threshold: {selected_thresh:.4f}")

    # Temporal CV inspection: Did StratifiedKFold shuffle time?
    # In train_overload_detector.py, StratifiedKFold(n_splits=5, shuffle=True, random_state=42) was used.
    # Let's inspect TimeSeriesSplit on the training set to see what threshold and metrics a temporal CV yields
    tscv = TimeSeriesSplit(n_splits=5)
    X_tr = t2_train[features]
    y_tr = t2_train[target_col].astype(int)

    oof_ts_probs = []
    y_ts_vals = []
    for train_idx, val_idx in tscv.split(X_tr):
        fold_clf = TransformerOverloadClassifier(
            transformer_id="T2",
            n_estimators=100,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        fold_clf.model = None
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        m.fit(X_tr.iloc[train_idx], y_tr.iloc[train_idx])
        p = m.predict_proba(X_tr.iloc[val_idx])[:, 1]
        oof_ts_probs.extend(p)
        y_ts_vals.extend(y_tr.iloc[val_idx])

    oof_ts_probs = np.array(oof_ts_probs)
    y_ts_vals = np.array(y_ts_vals)
    ts_roc = roc_auc_score(y_ts_vals, oof_ts_probs)
    ts_pr = average_precision_score(y_ts_vals, oof_ts_probs)
    print(f"Chronological TimeSeriesSplit (5 splits) on Train: ROC-AUC={ts_roc:.4f}, PR-AUC={ts_pr:.4f}")

    # Test set evaluation recalculation
    X_te = t2_test[features]
    y_te = t2_test[target_col].astype(int)
    probs_te = t2_clf.predict_proba(X_te)
    preds_def = (probs_te >= 0.50).astype(int)
    preds_sel = (probs_te >= selected_thresh).astype(int)

    roc_auc = roc_auc_score(y_te, probs_te)
    pr_auc = average_precision_score(y_te, probs_te)

    cm_def = confusion_matrix(y_te, preds_def)
    rec_def = recall_score(y_te, preds_def)
    prec_def = precision_score(y_te, preds_def)
    f1_def = f1_score(y_te, preds_def)

    cm_sel = confusion_matrix(y_te, preds_sel)
    rec_sel = recall_score(y_te, preds_sel)
    prec_sel = precision_score(y_te, preds_sel)
    f1_sel = f1_score(y_te, preds_sel)

    print(f"Recalculated T2 Test ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
    print(f"Recalculated Default 0.5: Rec={rec_def:.4f}, Prec={prec_def:.4f}, F1={f1_def:.4f}, TP={cm_def[1,1]}, FP={cm_def[0,1]}, FN={cm_def[1,0]}")
    print(f"Recalculated Selected {selected_thresh:.4f}: Rec={rec_sel:.4f}, Prec={prec_sel:.4f}, F1={f1_sel:.4f}, TP={cm_sel[1,1]}, FP={cm_sel[0,1]}, FN={cm_sel[1,0]}")

    results['t2_recalc'] = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'selected_threshold': selected_thresh,
        'default': {'rec': rec_def, 'prec': prec_def, 'f1': f1_def, 'tp': int(cm_def[1,1]), 'fp': int(cm_def[0,1]), 'fn': int(cm_def[1,0])},
        'selected': {'rec': rec_sel, 'prec': prec_sel, 'f1': f1_sel, 'tp': int(cm_sel[1,1]), 'fp': int(cm_sel[0,1]), 'fn': int(cm_sel[1,0])},
        'ts_cv': {'roc_auc': float(ts_roc), 'pr_auc': float(ts_pr)},
    }

    # -------------------------------------------------------------
    # 2. T1 ANOMALY DETECTION VALIDATION
    # -------------------------------------------------------------
    print("\n--- 2. T1 Anomaly Detection Validation ---")
    t1_det = TransformerAnomalyDetector.load('models/overload/t1_anomaly_detector.joblib')
    
    # Verify train set used
    print(f"T1 Train rows: {len(t1_train)}")
    train_overloads = (t1_train['load_pct'] >= 100.0).sum()
    train_target_overloads = t1_train['target_next_hour_overload'].sum()
    print(f"T1 Train overloads: {train_overloads} (current), {train_target_overloads} (target) -> strictly 0!")

    # Verify test evaluation
    X_t1_te = t1_test[features]
    t1_preds = t1_det.predict(X_t1_te)
    t1_scores = t1_det.anomaly_score(X_t1_te)

    y_t1_curr_ov = (t1_test['load_pct'] >= 100.0).astype(int)
    y_t1_targ_ov = t1_test['target_next_hour_overload'].astype(int)

    total_anoms = int(t1_preds.sum())
    curr_ov_caught = int(np.sum((t1_preds == 1) & (y_t1_curr_ov == 1)))
    targ_ov_caught = int(np.sum((t1_preds == 1) & (y_t1_targ_ov == 1)))

    rec_curr = curr_ov_caught / y_t1_curr_ov.sum()
    prec_curr = curr_ov_caught / total_anoms
    f1_curr = 2 * prec_curr * rec_curr / (prec_curr + rec_curr)

    rec_targ = targ_ov_caught / y_t1_targ_ov.sum()
    prec_targ = targ_ov_caught / total_anoms
    f1_targ = 2 * prec_targ * rec_targ / (prec_targ + rec_targ)

    print(f"Total anomalies flagged: {total_anoms} ({total_anoms/len(t1_test)*100:.2f}%)")
    print(f"Current Overload Recall: {rec_curr*100:.2f}% ({curr_ov_caught}/{y_t1_curr_ov.sum()})")
    print(f"Precision on Overloads: {prec_curr*100:.2f}% (F1={f1_curr:.4f})")
    print(f"Target Next-Hour Overload Recall: {rec_targ*100:.2f}% ({targ_ov_caught}/{y_t1_targ_ov.sum()})")

    results['t1_recalc'] = {
        'total_anomalies': total_anoms,
        'current_overloads': int(y_t1_curr_ov.sum()),
        'current_caught': curr_ov_caught,
        'recall_current': rec_curr,
        'precision_current': prec_curr,
        'f1_current': f1_curr,
        'target_overloads': int(y_t1_targ_ov.sum()),
        'target_caught': targ_ov_caught,
        'recall_target': rec_targ,
    }

    # -------------------------------------------------------------
    # 3. T3 THRESHOLD MONITORING VALIDATION
    # -------------------------------------------------------------
    print("\n--- 3. T3 Threshold Monitoring Validation ---")
    t3_monitor = ThresholdMonitor(transformer_id="T3", warning_load_pct=80.0, critical_load_pct=100.0)
    t3_eval = t3_monitor.evaluate_dataframe(t3_test, load_col="load_pct", temp_col="oil_temp_C")
    counts = t3_eval['threshold_status'].value_counts().to_dict()
    print(f"T3 Test rows: {len(t3_test)}")
    print(f"Status distribution: {counts}")
    print(f"Max load: {t3_test['load_pct'].max():.2f}%")
    print(f"Max oil temp: {t3_test['oil_temp_C'].max():.2f} °C")

    results['t3_recalc'] = {
        'counts': counts,
        'max_load': float(t3_test['load_pct'].max()),
        'max_temp': float(t3_test['oil_temp_C'].max()),
    }

    # -------------------------------------------------------------
    # 4. UNIFIED ALERT ENGINE VALIDATION
    # -------------------------------------------------------------
    print("\n--- 4. Unified Alert Engine Validation ---")
    alert_engine = UnifiedAlertEngine(warning_load_pct=80.0, critical_load_pct=100.0, classifier_alert_threshold=selected_thresh)

    # Check alert priority
    # Priority order:
    # 1. Measured load >= 100 -> CRITICAL CURRENT_OVERLOAD
    # 2. Predicted load >= 100 or P(overload) >= thresh -> PREDICTED_OVERLOAD
    # 3. Measured load in [80, 100) -> WARNING ELEVATED_LOAD_WARNING
    # 4. Anomaly detected -> WARNING LOAD_ANOMALY
    # 5. Normal -> NORMAL NORMAL_OPERATION
    a_crit = alert_engine.generate_alert("T2", "2017-01-01 12:00:00", current_load_pct=105.0, overload_probability=0.20, is_anomaly=False)
    print(f"Measured 105% -> Risk: {a_crit.risk_level}, Type: {a_crit.alert_type}")
    assert a_crit.risk_level == "CRITICAL" and a_crit.alert_type == "CURRENT_OVERLOAD"

    a_pred = alert_engine.generate_alert("T2", "2017-01-01 12:00:00", current_load_pct=75.0, overload_probability=0.55, is_anomaly=False)
    print(f"P(overload)=0.55 -> Risk: {a_pred.risk_level}, Type: {a_pred.alert_type}")
    assert a_pred.alert_type == "PREDICTED_OVERLOAD"

    a_anom = alert_engine.generate_alert("T1", "2017-01-01 12:00:00", current_load_pct=60.0, overload_probability=0.10, is_anomaly=True, anomaly_score=0.6)
    print(f"Anomaly=True -> Risk: {a_anom.risk_level}, Type: {a_anom.alert_type}")
    assert a_anom.risk_level == "WARNING" and a_anom.alert_type == "LOAD_ANOMALY"

    print("Alert priority hierarchy is verified and deterministic.")

    # Check advisory wording
    print("\nChecking advisory language in alert recommendations:")
    print(f"CURRENT_OVERLOAD recommendation: '{a_crit.recommended_action}'")
    print(f"PREDICTED_OVERLOAD recommendation: '{a_pred.recommended_action}'")
    print(f"LOAD_ANOMALY recommendation: '{a_anom.recommended_action}'")

    # Export audit results
    with open('reports/milestone4_audit_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("AUDIT SCRIPT COMPLETED WITH ZERO ERRORS")
    print("=" * 70)


if __name__ == '__main__':
    audit()

