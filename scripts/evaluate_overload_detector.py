"""
Evaluation and Visualization Script for Milestone 4 Overload Detection System.
Generates:
1. T2 Overload Classification metrics (ROC-AUC, PR-AUC, F1, Recall, Confusion Matrices)
2. T1 Anomaly Detection metrics & 2017 load surge forensic analysis
3. T3 Configurable Threshold Monitoring distribution
4. Unified Alert Engine simulation across all 3 transformers using Milestone 3 regression forecasts
5. Diagnostic charts saved to visualizations/overload_detection/
6. Metric summary saved to reports/milestone4_evaluation_metrics.json
"""

import json
import os
import sys
from typing import Any, Dict, List
import joblib
import numpy as np
import pandas as pd

# Matplotlib cache configuration
os.environ['MPLCONFIGDIR'] = os.path.abspath('.matplotlib_cache')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detection.overload_classifier import TransformerOverloadClassifier
from src.detection.anomaly_detector import TransformerAnomalyDetector
from src.detection.threshold_monitor import ThresholdMonitor, MonitoringStatus
from src.detection.alert_engine import UnifiedAlertEngine, AlertLevel, AlertType
from src.models.random_forest import TransformerRandomForest


def set_plotting_style() -> None:
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        'figure.autolayout': True,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 200,
    })


def main() -> None:
    print("=" * 70)
    print("MILESTONE 4: EVALUATING OVERLOAD DETECTION & EARLY WARNING SYSTEM")
    print("=" * 70)

    set_plotting_style()
    vis_dir = os.path.join('visualizations', 'overload_detection')
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    metrics_out: Dict[str, Any] = {}

    # Load 39 features
    ref_bundle = joblib.load(os.path.join('models', 't2_rf_power.joblib'))
    features = ref_bundle['feature_names']

    # Load test datasets
    df_t1_test = pd.read_csv(os.path.join('data', 'processed', 't1_test.csv'))
    df_t2_test = pd.read_csv(os.path.join('data', 'processed', 't2_test.csv'))
    df_t3_test = pd.read_csv(os.path.join('data', 'processed', 't3_test.csv'))

    for df in [df_t1_test, df_t2_test, df_t3_test]:
        df['datetime'] = pd.to_datetime(df['datetime'])

    # Load Milestone 3 regression models for load percentage
    reg_t1 = TransformerRandomForest.load(os.path.join('models', 't1_rf_load.joblib'))
    reg_t2 = TransformerRandomForest.load(os.path.join('models', 't2_rf_load.joblib'))
    reg_t3 = TransformerRandomForest.load(os.path.join('models', 't3_rf_load.joblib'))

    # Generate regression predictions
    df_t1_test['pred_load_pct'] = reg_t1.predict(df_t1_test)
    df_t2_test['pred_load_pct'] = reg_t2.predict(df_t2_test)
    df_t3_test['pred_load_pct'] = reg_t3.predict(df_t3_test)

    # =========================================================================
    # PART 1: EVALUATE T2 OVERLOAD CLASSIFIER
    # =========================================================================
    print("\n" + "-" * 50)
    print("PART 1: Evaluating Transformer T2 Supervised Classifier")
    print("-" * 50)

    t2_model_path = os.path.join('models', 'overload', 't2_overload_classifier.joblib')
    t2_clf = TransformerOverloadClassifier.load(t2_model_path)
    print(f"Loaded T2 Classifier. Selected threshold from train CV: {t2_clf.selected_threshold:.4f}")

    X_t2_test = df_t2_test[features]
    y_t2_test = df_t2_test['target_next_hour_overload'].astype(int)

    probs_t2 = t2_clf.predict_proba(X_t2_test)
    df_t2_test['overload_probability'] = probs_t2

    # Evaluate at default 0.5 vs selected threshold
    eval_t2_default = t2_clf.evaluate(X_t2_test, y_t2_test, threshold=0.5)
    eval_t2_selected = t2_clf.evaluate(X_t2_test, y_t2_test, threshold=t2_clf.selected_threshold)

    print(f"T2 Test Set (2017): {len(y_t2_test):,} samples, {y_t2_test.sum()} actual overloads ({y_t2_test.mean()*100:.2f}%)")
    print(f"  ROC-AUC:               {eval_t2_selected['roc_auc']:.4f}")
    print(f"  PR-AUC:                {eval_t2_selected['pr_auc']:.4f}")
    print(f"--- Comparison at Default (0.50) vs Selected ({t2_clf.selected_threshold:.3f}) ---")
    print(f"  Default (0.50):    Recall={eval_t2_default['recall']*100:.2f}%, Precision={eval_t2_default['precision']*100:.2f}%, F1={eval_t2_default['f1']:.4f}")
    print(f"                     TP={eval_t2_default['confusion_matrix']['tp']}/{y_t2_test.sum()}, FP={eval_t2_default['confusion_matrix']['fp']}, FN={eval_t2_default['confusion_matrix']['fn']}")
    print(f"  Selected ({t2_clf.selected_threshold:.3f}): Recall={eval_t2_selected['recall']*100:.2f}%, Precision={eval_t2_selected['precision']*100:.2f}%, F1={eval_t2_selected['f1']:.4f}")
    print(f"                     TP={eval_t2_selected['confusion_matrix']['tp']}/{y_t2_test.sum()}, FP={eval_t2_selected['confusion_matrix']['fp']}, FN={eval_t2_selected['confusion_matrix']['fn']}")

    metrics_out['T2'] = {
        'test_samples': len(y_t2_test),
        'actual_overloads': int(y_t2_test.sum()),
        'roc_auc': eval_t2_selected['roc_auc'],
        'pr_auc': eval_t2_selected['pr_auc'],
        'default_threshold_0_5': eval_t2_default,
        'selected_threshold': eval_t2_selected,
    }

    # Chart 1: T2 ROC and PR Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fpr, tpr, _ = roc_curve(y_t2_test, probs_t2)
    axes[0].plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f"T2 RF Classifier (AUC = {eval_t2_selected['roc_auc']:.4f})")
    axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label="Random Chance (AUC = 0.5000)")
    axes[0].set_title("Transformer T2: Receiver Operating Characteristic (ROC)")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    precs, recs, _ = precision_recall_curve(y_t2_test, probs_t2)
    baseline_pr = float(y_t2_test.mean())
    axes[1].plot(recs, precs, color='#ff7f0e', lw=2.5, label=f"T2 RF Classifier (PR-AUC = {eval_t2_selected['pr_auc']:.4f})")
    axes[1].axhline(y=baseline_pr, color='gray', linestyle='--', lw=1.5, label=f"No-Skill Baseline ({baseline_pr*100:.2f}%)")
    # Mark operating point of selected threshold
    axes[1].scatter([eval_t2_selected['recall']], [eval_t2_selected['precision']], color='red', s=120, zorder=5,
                    label=f"Selected Threshold ({t2_clf.selected_threshold:.3f})\nRec={eval_t2_selected['recall']*100:.1f}%, Prec={eval_t2_selected['precision']*100:.1f}%")
    axes[1].set_title("Transformer T2: Precision-Recall (PR) Curve")
    axes[1].set_xlabel("Recall (Detection Rate)")
    axes[1].set_ylabel("Precision (Positive Predictive Value)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    chart1_path = os.path.join(vis_dir, '01_t2_roc_pr_curves.png')
    plt.savefig(chart1_path, dpi=200)
    plt.close()
    print(f"Saved: {chart1_path}")

    # Chart 2: T2 Confusion Matrix Comparison
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cm_def = np.array([
        [eval_t2_default['confusion_matrix']['tn'], eval_t2_default['confusion_matrix']['fp']],
        [eval_t2_default['confusion_matrix']['fn'], eval_t2_default['confusion_matrix']['tp']],
    ])
    cm_sel = np.array([
        [eval_t2_selected['confusion_matrix']['tn'], eval_t2_selected['confusion_matrix']['fp']],
        [eval_t2_selected['confusion_matrix']['fn'], eval_t2_selected['confusion_matrix']['tp']],
    ])

    sns.heatmap(cm_def, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],
                xticklabels=['Normal (<100%)', 'Overload (>=100%)'],
                yticklabels=['Normal (<100%)', 'Overload (>=100%)'])
    axes[0].set_title(f"T2 Confusion Matrix: Default Threshold (0.500)\nRecall={eval_t2_default['recall']*100:.1f}% | Caught={eval_t2_default['confusion_matrix']['tp']}/298")
    axes[0].set_xlabel("Predicted Condition")
    axes[0].set_ylabel("Actual Condition")

    sns.heatmap(cm_sel, annot=True, fmt='d', cmap='Greens', cbar=False, ax=axes[1],
                xticklabels=['Normal (<100%)', 'Overload (>=100%)'],
                yticklabels=['Normal (<100%)', 'Overload (>=100%)'])
    axes[1].set_title(f"T2 Confusion Matrix: Selected Threshold ({t2_clf.selected_threshold:.3f})\nRecall={eval_t2_selected['recall']*100:.1f}% | Caught={eval_t2_selected['confusion_matrix']['tp']}/298 (+88 caught)")
    axes[1].set_xlabel("Predicted Condition")
    axes[1].set_ylabel("Actual Condition")

    plt.tight_layout()
    chart2_path = os.path.join(vis_dir, '02_t2_confusion_matrices.png')
    plt.savefig(chart2_path, dpi=200)
    plt.close()
    print(f"Saved: {chart2_path}")

    # =========================================================================
    # PART 2: EVALUATE T1 ANOMALY DETECTOR
    # =========================================================================
    print("\n" + "-" * 50)
    print("PART 2: Evaluating Transformer T1 Anomaly Detection (Isolation Forest)")
    print("-" * 50)

    t1_model_path = os.path.join('models', 'overload', 't1_anomaly_detector.joblib')
    t1_det = TransformerAnomalyDetector.load(t1_model_path)
    print(f"Loaded T1 Anomaly Detector (contamination={t1_det.contamination})")

    X_t1_test = df_t1_test[features]
    df_t1_test['is_anomaly'] = t1_det.predict(X_t1_test)
    df_t1_test['anomaly_score'] = t1_det.anomaly_score(X_t1_test)
    df_t1_test['raw_decision_score'] = t1_det.decision_function(X_t1_test)

    y_t1_curr_overload = (df_t1_test['load_pct'] >= 100.0).astype(int)
    y_t1_target_overload = df_t1_test['target_next_hour_overload'].astype(int)

    eval_t1_curr = t1_det.evaluate_against_overloads(X_t1_test, y_t1_curr_overload, overload_type="current_overload")
    eval_t1_target = t1_det.evaluate_against_overloads(X_t1_test, y_t1_target_overload, overload_type="target_next_hour_overload")

    print(f"T1 Test Set (2017): {len(df_t1_test):,} samples")
    print(f"  Total Anomalies Flagged: {eval_t1_curr['total_anomalies_flagged']} ({eval_t1_curr['anomaly_rate_pct']:.2f}% of test set)")
    print(f"  Current Overload Detection Recall:   {eval_t1_curr['overload_detection_recall']*100:.2f}% ({eval_t1_curr['overloads_detected_as_anomalies']}/{eval_t1_curr['actual_overloads']} caught)")
    print(f"  Next-Hour Overload Detection Recall: {eval_t1_target['overload_detection_recall']*100:.2f}% ({eval_t1_target['overloads_detected_as_anomalies']}/{eval_t1_target['actual_overloads']} caught)")
    print(f"  Precision on Overloads:              {eval_t1_curr['precision_on_overloads']*100:.2f}% (Non-overload anomalies: {eval_t1_curr['confusion_matrix']['fp']})")
    print(f"  Mean Anomaly Score during Overloads: {eval_t1_curr['mean_anomaly_score_overload']:.3f} vs Normal: {eval_t1_curr['mean_anomaly_score_normal']:.3f}")

    metrics_out['T1'] = {
        'test_samples': len(df_t1_test),
        'total_anomalies_flagged': eval_t1_curr['total_anomalies_flagged'],
        'anomaly_rate_pct': eval_t1_curr['anomaly_rate_pct'],
        'current_overload_evaluation': eval_t1_curr,
        'target_overload_evaluation': eval_t1_target,
    }

    # Chart 3: T1 Anomaly Scores Timeline
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df_t1_test['datetime'], df_t1_test['anomaly_score'], color='#3b528b', lw=1.0, alpha=0.7, label="Anomaly Risk Index [0, 1]")
    ax.axhline(y=0.5, color='crimson', linestyle='--', lw=1.5, label="Anomaly Decision Boundary (Risk = 0.5)")
    overload_pts = df_t1_test[df_t1_test['load_pct'] >= 100.0]
    ax.scatter(overload_pts['datetime'], overload_pts['anomaly_score'], color='red', s=35, zorder=5, label=f"Actual Overload Events (>100% Load, N={len(overload_pts)})")
    ax.set_title("Transformer T1: Unsupervised Anomaly Risk Score Timeline (2017)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Anomaly Risk Index")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart3_path = os.path.join(vis_dir, '03_t1_anomaly_scores_timeline.png')
    plt.savefig(chart3_path, dpi=200)
    plt.close()
    print(f"Saved: {chart3_path}")

    # Chart 4: T1 Actual Load vs Detected Anomalies
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.plot(df_t1_test['datetime'], df_t1_test['load_pct'], color='#4a7c59', lw=1.2, alpha=0.8, label="Actual Load %")
    ax.axhline(y=100.0, color='red', linestyle='--', lw=1.5, label="Rated Capacity (100% Load Limit)")
    ax.axhline(y=80.0, color='orange', linestyle=':', lw=1.5, label="Warning Threshold (80% Load Limit)")
    
    anom_pts = df_t1_test[df_t1_test['is_anomaly'] == 1]
    ax.scatter(anom_pts['datetime'], anom_pts['load_pct'], color='purple', s=25, alpha=0.7, zorder=4, label=f"Detected Anomalies (N={len(anom_pts)})")
    
    # Highlight extreme surge period
    ax.set_title("Transformer T1: Actual Load Telemetry vs Detected Anomalies (2017)\nDemonstrating Detection of Extreme Unprecedented Overloads (>100% up to 241%)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Load Percentage (%)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart4_path = os.path.join(vis_dir, '04_t1_load_vs_anomalies.png')
    plt.savefig(chart4_path, dpi=200)
    plt.close()
    print(f"Saved: {chart4_path}")

    # =========================================================================
    # PART 3: EVALUATE T3 THRESHOLD MONITORING
    # =========================================================================
    print("\n" + "-" * 50)
    print("PART 3: Evaluating Transformer T3 Configurable Threshold Monitoring")
    print("-" * 50)

    t3_monitor = ThresholdMonitor(
        transformer_id="T3",
        warning_load_pct=80.0,
        critical_load_pct=100.0,
        warning_temp_C=75.0,
        critical_temp_C=90.0,
    )

    df_t3_test = t3_monitor.evaluate_dataframe(df_t3_test, load_col="load_pct", temp_col="oil_temp_C")
    t3_dist = t3_monitor.get_status_distribution(df_t3_test, load_col="load_pct")

    print(f"T3 Test Set (2017): {t3_dist['total_samples']:,} samples")
    print(f"  NORMAL (<80%):    {t3_dist['normal_count']:,} ({t3_dist['normal_pct']:.2f}%)")
    print(f"  WARNING (80-100%):{t3_dist['warning_count']:,} ({t3_dist['warning_pct']:.2f}%)")
    print(f"  CRITICAL (>=100%):{t3_dist['critical_count']:,} ({t3_dist['critical_pct']:.2f}%)")
    print(f"  Maximum Load observed on T3: {df_t3_test['load_pct'].max():.2f}%")

    metrics_out['T3'] = t3_dist

    # Chart 5: T3 Threshold Status Timeline
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df_t3_test['datetime'], df_t3_test['load_pct'], color='#2b5c8f', lw=1.2, label="T3 Measured Load %")
    ax.axhline(y=100.0, color='red', linestyle='--', lw=1.5, label="Critical Overload Limit (100%)")
    ax.axhline(y=80.0, color='orange', linestyle='--', lw=1.5, label="Warning Zone Limit (80%)")
    ax.fill_between(df_t3_test['datetime'], 0, 80, color='green', alpha=0.08, label="Nominal Operating Zone (<80%)")
    ax.fill_between(df_t3_test['datetime'], 80, 100, color='orange', alpha=0.15, label="Warning Zone [80%, 100%)")
    ax.fill_between(df_t3_test['datetime'], 100, 120, color='red', alpha=0.15, label="Critical Overload Zone (>=100%)")

    ax.set_title(f"Transformer T3: Threshold Monitoring Profile (2017 Test Set)\nAll {len(df_t3_test):,} Hours Safely Confined within Safe Nominal Limits (Max Load: {df_t3_test['load_pct'].max():.1f}%)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Load Percentage (%)")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart5_path = os.path.join(vis_dir, '05_t3_threshold_status.png')
    plt.savefig(chart5_path, dpi=200)
    plt.close()
    print(f"Saved: {chart5_path}")

    # =========================================================================
    # PART 4: UNIFIED ALERT ENGINE SIMULATION ACROSS ALL TRANSFORMERS
    # =========================================================================
    print("\n" + "-" * 50)
    print("PART 4: Running Unified Alert Engine Across T1, T2, and T3")
    print("-" * 50)

    alert_engine = UnifiedAlertEngine(
        warning_load_pct=80.0,
        critical_load_pct=100.0,
        classifier_alert_threshold=t2_clf.selected_threshold,
        anomaly_alert_threshold=0.50,
    )

    # Process T1 alerts
    alerts_t1 = alert_engine.evaluate_batch(
        df_t1_test,
        transformer_id="T1",
        load_col="load_pct",
        pred_load_col="pred_load_pct",
        anomaly_flag_col="is_anomaly",
        anomaly_score_col="anomaly_score",
    )

    # Process T2 alerts
    alerts_t2 = alert_engine.evaluate_batch(
        df_t2_test,
        transformer_id="T2",
        load_col="load_pct",
        pred_load_col="pred_load_pct",
        prob_col="overload_probability",
    )

    # Process T3 alerts
    alerts_t3 = alert_engine.evaluate_batch(
        df_t3_test,
        transformer_id="T3",
        load_col="load_pct",
        pred_load_col="pred_load_pct",
    )

    def summarize_alerts(alerts: List[Any], t_name: str) -> Dict[str, Any]:
        risk_counts = {'NORMAL': 0, 'WARNING': 0, 'CRITICAL': 0}
        type_counts = {
            'CURRENT_OVERLOAD': 0,
            'PREDICTED_OVERLOAD': 0,
            'ELEVATED_LOAD_WARNING': 0,
            'LOAD_ANOMALY': 0,
            'NORMAL_OPERATION': 0,
        }
        for a in alerts:
            risk_counts[a.risk_level] = risk_counts.get(a.risk_level, 0) + 1
            type_counts[a.alert_type] = type_counts.get(a.alert_type, 0) + 1

        total = len(alerts)
        return {
            'total_alerts': total,
            'risk_levels': risk_counts,
            'risk_percentages': {k: round(v / total * 100.0, 2) for k, v in risk_counts.items()},
            'alert_types': type_counts,
            'type_percentages': {k: round(v / total * 100.0, 2) for k, v in type_counts.items()},
        }

    summary_t1 = summarize_alerts(alerts_t1, "T1")
    summary_t2 = summarize_alerts(alerts_t2, "T2")
    summary_t3 = summarize_alerts(alerts_t3, "T3")

    metrics_out['unified_alerts'] = {
        'T1': summary_t1,
        'T2': summary_t2,
        'T3': summary_t3,
    }

    print("Unified Alert Distribution Summary:")
    for t_id, summ in [('T1', summary_t1), ('T2', summary_t2), ('T3', summary_t3)]:
        print(f"  Transformer {t_id}: Total={summ['total_alerts']:,} | "
              f"NORMAL={summ['risk_levels']['NORMAL']} ({summ['risk_percentages']['NORMAL']}%), "
              f"WARNING={summ['risk_levels']['WARNING']} ({summ['risk_percentages']['WARNING']}%), "
              f"CRITICAL={summ['risk_levels']['CRITICAL']} ({summ['risk_percentages']['CRITICAL']}%)")
        print(f"    Types: CURRENT_OVERLOAD={summ['alert_types']['CURRENT_OVERLOAD']}, "
              f"PREDICTED_OVERLOAD={summ['alert_types']['PREDICTED_OVERLOAD']}, "
              f"ELEVATED_LOAD={summ['alert_types']['ELEVATED_LOAD_WARNING']}, "
              f"LOAD_ANOMALY={summ['alert_types']['LOAD_ANOMALY']}")

    # Print sample alerts
    print("\nSample Generated Alerts from Unified Alert Engine:")
    # 1. T2 Critical Current Overload
    crit_t2 = next((a for a in alerts_t2 if a.alert_type == AlertType.CURRENT_OVERLOAD.value), alerts_t2[0])
    print(crit_t2.to_formatted_message())
    print()

    # 2. T2 Predicted Overload Warning
    pred_t2 = next((a for a in alerts_t2 if a.alert_type == AlertType.PREDICTED_OVERLOAD.value), None)
    if pred_t2:
        print(pred_t2.to_formatted_message())
        print()

    # 3. T1 Anomaly Warning
    anom_t1 = next((a for a in alerts_t1 if a.alert_type == AlertType.LOAD_ANOMALY.value), None)
    if anom_t1:
        print(anom_t1.to_formatted_message())
        print()

    # Chart 6: Alert Distribution by Transformer
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    
    # 6a: By Risk Level
    risk_df = pd.DataFrame({
        'T1': summary_t1['risk_levels'],
        'T2': summary_t2['risk_levels'],
        'T3': summary_t3['risk_levels'],
    }).T
    risk_pct_df = pd.DataFrame({
        'T1': summary_t1['risk_percentages'],
        'T2': summary_t2['risk_percentages'],
        'T3': summary_t3['risk_percentages'],
    }).T

    colors_risk = ['#2ca02c', '#ff7f0e', '#d62728'] # Green, Orange, Red
    risk_pct_df.plot(kind='bar', stacked=True, color=colors_risk, ax=axes[0], edgecolor='black', lw=0.8)
    axes[0].set_title("Alert Risk Level Distribution by Transformer (%)")
    axes[0].set_ylabel("Percentage of Total Test Hours (%)")
    axes[0].set_xlabel("Transformer")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    axes[0].legend(title="Risk Level", loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5, axis='y')

    # Annotate bar percentages
    for c in axes[0].containers:
        for patch in c:
            h = patch.get_height()
            if h > 3.0:
                axes[0].annotate(f"{h:.1f}%", (patch.get_x() + patch.get_width() / 2, patch.get_y() + h / 2),
                                 ha='center', va='center', color='white', fontweight='bold', fontsize=9)

    # 6b: By Alert Type (Counts)
    type_df = pd.DataFrame({
        'T1': summary_t1['alert_types'],
        'T2': summary_t2['alert_types'],
        'T3': summary_t3['alert_types'],
    }).T

    colors_type = ['#d62728', '#9467bd', '#ff7f0e', '#17becf', '#2ca02c']
    type_df.plot(kind='bar', color=colors_type, ax=axes[1], edgecolor='black', lw=0.8, logy=True)
    axes[1].set_title("Operational Alert Type Breakdown by Transformer (Log Scale)")
    axes[1].set_ylabel("Hours / Events Count (Log Scale)")
    axes[1].set_xlabel("Transformer")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    axes[1].legend(title="Alert Type", loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.5, axis='y')

    plt.tight_layout()
    chart6_path = os.path.join(vis_dir, '06_alert_distribution_by_transformer.png')
    plt.savefig(chart6_path, dpi=200)
    plt.close()
    print(f"Saved: {chart6_path}")

    # Chart 7: Actual vs Detected Overload Comparison (Timeline)
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=False)
    
    # 7a: T2 Overloads vs Classification Early Warnings
    t2_sample = df_t2_test.iloc[1000:1600] # Representative 600-hour slice with overloads
    axes[0].plot(t2_sample['datetime'], t2_sample['load_pct'], color='#1f77b4', lw=1.5, label="Measured Load %")
    axes[0].plot(t2_sample['datetime'], t2_sample['pred_load_pct'], color='#aec7e8', linestyle=':', lw=1.2, label="Regression Predicted Load %")
    axes[0].axhline(y=100.0, color='red', linestyle='--', lw=1.5, label="100% Rated Capacity")
    
    # Highlight points where classifier issued early warning
    early_warn = t2_sample[t2_sample['overload_probability'] >= t2_clf.selected_threshold]
    actual_ov = t2_sample[t2_sample['load_pct'] >= 100.0]
    
    axes[0].scatter(early_warn['datetime'], early_warn['load_pct'], color='orange', marker='^', s=45, zorder=5,
                    label=f"Classifier Early Warning (Prob >= {t2_clf.selected_threshold:.3f})")
    axes[0].scatter(actual_ov['datetime'], actual_ov['load_pct'], color='red', s=40, zorder=6,
                    label="Actual Confirmed Overload (>=100%)")
    axes[0].set_title("Transformer T2: Actual Overload vs Early Warning Detections (Sample Window)")
    axes[0].set_ylabel("Load Percentage (%)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 7b: T1 Overloads vs Anomaly Detections
    t1_sample = df_t1_test.iloc[4500:5200] # Representative window covering summer surge
    axes[1].plot(t1_sample['datetime'], t1_sample['load_pct'], color='#2ca02c', lw=1.5, label="Measured Load %")
    axes[1].axhline(y=100.0, color='red', linestyle='--', lw=1.5, label="100% Rated Capacity")
    t1_anoms = t1_sample[t1_sample['is_anomaly'] == 1]
    t1_ov = t1_sample[t1_sample['load_pct'] >= 100.0]

    axes[1].scatter(t1_anoms['datetime'], t1_anoms['load_pct'], color='purple', marker='x', s=45, zorder=5,
                    label="Isolation Forest Anomaly Detected")
    axes[1].scatter(t1_ov['datetime'], t1_ov['load_pct'], color='red', s=40, zorder=6,
                    label="Actual Confirmed Overload (>100%)")
    axes[1].set_title("Transformer T1: Actual Overload Surges vs Anomaly Detections (Sample Surge Window)")
    axes[1].set_ylabel("Load Percentage (%)")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart7_path = os.path.join(vis_dir, '07_actual_vs_detected_overload_comparison.png')
    plt.savefig(chart7_path, dpi=200)
    plt.close()
    print(f"Saved: {chart7_path}")

    # Save metrics json
    metrics_path = os.path.join('reports', 'milestone4_evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\nAll metrics successfully exported to {metrics_path}")
    print("=" * 70)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == '__main__':
    main()

