#!/usr/bin/env python
"""
Milestone 3: Model Evaluation Script.
Evaluates all 6 trained Random Forest models on the unseen 2017 test datasets.
Generates evaluation metrics (MAE, RMSE, R², MAPE, wMAPE), residual analyses,
feature importance rankings, and high-resolution diagnostic charts.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Matplotlib local cache configuration
os.environ['MPLCONFIGDIR'] = os.path.abspath('.matplotlib_cache')
import matplotlib.pyplot as plt
import seaborn as sns

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.random_forest import TransformerRandomForest, calculate_metrics

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
VIS_DIR = "visualizations"
REPORTS_DIR = "reports"

os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Aesthetic styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

colors = {'T1': '#2563eb', 'T2': '#dc2626', 'T3': '#16a34a'}

def evaluate_all():
    print("=" * 70)
    print("MILESTONE 3: MODEL EVALUATION & DIAGNOSTICS PIPELINE")
    print("=" * 70)

    transformers = ['t1', 't2', 't3']
    results = {}
    preds_dict = {}

    for t in transformers:
        t_upper = t.upper()
        test_path = os.path.join(PROCESSED_DIR, f"{t}_test.csv")
        print(f"\n[{t_upper}] Loading 2017 test data from {test_path}...")
        df_test = pd.read_csv(test_path)
        df_test['datetime_dt'] = pd.to_datetime(df_test['datetime'])

        power_model_path = os.path.join(MODELS_DIR, f"{t}_rf_power.joblib")
        load_model_path = os.path.join(MODELS_DIR, f"{t}_rf_load.joblib")

        print(f"[{t_upper}] Loading models: {power_model_path} and {load_model_path}")
        power_model = TransformerRandomForest.load(power_model_path)
        load_model = TransformerRandomForest.load(load_model_path)

        # Generate predictions
        X_test = df_test[power_model.feature_names]
        power_preds = power_model.predict(X_test)
        load_preds = load_model.predict(X_test)

        y_power_true = df_test['target_next_hour_power_W'].values
        y_load_true = df_test['target_next_hour_load_pct'].values

        # Calculate metrics overall
        power_metrics = calculate_metrics(y_power_true, power_preds)
        load_metrics = calculate_metrics(y_load_true, load_preds)

        # Evaluate on normal vs festival subsets
        is_fest = df_test['is_festival'] == 1
        fest_power_metrics = calculate_metrics(y_power_true[is_fest], power_preds[is_fest]) if is_fest.sum() > 0 else {}
        norm_power_metrics = calculate_metrics(y_power_true[~is_fest], power_preds[~is_fest])

        print(f"[{t_upper}] Power (W) Metrics:  MAE={power_metrics['mae']:.1f} W ({power_metrics['mae']/1000:.2f} kW) | RMSE={power_metrics['rmse']:.1f} W ({power_metrics['rmse']/1000:.2f} kW) | R²={power_metrics['r2']:.4f} | wMAPE={power_metrics['wmape']:.2f}%")
        print(f"[{t_upper}] Load (%) Metrics:   MAE={load_metrics['mae']:.2f}% | RMSE={load_metrics['rmse']:.2f}% | R²={load_metrics['r2']:.4f} | wMAPE={load_metrics['wmape']:.2f}%")

        results[t_upper] = {
            'test_rows': len(df_test),
            'power': {
                'overall': power_metrics,
                'normal_days': norm_power_metrics,
                'festival_days': fest_power_metrics,
                'mae_kW': round(power_metrics['mae'] / 1000.0, 3),
                'rmse_kW': round(power_metrics['rmse'] / 1000.0, 3),
            },
            'load': {
                'overall': load_metrics
            },
            'top_features_power': power_model.get_feature_importances().head(10).to_dict(orient='records'),
            'top_features_load': load_model.get_feature_importances().head(10).to_dict(orient='records')
        }

        preds_dict[t_upper] = {
            'df': df_test,
            'y_power_true': y_power_true,
            'y_power_pred': power_preds,
            'y_load_true': y_load_true,
            'y_load_pred': load_preds,
            'power_model': power_model,
            'load_model': load_model
        }

    # Save metrics JSON
    metrics_path = os.path.join(REPORTS_DIR, "milestone3_evaluation_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation metrics to {metrics_path}")

    # Generate Visualizations
    generate_charts(preds_dict)

def generate_charts(preds_dict):
    print("\nGenerating Diagnostic Charts...")

    # 1. Actual vs Predicted Power (Scatter + Sample Time Series)
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    for i, (t_name, data) in enumerate(preds_dict.items()):
        y_true = data['y_power_true'] / 1000.0 # in kW
        y_pred = data['y_power_pred'] / 1000.0 # in kW
        df = data['df']

        # (a) Scatter Plot
        ax_scat = axes[i, 0]
        ax_scat.scatter(y_true, y_pred, alpha=0.25, color=colors[t_name], s=12)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax_scat.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, label='Ideal 1:1 Fit')
        ax_scat.set_title(f"{t_name} Power: Actual vs Predicted (2017 Test Set)", fontsize=11, fontweight='bold')
        ax_scat.set_xlabel("Actual Active Power (kW)", fontsize=10)
        ax_scat.set_ylabel("Predicted Active Power (kW)", fontsize=10)
        ax_scat.legend(loc='upper left', frameon=True)
        ax_scat.grid(True, ls=':', alpha=0.6)

        # (b) Time Series Excerpt (7 days: 168 hours in July 2017)
        ax_ts = axes[i, 1]
        sample_slice = slice(4000, 4168)
        sample_time = df['datetime_dt'].iloc[sample_slice]
        ax_ts.plot(sample_time, y_true[sample_slice], label='Actual Power (kW)', color='#334155', lw=2.0)
        ax_ts.plot(sample_time, y_pred[sample_slice], label='Predicted Power (kW)', color=colors[t_name], lw=2.0, ls='--')
        ax_ts.set_title(f"{t_name} Power: 1-Week Sample Forecast Tracking (July 2017)", fontsize=11, fontweight='bold')
        ax_ts.set_xlabel("Timestamp", fontsize=10)
        ax_ts.set_ylabel("Active Power (kW)", fontsize=10)
        ax_ts.legend(loc='upper right', frameon=True)
        ax_ts.grid(True, ls=':', alpha=0.6)
        plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=30, ha='right')

    plt.tight_layout()
    chart1 = os.path.join(VIS_DIR, '07_actual_vs_predicted_power.png')
    fig.savefig(chart1, dpi=200)
    plt.close(fig)
    print(f"  Saved {chart1}")

    # 2. Actual vs Predicted Load Percentage
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for i, (t_name, data) in enumerate(preds_dict.items()):
        ax = axes[i]
        y_true = data['y_load_true']
        y_pred = data['y_load_pred']
        ax.scatter(y_true, y_pred, alpha=0.25, color=colors[t_name], s=12)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, label='Ideal 1:1 Fit')
        ax.axhline(100.0, color='red', ls=':', lw=1.2, label='100% Rated')
        ax.axvline(100.0, color='red', ls=':', lw=1.2)
        ax.set_title(f"{t_name} Load Percentage: Actual vs Predicted", fontsize=12, fontweight='bold')
        ax.set_xlabel("Actual Load (%)", fontsize=10)
        ax.set_ylabel("Predicted Load (%)", fontsize=10)
        ax.legend(loc='upper left', frameon=True)
        ax.grid(True, ls=':', alpha=0.6)

    plt.tight_layout()
    chart2 = os.path.join(VIS_DIR, '08_actual_vs_predicted_load.png')
    fig.savefig(chart2, dpi=200)
    plt.close(fig)
    print(f"  Saved {chart2}")

    # 3. Residual Distribution & Analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for i, (t_name, data) in enumerate(preds_dict.items()):
        y_true = data['y_power_true'] / 1000.0
        y_pred = data['y_power_pred'] / 1000.0
        residuals = y_true - y_pred

        # Residual histogram / KDE
        ax_hist = axes[0, i]
        sns.histplot(residuals, kde=True, ax=ax_hist, color=colors[t_name], bins=40)
        ax_hist.axvline(0, color='black', ls='--', lw=1.2)
        ax_hist.set_title(f"{t_name} Power Residuals Distribution", fontsize=12, fontweight='bold')
        ax_hist.set_xlabel("Residual Error (Actual - Predicted, kW)", fontsize=10)
        ax_hist.set_ylabel("Count", fontsize=10)

        # Residual vs Predicted Plot
        ax_res = axes[1, i]
        ax_res.scatter(y_pred, residuals, alpha=0.2, color=colors[t_name], s=10)
        ax_res.axhline(0, color='black', ls='--', lw=1.2)
        ax_res.set_title(f"{t_name} Residuals vs Predicted Power", fontsize=12, fontweight='bold')
        ax_res.set_xlabel("Predicted Active Power (kW)", fontsize=10)
        ax_res.set_ylabel("Residual Error (kW)", fontsize=10)
        ax_res.grid(True, ls=':', alpha=0.6)

    plt.tight_layout()
    chart3 = os.path.join(VIS_DIR, '09_residual_analysis.png')
    fig.savefig(chart3, dpi=200)
    plt.close(fig)
    print(f"  Saved {chart3}")

    # 4. Feature Importance Top 12 Ranking
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for i, (t_name, data) in enumerate(preds_dict.items()):
        ax = axes[i]
        top_imp = data['power_model'].get_feature_importances().head(12)
        sns.barplot(data=top_imp, x='importance', y='feature', ax=ax, color=colors[t_name], orient='h')
        ax.set_title(f"{t_name} Top 12 Feature Importances (Power Model)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Gini Feature Importance", fontsize=10)
        ax.set_ylabel("Input Feature", fontsize=10)
        ax.grid(True, ls=':', alpha=0.6)

    plt.tight_layout()
    chart4 = os.path.join(VIS_DIR, '10_feature_importance.png')
    fig.savefig(chart4, dpi=200)
    plt.close(fig)
    print(f"  Saved {chart4}")

    # 5. Festival vs Normal Day Performance Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    comparison_data = []
    for t_name, data in preds_dict.items():
        y_true = data['y_power_true'] / 1000.0
        y_pred = data['y_power_pred'] / 1000.0
        is_fest = data['df']['is_festival'] == 1

        norm_mae = np.mean(np.abs(y_true[~is_fest] - y_pred[~is_fest]))
        fest_mae = np.mean(np.abs(y_true[is_fest] - y_pred[is_fest]))

        comparison_data.append({'Transformer': t_name, 'Regime': 'Normal Days', 'MAE (kW)': norm_mae})
        comparison_data.append({'Transformer': t_name, 'Regime': 'Festival Days', 'MAE (kW)': fest_mae})

    comp_df = pd.DataFrame(comparison_data)
    sns.barplot(data=comp_df, x='Transformer', y='MAE (kW)', hue='Regime', ax=ax, palette=['#64748b', '#f59e0b'])
    ax.set_title("Forecast Accuracy Comparison: Normal Days vs Festival Days (2017)", fontsize=13, fontweight='bold', pad=10)
    ax.set_ylabel("Mean Absolute Error (kW)", fontsize=11)
    ax.legend(title='Operational Regime', frameon=True)
    ax.grid(True, ls=':', alpha=0.6)

    plt.tight_layout()
    chart5 = os.path.join(VIS_DIR, '11_festival_prediction_performance.png')
    fig.savefig(chart5, dpi=200)
    plt.close(fig)
    print(f"  Saved {chart5}")
    print("\nAll evaluation charts generated successfully!")

if __name__ == '__main__':
    evaluate_all()

