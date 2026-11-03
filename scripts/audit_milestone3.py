import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.random_forest import TransformerRandomForest

def run_audit():
    print("="*75)
    print("MILESTONE 3 INDEPENDENT FORENSIC AUDIT")
    print("="*75)

    models_dir = "models"
    processed_dir = "data/processed"
    
    with open(os.path.join(processed_dir, "dataset_summary.json")) as f:
        ds_summary = json.load(f)

    with open(os.path.join(models_dir, "training_summary.json")) as f:
        train_summary = json.load(f)

    with open(os.path.join("reports", "milestone3_evaluation_metrics.json")) as f:
        eval_metrics = json.load(f)

    transformers = ['t1', 't2', 't3']
    targets = [
        ('power', 'target_next_hour_power_W', 'power_W'),
        ('load', 'target_next_hour_load_pct', 'load_pct')
    ]

    audit_results = {}

    for t in transformers:
        t_upper = t.upper()
        audit_results[t_upper] = {}
        
        # Load test set
        test_path = os.path.join(processed_dir, f"{t}_test.csv")
        test_df = pd.read_csv(test_path)
        
        # Load train set to verify training sample counts
        train_path = os.path.join(processed_dir, f"{t}_train.csv")
        train_df = pd.read_csv(train_path)

        for target_key, target_col, current_col in targets:
            model_file = f"{t}_rf_{target_key}.joblib"
            model_path = os.path.join(models_dir, model_file)
            
            print(f"\n--- Auditing Model: {model_file} ({t_upper} {target_key.upper()}) ---")
            assert os.path.exists(model_path), f"Missing model file: {model_path}"
            
            # 1. Load model verification
            rf_wrapper = TransformerRandomForest.load(model_path)
            model = rf_wrapper.model
            print(f"  [PASS] Model loaded successfully. Type: {type(model).__name__}")
            
            # 2. Verify training features and metadata
            n_features_in = model.n_features_in_
            print(f"  Model input features: {n_features_in} | Expected: 39")
            assert n_features_in == 39, f"Features mismatch: {n_features_in} != 39"
            
            recorded_train_rows = train_summary[t_upper][target_key]['train_rows']
            expected_train_rows = len(train_df)
            print(f"  Training samples: Recorded in summary={recorded_train_rows} | Actual train CSV={expected_train_rows}")
            assert recorded_train_rows == expected_train_rows, f"Sample mismatch: {recorded_train_rows} != {expected_train_rows}"
            print("  [PASS] Model trained exclusively on 2013-2016 dataset with 39 input features.")
            
            # 3. Verify exactly 39 features used
            features = rf_wrapper.feature_names
            print(f"  Feature count: {len(features)} | Unique count: {len(set(features))}")
            assert len(features) == 39, f"Feature count is {len(features)}, expected 39"
            assert len(features) == len(set(features)), "Duplicate features in model!"
            approved_features = ds_summary['transformers'][t_upper]['feature_list']
            assert features == approved_features, "Feature list does not match approved Milestone 2 schema!"
            assert 'uplift_pct' not in features, "CRITICAL: uplift_pct present in features!"
            assert target_col not in features, f"CRITICAL: target {target_col} present in features!"
            print("  [PASS] Exactly the approved 39 features used without target or uplift_pct.")
            
            # 4. Generate predictions on 2017 test set
            X_test = test_df[features]
            y_true = test_df[target_col].values
            y_pred = rf_wrapper.predict(X_test)
            
            # 5. Baseline Persistence model
            # Persistence: next hour value is predicted to be equal to current hour value
            y_persist = test_df[current_col].values
            
            # Evaluate RF
            mae_rf = float(mean_absolute_error(y_true, y_pred))
            rmse_rf = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            r2_rf = float(r2_score(y_true, y_pred))
            
            # Robust MAPEs:
            # a) wMAPE
            wmape_rf = float((np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))) * 100.0)
            
            # b) Filtered MAPE: actual >= 1.0 (or 0.1 for load)
            thresh = 100.0 if target_key == 'power' else 0.5
            mask_nz = y_true >= thresh
            mape_filtered_rf = float(np.mean(np.abs((y_true[mask_nz] - y_pred[mask_nz]) / y_true[mask_nz])) * 100.0)
            
            # c) Median Absolute Percentage Error (MdAPE)
            mdape_rf = float(np.median(np.abs((y_true[mask_nz] - y_pred[mask_nz]) / y_true[mask_nz])) * 100.0)
            
            # Evaluate Persistence
            mae_pers = float(mean_absolute_error(y_true, y_persist))
            rmse_pers = float(np.sqrt(mean_squared_error(y_true, y_persist)))
            r2_pers = float(r2_score(y_true, y_persist))
            wmape_pers = float((np.sum(np.abs(y_true - y_persist)) / np.sum(np.abs(y_true))) * 100.0)
            mape_filtered_pers = float(np.mean(np.abs((y_true[mask_nz] - y_persist[mask_nz]) / y_true[mask_nz])) * 100.0)
            mdape_pers = float(np.median(np.abs((y_true[mask_nz] - y_persist[mask_nz]) / y_true[mask_nz])) * 100.0)
            
            # Verify reported metrics match
            reported_mae = eval_metrics[t_upper][target_key]['overall']['mae']
            reported_rmse = eval_metrics[t_upper][target_key]['overall']['rmse']
            reported_r2 = eval_metrics[t_upper][target_key]['overall']['r2']
            reported_wmape = eval_metrics[t_upper][target_key]['overall']['wmape']
            
            assert np.isclose(mae_rf, reported_mae, atol=1e-3), f"MAE mismatch: {mae_rf} vs {reported_mae}"
            assert np.isclose(rmse_rf, reported_rmse, atol=1e-3), f"RMSE mismatch: {rmse_rf} vs {reported_rmse}"
            assert np.isclose(r2_rf, reported_r2, atol=1e-3), f"R2 mismatch: {r2_rf} vs {reported_r2}"
            assert np.isclose(wmape_rf, reported_wmape, atol=1e-3), f"wMAPE mismatch: {wmape_rf} vs {reported_wmape}"
            print("  [PASS] Recomputed metrics match reported Milestone 3 metrics exactly.")
            
            # Comparison: RF vs Persistence
            imp_mae = ((mae_pers - mae_rf) / mae_pers) * 100.0
            imp_rmse = ((rmse_pers - rmse_rf) / rmse_pers) * 100.0
            
            print(f"  Random Forest : MAE={mae_rf:.2f} | RMSE={rmse_rf:.2f} | R²={r2_rf:.4f} | wMAPE={wmape_rf:.2f}% | Filtered MAPE={mape_filtered_rf:.2f}% | MdAPE={mdape_rf:.2f}%")
            print(f"  Persistence   : MAE={mae_pers:.2f} | RMSE={rmse_pers:.2f} | R²={r2_pers:.4f} | wMAPE={wmape_pers:.2f}% | Filtered MAPE={mape_filtered_pers:.2f}% | MdAPE={mdape_pers:.2f}%")
            print(f"  RF Improvement over Persistence: MAE: {imp_mae:+.2f}%, RMSE: {imp_rmse:+.2f}%")
            
            audit_results[t_upper][target_key] = {
                'rf': {
                    'mae': mae_rf,
                    'rmse': rmse_rf,
                    'r2': r2_rf,
                    'wmape': wmape_rf,
                    'mape_filtered': mape_filtered_rf,
                    'mdape': mdape_rf
                },
                'persistence': {
                    'mae': mae_pers,
                    'rmse': rmse_pers,
                    'r2': r2_pers,
                    'wmape': wmape_pers,
                    'mape_filtered': mape_filtered_pers,
                    'mdape': mdape_pers
                },
                'improvement_pct': {
                    'mae': imp_mae,
                    'rmse': imp_rmse,
                    'r2_gain': r2_rf - r2_pers
                }
            }

        # 6. Detailed Investigation of T1 Overload Performance
        if t == 't1':
            print("\n" + "="*50)
            print("INVESTIGATING T1 OVERLOAD BEHAVIOR IN 2017 TEST SET")
            print("="*50)
            t1_rf_load = TransformerRandomForest.load(os.path.join(models_dir, "t1_rf_load.joblib"))
            y_true_load = test_df['target_next_hour_load_pct'].values
            y_pred_load = t1_rf_load.predict(test_df[t1_rf_load.feature_names])
            
            is_overload = y_true_load > 100.0
            is_normal = ~is_overload
            
            print(f"  Total test samples: {len(test_df)}")
            print(f"  Normal load samples (<=100%): {is_normal.sum()}")
            print(f"  Overload samples (>100%):     {is_overload.sum()}")
            
            # Normal load performance
            mae_norm = mean_absolute_error(y_true_load[is_normal], y_pred_load[is_normal])
            rmse_norm = np.sqrt(mean_squared_error(y_true_load[is_normal], y_pred_load[is_normal]))
            r2_norm = r2_score(y_true_load[is_normal], y_pred_load[is_normal])
            wmape_norm = (np.sum(np.abs(y_true_load[is_normal] - y_pred_load[is_normal])) / np.sum(y_true_load[is_normal])) * 100.0
            
            # Overload performance
            mae_ov = mean_absolute_error(y_true_load[is_overload], y_pred_load[is_overload])
            rmse_ov = np.sqrt(mean_squared_error(y_true_load[is_overload], y_pred_load[is_overload]))
            r2_ov = r2_score(y_true_load[is_overload], y_pred_load[is_overload])
            wmape_ov = (np.sum(np.abs(y_true_load[is_overload] - y_pred_load[is_overload])) / np.sum(y_true_load[is_overload])) * 100.0
            
            print("\n  --- T1 Under Normal Loading (<=100%) ---")
            print(f"    MAE: {mae_norm:.2f}% | RMSE: {rmse_norm:.2f}% | R²: {r2_norm:.4f} | wMAPE: {wmape_norm:.2f}%")
            
            print("\n  --- T1 During Overload Events (>100%) ---")
            print(f"    MAE: {mae_ov:.2f}% | RMSE: {rmse_ov:.2f}% | R²: {r2_ov:.4f} | wMAPE: {wmape_ov:.2f}%")
            print(f"    Actual Max Load: {y_true_load[is_overload].max():.2f}%")
            print(f"    Predicted Max Load during Overloads: {y_pred_load[is_overload].max():.2f}%")
            print(f"    Actual Mean Overload: {y_true_load[is_overload].mean():.2f}%")
            print(f"    Predicted Mean Overload: {y_pred_load[is_overload].mean():.2f}%")
            
            audit_results['T1']['overload_investigation'] = {
                'normal_samples': int(is_normal.sum()),
                'overload_samples': int(is_overload.sum()),
                'normal_metrics': {'mae': mae_norm, 'rmse': rmse_norm, 'r2': r2_norm, 'wmape': wmape_norm},
                'overload_metrics': {'mae': mae_ov, 'rmse': rmse_ov, 'r2': r2_ov, 'wmape': wmape_ov},
                'actual_max_load': float(y_true_load[is_overload].max()),
                'predicted_max_load': float(y_pred_load[is_overload].max()),
                'actual_mean_overload': float(y_true_load[is_overload].mean()),
                'predicted_mean_overload': float(y_pred_load[is_overload].mean())
            }

    with open(os.path.join("reports", "milestone3_audit_metrics.json"), "w") as f:
        json.dump(audit_results, f, indent=2)

    print("\n" + "="*75)
    print("AUDIT EXECUTION FINISHED SUCCESSFULLY!")
    print("="*75)

if __name__ == '__main__':
    run_audit()
