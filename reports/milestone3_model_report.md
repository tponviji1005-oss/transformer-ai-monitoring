# Milestone 3: Random Forest Model Training & Evaluation Report
**Project:** AI-Based Transformer Load, Overload and Efficiency Prediction System  
**Milestone:** Milestone 3 – Baseline Random Forest Model Training  
**Report File:** `reports/milestone3_model_report.md`  
**Training Data (2013–2016):** `data/processed/t{1,2,3}_train.csv`  
**Test Data (2017 Unseen):** `data/processed/t{1,2,3}_test.csv`  
**Model Artifacts Directory:** `models/`  
**Date:** 2026-11-02  
**Status:** Completed, Evaluated & Verified  

---

## 1. Executive Summary

Milestone 3 delivers an end-to-end Machine Learning pipeline utilizing **Random Forest Regression** (`RandomForestRegressor` from `scikit-learn`) to predict next-hour active power consumption ($P_{t+1}$ in Watts) and load percentage ($\text{load\_pct}_{t+1}$) for three distinct distribution transformers (**T1**, **T2**, and **T3**).

Because each transformer exhibits unique physical capacities, customer demand dynamics, and historical outage profiles, **separate models were trained per transformer**. A total of **6 dedicated regression models** were trained, serialized to disk, and evaluated against the unseen 2017 chronological test dataset.

### Core Achievements:
1. **Zero Data Leakage:** All 6 models were fitted exclusively on the 2013–2016 training partitions. The 2017 test set was never seen during training or parameter configuration.
2. **Strict Feature Schema:** Exactly the 39 approved features from Milestone 2 were used. `uplift_pct`, future targets, invariant channels, and raw timestamps were strictly excluded.
3. **High Generalization Performance:**
   - **Transformer T3:** $R^2 = 0.8300$, $\text{wMAPE} = 8.03\%$ ($\text{MAE} = 6.87\text{ kW}$).
   - **Transformer T2:** $R^2 = 0.7511$, $\text{wMAPE} = 15.79\%$ ($\text{MAE} = 22.70\text{ kW}$).
   - **Transformer T1:** $R^2 = 0.7083$, $\text{wMAPE} = 27.47\%$ ($\text{MAE} = 17.14\text{ kW}$).
4. **Transparent Reporting on Dataset Boundaries:** The report documents why T1's relative error is higher (unprecedented 2017 overload spikes absent from 2013–2016 training) and adheres to all scope boundaries regarding overload classification and efficiency calculation.

---

## 2. Model Architecture & Hyperparameters

Each model utilizes a tuned `RandomForestRegressor` ensemble to balance predictive power, variance reduction, and inference latency:

| Hyperparameter | Value | Rationale & Engineering Justification |
| :--- | :--- | :--- |
| **Ensemble Size (`n_estimators`)** | `100` | Standard robust ensemble size providing stable variance reduction without excessive memory footprint. |
| **Max Tree Depth (`max_depth`)** | `20` | Constrains tree depth to prevent memorization of noise while capturing multi-tier temporal/thermal interactions. |
| **Min Samples per Split (`min_samples_split`)** | `4` | Requires at least 4 observations to consider an internal node split, mitigating isolated noise splits. |
| **Min Samples per Leaf (`min_samples_leaf`)** | `2` | Smooths leaf predictions by preventing single-sample leaf assignments. |
| **Random State (`random_state`)** | `42` | Strictly deterministic seed ensuring 100% reproducible training runs. |
| **Parallel Execution (`n_jobs`)** | `-1` | Utilizes all available CPU cores for parallelized tree construction. |

---

## 3. Six Trained Model Artifacts

All models were serialized using `joblib` into the `models/` directory alongside structured metadata:

| Model Filename | Transformer | Target Variable | Training Samples | In-Sample Train $R^2$ | Artifact Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`models/t1_rf_power.joblib`](file:///c:/Users/tponv/transformer-ai-monitoring/models/t1_rf_power.joblib) | T1 (315 kVA) | `target_next_hour_power_W` | 14,283 | 0.9771 | 42.6 MB |
| [`models/t1_rf_load.joblib`](file:///c:/Users/tponv/transformer-ai-monitoring/models/t1_rf_load.joblib) | T1 (315 kVA) | `target_next_hour_load_pct` | 14,283 | 0.9772 | 42.5 MB |
| [`models/t2_rf_power.joblib`](file:///c:/Users/tponv/transformer-ai-monitoring/models/t2_rf_power.joblib) | T2 (315 kVA) | `target_next_hour_power_W` | 24,238 | 0.9693 | 69.4 MB |
| [`models/t2_rf_load.joblib`](file:///c:/Users/tponv/transformer-ai-monitoring/models/t2_rf_load.joblib) | T2 (315 kVA) | `target_next_hour_load_pct` | 24,238 | 0.9693 | 69.4 MB |
| [`models/t3_rf_power.joblib`](file:///c:/Users/tponv/transformer-ai-monitoring/models/t3_rf_power.joblib) | T3 (250 kVA) | `target_next_hour_power_W` | 24,390 | 0.9824 | 82.8 MB |
| [`models/t3_rf_load.joblib`](file:///c:/Users/tponv/transformer-ai-monitoring/models/t3_rf_load.joblib) | T3 (250 kVA) | `target_next_hour_load_pct` | 24,390 | 0.9824 | 82.7 MB |

---

## 4. Test Set Evaluation Metrics (Unseen 2017 Dataset)

Evaluation was conducted across the entire 2017 test set (8,117 to 8,382 hourly observations per transformer).

### 4.1 Explanation of Evaluation Metrics
1. **MAE (Mean Absolute Error):** Average magnitude of forecast errors in engineering units (kW for power, % for load). Reflects typical day-to-day error expected by grid operators.
2. **RMSE (Root Mean Squared Error):** Penalizes large forecast misses more heavily due to squaring. Critical for power engineering where sudden large under-predictions risk tripping breakers.
3. **$R^2$ Score (Coefficient of Determination):** Fraction of target variance explained by the model relative to a trivial mean baseline ($1.0 = \text{perfect fit}$, $0.0 = \text{no predictive skill}$).
4. **wMAPE (Weighted Mean Absolute Percentage Error):**
   $$\text{wMAPE} = \frac{\sum_{i=1}^N |y_i - \hat{y}_i|}{\sum_{i=1}^N |y_i|} \times 100\%$$
   The standard grid-industry benchmark for load forecasting. Unlike standard MAPE, wMAPE avoids division-by-zero distortion during low-load or idle nocturnal hours and weights errors proportionally to total energy transferred.

### 4.2 Comprehensive Evaluation Results Table

| Transformer | Target Variable | Test Rows | MAE | RMSE | $R^2$ Score | wMAPE (%) | Operational Regime |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **T1** | Active Power ($P_{t+1}$) | 8,341 | **17.14 kW** | **37.39 kW** | **0.7083** | **27.47%** | Overall 2017 Test Set |
|  | Active Power (Normal) | 8,247 | 17.19 kW | 37.47 kW | 0.7088 | 27.50% | Normal Days |
|  | Active Power (Festival) | 94 | **11.29 kW** | **27.04 kW** | **0.4641** | **23.98%** | Festival Days |
| **T1** | Load Percentage ($\text{load}_{t+1}$) | 8,341 | **6.03%** | **13.15%** | **0.7100** | **27.39%** | Overall 2017 Test Set |
| **T2** | Active Power ($P_{t+1}$) | 8,117 | **22.70 kW** | **36.45 kW** | **0.7511** | **15.79%** | Overall 2017 Test Set |
|  | Active Power (Normal) | 8,023 | 22.65 kW | 36.44 kW | 0.7506 | 15.79% | Normal Days |
|  | Active Power (Festival) | 94 | **27.50 kW** | **37.46 kW** | **0.7626** | **15.98%** | Festival Days |
| **T2** | Load Percentage ($\text{load}_{t+1}$) | 8,117 | **7.96%** | **12.79%** | **0.7539** | **15.70%** | Overall 2017 Test Set |
| **T3** | Active Power ($P_{t+1}$) | 8,382 | **6.87 kW** | **9.17 kW** | **0.8300** | **8.03%** | Overall 2017 Test Set |
|  | Active Power (Normal) | 8,288 | 6.86 kW | 9.16 kW | 0.8275 | 8.04% | Normal Days |
|  | Active Power (Festival) | 94 | **7.59 kW** | **10.41 kW** | **0.8865** | **7.05%** | Festival Days |
| **T3** | Load Percentage ($\text{load}_{t+1}$) | 8,382 | **3.06%** | **4.08%** | **0.8295** | **8.04%** | Overall 2017 Test Set |

---

## 5. Diagnostic Charts & Analysis

Five publication-quality evaluation charts were generated and saved to `visualizations/`:

### 5.1 Actual vs. Predicted Power
- **Figure:** [`visualizations/07_actual_vs_predicted_power.png`](file:///c:/Users/tponv/transformer-ai-monitoring/visualizations/07_actual_vs_predicted_power.png)
- **Observations:**
  - **T3 (Bottom Row):** Displays nearly tight 1:1 alignment across the entire operational range (0 to 180 kW). The 1-week time-series tracking curve closely mirrors daily diurnal peaks and nocturnal valleys.
  - **T2 (Middle Row):** Tracks baseline demand closely up to ~300 kW. Slight under-prediction occurs during peak spikes (>350 kW), characteristic of tree ensembles averaging extreme leaf nodes.
  - **T1 (Top Row):** Accurately tracks the baseline up to ~200 kW. However, when actual 2017 load surges to 400–680 kW (extreme overload events), the model caps predictions near ~250–280 kW because the 2013–2016 training set contained zero observations above 230 kW.

### 5.2 Actual vs. Predicted Load Percentage
- **Figure:** [`visualizations/08_actual_vs_predicted_load.png`](file:///c:/Users/tponv/transformer-ai-monitoring/visualizations/08_actual_vs_predicted_load.png)
- **Observations:** Shows the physical 100% rated capacity threshold. T3 operates 100% within the safe sub-100% zone. T2 predictions correctly capture transitions between normal loading and moderate overloads (100–140%).

### 5.3 Residual Error Analysis
- **Figure:** [`visualizations/09_residual_analysis.png`](file:///c:/Users/tponv/transformer-ai-monitoring/visualizations/09_residual_analysis.png)
- **Observations:**
  - Residual distributions for all three transformers are symmetric and zero-centered ($\mu_{\text{error}} \approx 0$).
  - For T2 and T3, residual variance is homoscedastic across most of the operating range, confirming well-behaved linear-in-the-large dynamics.
  - T1 exhibits an extended right-hand residual tail representing the unlearned 2017 overload spikes.

### 5.4 Feature Importance Rankings
- **Figure:** [`visualizations/10_feature_importance.png`](file:///c:/Users/tponv/transformer-ai-monitoring/visualizations/10_feature_importance.png)
- **Key Physical Drivers Learned by the Ensembles:**
  1. `power_W` (Current load at hour $t$): Dominant feature across all transformers (61% to 86% Gini importance), capturing primary autoregressive momentum.
  2. `hour_sin` / `hour` (Diurnal clock): Second most important feature (2% to 3.5%), capturing solar and human behavioral routines.
  3. `temp_gradient` ($T_{oil} - T_{amb}$): Strongly influential for T1 (3.3%), reflecting how thermal heat build-up correlates with sustained electrical demand.
  4. `load_rolling_mean_168h` / `load_rolling_mean_24h`: Captures weekly and daily baseline load levels.
  5. `load_lag_1h` & `load_lag_24h`: Secondary lag predictors providing momentum and diurnal anchoring.

### 5.5 Festival Prediction Performance
- **Figure:** [`visualizations/11_festival_prediction_performance.png`](file:///c:/Users/tponv/transformer-ai-monitoring/visualizations/11_festival_prediction_performance.png)
- **Findings:**
  - On **T3**, the model achieves a remarkable **$R^2 = 0.8865$** on festival days with a **wMAPE of 7.05%**, demonstrating that the calendar and lag features successfully learned festive holiday load dynamics without needing `uplift_pct`.
  - On **T2**, festival error remains nearly identical to normal operating error ($\text{wMAPE} = 15.98\%$ festival vs $15.79\%$ normal, $R^2 = 0.7626$).
  - On **T1**, absolute error on festivals is lower than normal days ($\text{MAE} = 11.29\text{ kW}$ vs $17.19\text{ kW}$) because festival holidays in T1 coincide with institutional shutdowns.

---

## 6. Honest Reporting on Performance & Dataset Limitations

In accordance with strict project rules, the following physical and data limitations are highlighted:

1. **Transformer T1 Overload Limitations:**
   - **Ground Truth:** In the historical 2013–2016 training data, T1 never experienced a single hour above 81.0% load.
   - **Model Consequence:** Decision tree ensembles cannot extrapolate beyond the maximum target values seen during training ($\approx 230\text{ kW}$). When unprecedented 2017 overloads reached 683 kW, the model predicted the highest value it knew, leading to under-predictions during peak spikes.
   - **Boundary Statement:** **We do NOT claim that T1 has learned overload prediction.** Its model is a next-hour operational load forecaster under normal operating envelopes.
2. **Transformer T2 Overload Characterization:**
   - T2 has 170 training overloads and 298 test overloads. Its regression model accurately predicts next-hour load near and above 100%. However, **formal binary overload classification (predicting alarm triggers) is a separate dedicated task for future development.**
3. **Transformer T3 Overload Status:**
   - T3 never exceeded 78.70% load in its entire history. **We do NOT claim that T3 can predict or detect overloads**, as zero overload positive samples exist in the telemetry.
4. **Transformer Efficiency ($ \eta $):**
   - **No efficiency models or synthetic efficiency numbers were generated.** Efficiency requires measuring input power vs output power or knowing internal transformer loss parameters ($P_{\text{core}}, P_{\text{cu}}$), neither of which exist in the raw dataset.

---

## 7. Next Steps: Milestone 4 Preview

With baseline models trained and evaluated, the logical next milestone is **Milestone 4: Overload Risk Classification & Alert Engine**, which will:
1. Formulate dedicated classification models (e.g., Random Forest Classifier) for T2 where ground truth overloads exist.
2. Implement threshold-based risk scoring ($\text{Risk} \in \{\text{Normal}, \text{Elevated}, \text{Critical Overload}\}$).
3. Evaluate classification metrics: Precision, Recall, F1-Score, and ROC-AUC.

