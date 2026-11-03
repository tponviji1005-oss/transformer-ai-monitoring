# Data Preprocessing & Feature Engineering Report
**Project:** AI-Based Transformer Load, Overload and Efficiency Prediction System  
**Milestone:** Milestone 2 – Data Preprocessing and Feature Engineering  
**Raw Source File:** `data/raw/transformer_hourly_festival_with_temperature.csv` (Untouched, Pristine)  
**Processed Datasets Directory:** `data/processed/`  
**Date:** 2026-11-02  
**Status:** Completed, Validated & Verified  

---

## 1. Executive Summary

Milestone 2 establishes a leakage-free preprocessing pipeline and tabular feature store for **Transformer 1 (T1)**, **Transformer 2 (T2)**, and **Transformer 3 (T3)**. 

In strict adherence to the project scope and milestone guidelines:
1. The original raw CSV was never modified, altered, or overwritten.
2. `uplift_pct` was strictly excluded from all feature sets ($X$), ensuring the model learns festival demand dynamics directly from historical measurements and calendar features.
3. Invariant features (voltage, power factor, frequency) and redundant collinear signals (`watts_per_frequency`, `current_A`, `observed_W`) were systematically removed with documented engineering rationale.
4. Missing values and long sensor blackouts were **never artificially imputed or forward-filled**. Unusable rows were dropped to protect model integrity.
5. All temporal lag and rolling statistics were generated strictly from past observations ($t-1, \dots, t-k$), guaranteeing zero future data leakage.
6. Chronological splitting was executed at `2017-01-01 00:00:00 IST`:
   - **Training Set:** `2013-11-26 00:00:00` to `2016-12-31 23:00:00` (4 years)
   - **Testing Set:** `2017-01-01 00:00:00` to `2017-12-31 16:00:00` (1 year, unseen)

---

## 2. Feature Engineering Architecture

For each transformer, **39 input features** and **3 future target variables** were engineered.

### 2.1 Calendar & Temporal Cyclical Features (17 Features)
- **Base Calendar Units:** `hour` (0–23), `day_of_week` (0–6), `month` (1–12), `is_weekend` (1 if Saturday/Sunday, else 0).
- **Trigonometric Cyclical Encodings:**
  - $\text{hour\_sin} = \sin(2\pi \cdot \text{hour} / 24)$, $\text{hour\_cos} = \cos(2\pi \cdot \text{hour} / 24)$
  - $\text{day\_sin} = \sin(2\pi \cdot \text{day} / 7)$, $\text{day\_cos} = \cos(2\pi \cdot \text{day} / 7)$
  - $\text{month\_sin} = \sin(2\pi \cdot (\text{month}-1) / 12)$, $\text{month\_cos} = \cos(2\pi \cdot (\text{month}-1) / 12)$
- **Festival Indicators:**
  - Binary flag: `is_festival` (1 if festival != 'Normal day', else 0).
  - One-hot festival categorical channels: `fest_christmas`, `fest_diwali`, `fest_dussehra`, `fest_eid_ul_fitr`, `fest_holi`, `fest_new_year`, `fest_pongal_sankranti`.

### 2.2 Instantaneous Telemetry & Physical Gradients (4 Features)
- `power_W`: Active load power at current hour $t$.
- `oil_temp_C`: Top-oil temperature at current hour $t$.
- `ambient_temp_C`: Ambient weather temperature at current hour $t$.
- `temp_gradient`: Thermal elevation above ambient ($T_{oil}(t) - T_{amb}(t)$), a vital indicator of transformer copper/iron heating.

### 2.3 Autoregressive Lag Features (7 Features)
- **Active Load Lags:**
  - `load_lag_1h`: Load at $t-1$ (immediate previous hour).
  - `load_lag_24h`: Load at $t-24$ (same hour on previous day; diurnal periodicity).
  - `load_lag_168h`: Load at $t-168$ (same hour on previous week; weekly periodicity).
- **Thermal Lags:**
  - `oil_temp_lag_1h`: $T_{oil}(t-1)$
  - `oil_temp_lag_24h`: $T_{oil}(t-24)$
  - `ambient_temp_lag_1h`: $T_{amb}(t-1)$
  - `ambient_temp_lag_24h`: $T_{amb}(t-24)$
  - `temp_gradient_lag_1h`: Gradient at $t-1$.

### 2.4 Rolling Statistical Windows (7 Features)
All rolling windows were computed using `.shift(1)` to strictly exclude the current hour's observation:
- `load_rolling_mean_24h`: Moving average of past 24 hours.
- `load_rolling_std_24h`: Moving volatility/standard deviation of past 24 hours.
- `load_rolling_min_24h`: Past 24 hours minimum load.
- `load_rolling_max_24h`: Past 24 hours peak load.
- `load_rolling_mean_168h`: 7-day baseline moving average.
- `load_rolling_std_168h`: 7-day baseline moving volatility.
- `ambient_rolling_mean_24h`: 24-hour moving average ambient temperature.
- `oil_temp_rolling_mean_24h`: 24-hour moving average oil temperature.

### 2.5 Next-Hour Prediction Targets (3 Columns)
- **`target_next_hour_power_W`:** Next hour active power $P(t+1) = \text{shift}(-1)$. Primary regression target.
- **`target_next_hour_load_pct`:** Next hour load percentage $\text{load\_pct}(t+1) = \text{shift}(-1)$. Auxiliary regression target.
- **`target_next_hour_overload`:** Binary classification target $\mathbb{I}(\text{target\_next\_hour\_load\_pct} > 100.0\%)$.

---

## 3. Excluded & Dropped Features Rationale

| Excluded Signal | Engineering Rationale & Documented Reason |
| :--- | :--- |
| **`uplift_pct`** | **Strict user constraint:** A simulated percentage boost applied during synthetic data preparation. Using it would allow the model to cheat rather than learning actual festive load behavior from historical measurements and calendar features. |
| **`voltage_V`** | **Zero variance ($\sigma = 0.00$):** Invariant constant at $415.0\text{ V}$ across all 35,928 rows. Provides zero predictive signal. |
| **`pf`** | **Zero variance ($\sigma = 0.00$):** Invariant constant at $0.90$ across all rows. Provides zero predictive signal. |
| **`frequency_Hz`** | **Zero variance ($\sigma = 0.00$):** Invariant constant at $50.00\text{ Hz}$ across all rows. Provides zero predictive signal. |
| **`watts_per_frequency`** | **Deterministic collinearity:** Identical to active power divided by 50 ($\text{power\_W} / 50.0$). Retaining it introduces redundant collinearity. |
| **`current_A`** | **Deterministic collinearity:** In a constant 415V, 0.9 pf 3-phase system, current is perfectly proportional to power ($I = P / 646.9042$, $r = 1.0000$). |
| **`observed_W`** | **Redundancy & ambiguity:** Matches `power_W` on normal days and represents pre-uplift baseline during festivals. Excluded from feature inputs. |

---

## 4. Usable Records & Transformer Breakdown

All missing records and sensor blackouts were dropped without artificial interpolation. The chronological boundary condition strictly enforces that both feature timestamps and target timestamps land within the respective partitions ($\text{target\_dt} \le \text{'2016-12-31 23:00:00'}$ for training). The resulting clean, verified datasets are structured as follows:

| Transformer | Total Raw Hours | Dropped Outage Hours | Total Usable Hours | Train Rows (2013–2016) | Test Rows (2017) | Input Features | Overload Events (Train / Test) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **T1** | 35,928 | 13,304 (37.03%) | **22,624 (62.97%)** | **14,283** | **8,341** | 39 | 0 / 122 |
| **T2** | 35,928 | 3,573 (9.94%) | **32,355 (90.06%)** | **24,238** | **8,117** | 39 | 170 / 298 |
| **T3** | 35,928 | 3,156 (8.78%) | **32,772 (91.22%)** | **24,390** | **8,382** | 39 | 0 / 0 |

### 4.1 Chronological Date Boundaries
- **Transformer T1:**
  - Train: `2013-12-06 10:00:00` to `2016-12-31 22:00:00` (Target: `2013-12-06 11:00:00` to `2016-12-31 23:00:00`)
  - Test: `2017-01-01 00:00:00` to `2017-12-31 16:00:00` (Target: `2017-01-01 01:00:00` to `2017-12-31 17:00:00`)
- **Transformer T2:**
  - Train: `2013-12-06 10:00:00` to `2016-12-31 22:00:00` (Target: `2013-12-06 11:00:00` to `2016-12-31 23:00:00`)
  - Test: `2017-01-01 00:00:00` to `2017-12-31 16:00:00` (Target: `2017-01-01 01:00:00` to `2017-12-31 17:00:00`)
- **Transformer T3:**
  - Train: `2013-12-06 10:00:00` to `2016-12-31 22:00:00` (Target: `2013-12-06 11:00:00` to `2016-12-31 23:00:00`)
  - Test: `2017-01-01 00:00:00` to `2017-12-31 16:00:00` (Target: `2017-01-01 01:00:00` to `2017-12-31 17:00:00`)

---

## 5. Data Leakage Prevention Audits

1. **Strict Temporal & Target Ordering:**
   - $\max(\text{Train Feature Datetime}) = \text{2016-12-31 22:00:00} < \min(\text{Test Feature Datetime}) = \text{2017-01-01 00:00:00}$.
   - $\max(\text{Train Target Datetime}) = \text{2016-12-31 23:00:00} < \min(\text{Test Feature Datetime}) = \text{2017-01-01 00:00:00}$.
   - Zero future timestamp or target contamination exists in training partitions.
2. **Backward-Looking Windows:**
   - All rolling calculations use `.shift(1)` to ensure the rolling statistics at time $t$ only see past hours $\{t-1, t-2, \dots\}$. The current observation $P(t)$ is never included in the rolling window.
3. **Outage Continuity Protection:**
   - Lags across multi-month sensor outages evaluate to `NaN` and are discarded, preventing cross-blackout jumps.
4. **Target Separation:**
   - Targets ($t+1$) are strictly future and isolated from the feature matrix ($X$).

---

## 6. Generated Files & Storage Location

The following files are verified and saved in `data/processed/`:
- `data/processed/t1_train.csv` (14,283 rows, 45 columns)
- `data/processed/t1_test.csv` (8,341 rows, 45 columns)
- `data/processed/t2_train.csv` (24,238 rows, 45 columns)
- `data/processed/t2_test.csv` (8,117 rows, 45 columns)
- `data/processed/t3_train.csv` (24,390 rows, 45 columns)
- `data/processed/t3_test.csv` (8,382 rows, 45 columns)
- `data/processed/dataset_summary.json` (Structured metadata summary)

Reusable preprocessing codebase:
- `src/preprocessing/feature_engineering.py`
- `src/preprocessing/preprocessor.py`
- `src/preprocessing/__init__.py`
- `scripts/preprocess_data.py` (CLI runner)
- `scripts/validate_processed.py` (Integrity test suite)

