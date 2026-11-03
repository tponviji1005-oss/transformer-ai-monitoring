# Milestone 2: Final Data Preprocessing & Boundary Validation Report
**Project:** AI-Based Transformer Load, Overload and Efficiency Prediction System  
**Report File:** `reports/milestone2_final_validation.md`  
**Raw Source File:** `data/raw/transformer_hourly_festival_with_temperature.csv` (Untouched, Pristine)  
**Processed Datasets Directory:** `data/processed/`  
**Date:** 2026-11-02  
**Status:** All 10 Milestone 2 Requirements Verified & Approved  

---

## 1. Executive Summary

Following independent audit approval, the recommended one-row boundary correction was applied to `src/preprocessing/preprocessor.py`. The training datasets (`t1_train.csv`, `t2_train.csv`, `t3_train.csv`) were regenerated, cleanly eliminating the boundary row at `2016-12-31 23:00:00 IST` whose next-hour target crossed into `2017-01-01 00:00:00 IST`. 

All test datasets (`t1_test.csv`, `t2_test.csv`, `t3_test.csv`) remain completely untouched. All validation scripts passed with zero warnings or errors.

---

## 2. One-Row Boundary Correction & Final Row Counts

### 2.1 The Code Modification in `src/preprocessing/preprocessor.py`
```python
# Before: Allowed feature timestamp 2016-12-31 23:00 whose target was 2017-01-01 00:00:
train_mask = valid_df['datetime'] < self.train_end_date

# After: Enforces that target timestamp (datetime + 1 hour) also stays strictly within 2016:
train_mask = (valid_df['datetime'] + pd.Timedelta(hours=1)) < self.train_end_date
test_mask = valid_df['datetime'] >= self.train_end_date
```

### 2.2 Row Count Reconciliation (Before vs. After Correction)

| Dataset | Raw Hours | Pre-Correction Rows | Post-Correction Rows | Net Change | Usable Date Range (Post-Correction) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`t1_train.csv`** | 27,168 | 14,284 | **14,283** | **-1 row** | `2013-12-06 10:00:00` to `2016-12-31 22:00:00` |
| **`t1_test.csv`**  | 8,760  | 8,341  | **8,341**  | **0 rows (Preserved)** | `2017-01-01 00:00:00` to `2017-12-31 16:00:00` |
| **`t2_train.csv`** | 27,168 | 24,239 | **24,238** | **-1 row** | `2013-12-06 10:00:00` to `2016-12-31 22:00:00` |
| **`t2_test.csv`**  | 8,760  | 8,117  | **8,117**  | **0 rows (Preserved)** | `2017-01-01 00:00:00` to `2017-12-31 16:00:00` |
| **`t3_train.csv`** | 27,168 | 24,391 | **24,390** | **-1 row** | `2013-12-06 10:00:00` to `2016-12-31 22:00:00` |
| **`t3_test.csv`**  | 8,760  | 8,382  | **8,382**  | **0 rows (Preserved)** | `2017-01-01 00:00:00` to `2017-12-31 16:00:00` |

---

## 3. Strict Chronological Separation Verification

Both feature timestamps and target timestamps are now 100% chronologically partitioned:

| Transformer Partition | Maximum Feature Timestamp | Maximum Target Timestamp ($t+1$) | Minimum Test Feature Timestamp | Minimum Test Target Timestamp ($t+1$) | Chronological Separation Verified? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Transformer T1** | `2016-12-31 22:00:00` | `2016-12-31 23:00:00` | `2017-01-01 00:00:00` | `2017-01-01 01:00:00` | **PASS (0 overlap)** |
| **Transformer T2** | `2016-12-31 22:00:00` | `2016-12-31 23:00:00` | `2017-01-01 00:00:00` | `2017-01-01 01:00:00` | **PASS (0 overlap)** |
| **Transformer T3** | `2016-12-31 22:00:00` | `2016-12-31 23:00:00` | `2017-01-01 00:00:00` | `2017-01-01 01:00:00` | **PASS (0 overlap)** |

$$\max(\text{Train Target Timestamp}) = \text{2016-12-31 23:00:00} < \min(\text{Test Feature Timestamp}) = \text{2017-01-01 00:00:00}$$

---

## 4. Verification of the Exact 39 Input Features

The feature list contains **exactly 39 unique features** (0 duplicates, 0 unexpected columns). Every column is documented below:

| # | Feature Name | Domain / Engineering Meaning | Temporal Horizon | Future Info? |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `power_W` | Active power measurement | Contemporaneous ($t$) | **No** |
| 2 | `oil_temp_C` | Top-oil temperature sensor reading | Contemporaneous ($t$) | **No** |
| 3 | `load_pct` | Current load percentage | Contemporaneous ($t$) | **No** |
| 4 | `ambient_temp_C` | Ambient weather temperature | Contemporaneous ($t$) | **No** |
| 5 | `hour` | Hour of the day (0–23) | Calendar ($t$) | **No** |
| 6 | `day_of_week` | Day of week (0=Mon, 6=Sun) | Calendar ($t$) | **No** |
| 7 | `month` | Month of year (1–12) | Calendar ($t$) | **No** |
| 8 | `is_weekend` | Weekend binary indicator | Calendar ($t$) | **No** |
| 9 | `hour_sin` | Cyclical sine encoding of hour | Calendar ($t$) | **No** |
| 10 | `hour_cos` | Cyclical cosine encoding of hour | Calendar ($t$) | **No** |
| 11 | `day_sin` | Cyclical sine encoding of day | Calendar ($t$) | **No** |
| 12 | `day_cos` | Cyclical cosine encoding of day | Calendar ($t$) | **No** |
| 13 | `month_sin` | Cyclical sine encoding of month | Calendar ($t$) | **No** |
| 14 | `month_cos` | Cyclical cosine encoding of month | Calendar ($t$) | **No** |
| 15 | `is_festival` | Festival holiday binary flag | Calendar ($t$) | **No** |
| 16 | `fest_christmas` | Christmas indicator | Calendar ($t$) | **No** |
| 17 | `fest_diwali` | Diwali indicator | Calendar ($t$) | **No** |
| 18 | `fest_dussehra` | Dussehra indicator | Calendar ($t$) | **No** |
| 19 | `fest_eid-ul-fitr` | Eid-ul-Fitr indicator | Calendar ($t$) | **No** |
| 20 | `fest_holi` | Holi indicator | Calendar ($t$) | **No** |
| 21 | `fest_new_year` | New Year indicator | Calendar ($t$) | **No** |
| 22 | `fest_pongal_sankranti` | Pongal/Sankranti indicator | Calendar ($t$) | **No** |
| 23 | `load_lag_1h` | Active power at $t-1$ | Past ($t-1$) | **No** |
| 24 | `load_lag_24h` | Active power at $t-24$ (diurnal cycle) | Past ($t-24$) | **No** |
| 25 | `load_lag_168h` | Active power at $t-168$ (weekly cycle) | Past ($t-168$) | **No** |
| 26 | `oil_temp_lag_1h` | Oil temperature at $t-1$ | Past ($t-1$) | **No** |
| 27 | `oil_temp_lag_24h` | Oil temperature at $t-24$ | Past ($t-24$) | **No** |
| 28 | `ambient_temp_lag_1h` | Ambient temperature at $t-1$ | Past ($t-1$) | **No** |
| 29 | `ambient_temp_lag_24h` | Ambient temperature at $t-24$ | Past ($t-24$) | **No** |
| 30 | `temp_gradient` | Thermal heat elevation $T_{oil}(t) - T_{amb}(t)$ | Contemporaneous ($t$) | **No** |
| 31 | `temp_gradient_lag_1h` | Thermal heat elevation at $t-1$ | Past ($t-1$) | **No** |
| 32 | `load_rolling_mean_24h` | Moving average power $\{t-24 \dots t-1\}$ | Past (via `shift(1)`) | **No** |
| 33 | `load_rolling_std_24h` | Moving std power $\{t-24 \dots t-1\}$ | Past (via `shift(1)`) | **No** |
| 34 | `load_rolling_min_24h` | Minimum power $\{t-24 \dots t-1\}$ | Past (via `shift(1)`) | **No** |
| 35 | `load_rolling_max_24h` | Maximum power $\{t-24 \dots t-1\}$ | Past (via `shift(1)`) | **No** |
| 36 | `load_rolling_mean_168h` | 7-day moving average $\{t-168 \dots t-1\}$ | Past (via `shift(1)`) | **No** |
| 37 | `load_rolling_std_168h` | 7-day moving std $\{t-168 \dots t-1\}$ | Past (via `shift(1)`) | **No** |
| 38 | `ambient_rolling_mean_24h` | Moving average ambient $\{t-24 \dots t-1\}$ | Past (via `shift(1)`) | **No** |
| 39 | `oil_temp_rolling_mean_24h` | Moving average oil temp $\{t-24 \dots t-1\}$ | Past (via `shift(1)`) | **No** |

### 4.1 Confirmation of Exclusions:
- **`uplift_pct`:** Confirmed **100% absent** from all feature lists and CSV files.
- **Invariant Signals:** `voltage_V`, `pf`, `frequency_Hz` confirmed removed.
- **Collinear Signals:** `watts_per_frequency`, `current_A`, `observed_W` confirmed removed.

---

## 5. Overload Reconciliation (Final)

| Transformer | Processed Train Overloads (`target_overload = 1`) | Processed Test Overloads (`target_overload = 1`) | Physical Note |
| :--- | :--- | :--- | :--- |
| **Transformer T1** | **0** | **122** | T1 never exceeded 81.0% load in raw 2013–2016 telemetry. Overloads occurred exclusively in 2017. |
| **Transformer T2** | **170** | **298** | 100% of 2013–2016 overloads preserved in training set. |
| **Transformer T3** | **0** | **0** | T3 operated conservatively under 78.70% load across all 4 years. |

---

## 6. Verification Scripts Test Results

All test suites executed against the newly generated datasets:

1. `scripts/audit_check.py`:
   - `[T1] Train target timestamp in 2017 count: 0`
   - `[T2] Train target timestamp in 2017 count: 0`
   - `[T3] Train target timestamp in 2017 count: 0`
   - **Result: PASS**

2. `scripts/validate_processed.py`:
   - `uplift_pct` strictly excluded: **PASS**
   - Invariant/redundant signals removed: **PASS**
   - 0 missing/null values across all columns: **PASS**
   - All 3 targets present: **PASS**
   - Train and Test strictly chronological: **PASS**
   - **Result: PASS**

3. `scripts/reconcile_overloads.py`:
   - Complete overload reconciliation verified: **PASS**

---

## 7. Sign-Off & ML Readiness

Milestone 2 is formally complete, verified, and locked. The preprocessed datasets are ready for **Milestone 3: Baseline Random Forest Model Training**.

