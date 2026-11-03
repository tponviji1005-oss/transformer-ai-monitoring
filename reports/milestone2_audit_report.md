# Milestone 2 Independent Audit & Integrity Verification Report
**Project:** AI-Based Transformer Load, Overload and Efficiency Prediction System  
**Milestone:** Pre-Milestone 3 Audit (Milestone 2 Validation)  
**Report File:** `reports/milestone2_audit_report.md`  
**Raw Source:** `data/raw/transformer_hourly_festival_with_temperature.csv` (Untouched, Pristine)  
**Audit Date:** 2026-11-02  
**Status:** Audit Complete — 1 Boundary Leakage Identified & Documented with Exact Correction  

---

## 1. Executive Summary

This independent audit conducted a forensic verification of the Milestone 2 data preprocessing pipeline and processed datasets (`t1_train.csv`, `t1_test.csv`, `t2_train.csv`, `t2_test.csv`, `t3_train.csv`, `t3_test.csv`). 

### Core Audit Outcomes:
1. **Target Timestamp Boundary Finding:** In the current training datasets, exactly **one boundary row** (`2016-12-31 23:00:00 IST`) has a next-hour target timestamp of `2017-01-01 00:00:00 IST`, which crosses into the 2017 test period. A surgical correction rule (`feature_datetime <= '2016-12-31 22:00:00'`) is recommended.
2. **Sensor Outage Integrity Verified:** Lag and rolling features **never cross sensor outage blackouts**. Post-outage feature generation requires a full contiguous buffer (e.g. 168 hours = 7 continuous days) before records are admitted.
3. **T1 Zero Train Overload Mystery Resolved:** In the raw dataset, T1 **never experienced any overloads between 2013 and 2016** (peak load was 81.0%). All 134 historical overload events for T1 occurred exclusively in 2017.
4. **Complete Overload Reconciliation:** All historical overloads in T1 (134), T2 (513), and T3 (0) have been reconciled with 100% mathematical precision.
5. **Feature Set Validation:** All 39 input features represent strictly past or contemporaneous information. Zero future signal leakage exists.
6. **No Test Data Contamination:** No global scalers, global imputation, or target statistics used 2017 test data.

---

## 2. Audit Item 1 & 6: Chronological Boundary & Target Timestamp Audit

### 2.1 The Finding
In Milestone 2, the chronological partition was defined as:
- Train: `datetime < '2017-01-01 00:00:00'`
- Test: `datetime >= '2017-01-01 00:00:00'`

While the **feature timestamp** for the last training record is `2016-12-31 23:00:00 IST` (inside 2016), its **target value** (`target_next_hour_power_W`) represents the measurement at $t+1$:
$$\text{Target Timestamp} = \text{2016-12-31 23:00:00} + 1\text{ hour} = \mathbf{2017-01-01\ 00:00:00\ IST}$$

Because `2017-01-01 00:00:00 IST` is the very first hour of the test set:
- In `t1_train.csv`: Row 14,283 uses `target_next_hour_power_W` = 36,070.28 W (the exact power of test row 0).
- In `t2_train.csv`: Row 24,238 uses `target_next_hour_power_W` = 200,298.18 W (the exact power of test row 0).
- In `t3_train.csv`: Row 24,390 uses `target_next_hour_power_W` = 98,102.36 W (the exact power of test row 0).

### 2.2 Recommended Correction
To strictly guarantee that **both feature timestamps and target timestamps** belong exclusively to the 2013–2016 training epoch:
$$\text{Train Boundary Condition:} \quad \text{feature\_datetime} \le \text{'2016-12-31 22:00:00 IST'} \quad (\text{so target} \le \text{'2016-12-31 23:00:00 IST'})$$

This drops exactly **1 boundary row** per transformer:
- T1 Train: 14,284 $\rightarrow$ 14,283 rows
- T2 Train: 24,239 $\rightarrow$ 24,238 rows
- T3 Train: 24,391 $\rightarrow$ 24,390 rows

---

## 3. Audit Item 2: Outage Gap Continuity & Lag Feature Verification

### 3.1 Mechanism of Outage Protection
The preprocessing pipeline maps the raw time series to a strict 1-hour regular grid. Any sensor outage or unrecorded period is populated as `NaN`.

When calculating lag and rolling windows:
- $L_1(t) = P(t-1)$
- $L_{24}(t) = P(t-24)$
- $L_{168}(t) = P(t-168)$

If timestamp $t - k$ falls inside a sensor blackout, $L_k(t)$ is `NaN`. Any record with a `NaN` feature is discarded.

### 3.2 Empirical Proof on T1's 15.5-Month Blackout
- **Outage Span:** `2015-04-21 14:00:00` to `2016-08-05 10:00:00` (11,325 hours).
- **Post-Outage Sensor Resumption:** `2016-08-05 10:00:00`.
- **First Admitted Row in `t1_train.csv`:** `2016-08-12 11:00:00`.
- **Gap Between Outage End and First Admitted Row:** Exactly **169 hours** ($\approx 7\text{ days and } 1\text{ hour}$).

**Conclusion:** Zero lag or rolling features bridge across outage gaps. A full 168 hours of continuous operational telemetry is enforced before any record is deemed usable.

---

## 4. Audit Item 3 & 4: Overload Reconciliation & Investigation of T1

### 4.1 Investigation: Why T1 Has Zero Training Overload Cases
During Milestone 1 EDA, 134 overload hours (>100% load) were identified for T1. However, the Milestone 2 processed training set showed **0 overload events**.

**The Cause:**
Forensic inspection of the raw dataset partitioned by date revealed:
- **T1 Load Percentage in 2013–2016:**
  - Minimum: 0.00%
  - Mean: 13.98%
  - **Maximum: 81.00%**
  - Count $> 100\%$: **0 rows**
- **T1 Load Percentage in 2017:**
  - Minimum: 0.00%
  - Mean: 27.02%
  - **Maximum: 241.20%**
  - Count $> 100\%$: **134 rows**

**Finding:** The zero overload count in `t1_train` is **not a data preprocessing error or filtering bug**. It is an intrinsic ground-truth characteristic of the raw telemetry: T1 operated within 81.0% of its rated capacity throughout 2013, 2014, 2015, and 2016. Severe overload events on T1 occurred exclusively in 2017.

### 4.2 Comprehensive Overload Reconciliation Matrix

| Subsystem | Metric | Raw Dataset (2013–2016) | Raw Dataset (2017) | Raw Total | Processed Train (`load_pct > 100`) | Processed Test (`load_pct > 100`) | Processed Test Target (`target_overload = 1`) | Unprocessed / Dropped Overload Hours | Reason for Dropped Overloads |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Transformer T1** | Overload Count | 0 | 134 | 134 | **0** | **121** | **122** | 13 | 13 hours on `2017-07-15 00:00` to `12:00` were dropped because their 168h lag originated in the raw sensor outage of `2017-07-08 00:00` to `12:00`. |
| **Transformer T2** | Overload Count | 170 | 343 | 513 | **170 (100%)** | **296** | **298** | 47 | 47 hours in July/Aug/Oct 2017 dropped because their 168h lag landed in localized sensor outages. 100% of 2013–2016 overloads were retained. |
| **Transformer T3** | Overload Count | 0 | 0 | 0 | **0** | **0** | **0** | 0 | T3 never exceeded 78.70% load across entire 4-year timeline. |

---

## 5. Audit Item 5: Feature Validity & Future Information Check

All 39 features present in the processed datasets were audited:

| Category | Feature Count | Features Included | Audit Verification |
| :--- | :--- | :--- | :--- |
| **Calendar / Temporal** | 4 | `hour`, `day_of_week`, `month`, `is_weekend` | Purely astronomical/calendar properties known at hour $t$. |
| **Cyclical Encodings** | 6 | `hour_sin`, `hour_cos`, `day_sin`, `day_cos`, `month_sin`, `month_cos` | Deterministic mathematical functions of hour, day, and month. |
| **Festival Indicators** | 8 | `is_festival`, `fest_christmas`, `fest_diwali`, `fest_dussehra`, `fest_eid-ul-fitr`, `fest_holi`, `fest_new_year`, `fest_pongal_sankranti` | Calendar holiday dates known in advance. **`uplift_pct` strictly excluded.** |
| **Current Telemetry** | 4 | `power_W`, `oil_temp_C`, `load_pct`, `ambient_temp_C` | Instantaneous readings recorded at hour $t$. |
| **Physical Gradient** | 2 | `temp_gradient`, `temp_gradient_lag_1h` | $T_{oil}(t) - T_{amb}(t)$ contemporaneous physical heat rise. |
| **Autoregressive Lags** | 7 | `load_lag_1h`, `load_lag_24h`, `load_lag_168h`, `oil_temp_lag_1h`, `oil_temp_lag_24h`, `ambient_temp_lag_1h`, `ambient_temp_lag_24h` | Measurements strictly at $t-1$, $t-24$, and $t-168$. |
| **Rolling Windows** | 8 | `load_rolling_mean_24h`, `load_rolling_std_24h`, `load_rolling_min_24h`, `load_rolling_max_24h`, `load_rolling_mean_168h`, `load_rolling_std_168h`, `ambient_rolling_mean_24h`, `oil_temp_rolling_mean_24h` | Computed using `.shift(1)` to cover past hours $\{t-1, \dots, t-W\}$. Observation at time $t$ is strictly excluded. |

**Audit Confirmation:** 100% of the 39 features are valid, physically grounded, and contain **zero future information**.

---

## 6. Audit Item 7: Test Data Contamination in Preprocessing

An audit was conducted on how preprocessing calculations were executed:
1. **Normalization / Scaling:** No `StandardScaler` or `MinMaxScaler` was applied. Features are preserved in raw physical units (Watts, °C, %, dimensionless cycles).
2. **Imputation:** No global or group mean imputation was performed. Rows with missing data were dropped.
3. **Encoding:** One-hot festival indicators were generated using a fixed domain list of the 7 Indian festivals present in the project scope.
4. **Rolling Windows:** Computed strictly online using backward sliding windows.

**Audit Confirmation:** **Zero test data statistics contaminated the training features.**

---

## 7. Applied Corrections & Verification Status

In accordance with user approval, the recommended boundary adjustment was applied to `src/preprocessing/preprocessor.py`:

```python
# Ensure target timestamp (datetime + 1 hour) also remains strictly inside 2016:
train_mask = (valid_df['datetime'] + pd.Timedelta(hours=1)) < self.train_end_date
test_mask = valid_df['datetime'] >= self.train_end_date
```

### Verification Results:
- **Boundary row `2016-12-31 23:00:00` removed** from all three training sets.
- **Final Training Row Counts:**
  - T1 Train: **14,283 rows** (Feature max: `2016-12-31 22:00:00`, Target max: `2016-12-31 23:00:00`)
  - T2 Train: **24,238 rows** (Feature max: `2016-12-31 22:00:00`, Target max: `2016-12-31 23:00:00`)
  - T3 Train: **24,390 rows** (Feature max: `2016-12-31 22:00:00`, Target max: `2016-12-31 23:00:00`)
- **Final Testing Row Counts (Unchanged):**
  - T1 Test: **8,341 rows** (Feature min: `2017-01-01 00:00:00`, Target min: `2017-01-01 01:00:00`)
  - T2 Test: **8,117 rows** (Feature min: `2017-01-01 00:00:00`, Target min: `2017-01-01 01:00:00`)
  - T3 Test: **8,382 rows** (Feature min: `2017-01-01 00:00:00`, Target min: `2017-01-01 01:00:00`)
- **Strict Chronological Separation:**
  $$\max(\text{Train Target Timestamp}) = \text{2016-12-31 23:00:00} < \min(\text{Test Feature Timestamp}) = \text{2017-01-01 00:00:00}$$
  Zero target overlap exists between training and testing data.

---

*Milestone 2 independent audit and boundary correction verified and closed.*

