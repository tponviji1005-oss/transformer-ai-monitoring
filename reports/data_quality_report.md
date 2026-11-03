# Data Quality & Integrity Report
**Project:** AI-Based Transformer Load, Overload and Efficiency Prediction System  
**Milestone:** Milestone 1 – Dataset Analysis  
**Dataset File:** `data/raw/transformer_hourly_festival_with_temperature.csv`  
**Date of Analysis:** 2026-11-02  
**Status:** Complete  

---

## 1. Executive Summary

This data quality audit evaluates the historical transformer telemetry dataset spanning from **2013-11-26 00:00:00 IST** to **2017-12-31 23:00:00 IST** (35,928 total hourly records across 32 columns). While the dataset demonstrates strict timestamp continuity (exact 1-hour intervals with zero duplicated timestamps or missing time steps), significant telemetry outages, physical sensor invariant patterns, and trailing null records require specialized preprocessing before machine learning model development.

---

## 2. Completeness & Missing Value Audit

### 2.1 Column-Wise Missing Values

Across the 35,928 rows, 140,832 cells are null. Missing values are exclusively confined to transformer-specific electrical and thermal sensor columns. Ambient temperature and calendar metadata are 100% complete.

| Feature Name | Total Rows | Non-Null Count | Missing Count | Missing Percentage |
| :--- | :--- | :--- | :--- | :--- |
| `datetime_IST` | 35,928 | 35,928 | 0 | 0.00% |
| `timestamp` | 35,928 | 35,928 | 0 | 0.00% |
| `festival` | 35,928 | 35,928 | 0 | 0.00% |
| `uplift_pct` | 35,928 | 35,928 | 0 | 0.00% |
| `ambient_temperature_C` | 35,928 | 35,928 | 0 | 0.00% |
| **`t1_observed_W`** | 35,928 | 23,946 | 11,982 | **33.35%** |
| **`t1_power_W`** | 35,928 | 23,940 | 11,988 | **33.37%** |
| **`t1_voltage_V`** | 35,928 | 23,940 | 11,988 | **33.37%** |
| **`t1_current_A`** | 35,928 | 23,940 | 11,988 | **33.37%** |
| **`t1_pf`** | 35,928 | 23,940 | 11,988 | **33.37%** |
| **`t1_watts_per_frequency`** | 35,928 | 23,940 | 11,988 | **33.37%** |
| **`t1_frequency_Hz`** | 35,928 | 23,940 | 11,988 | **33.37%** |
| **`t1_load_pct`** | 35,928 | 23,940 | 11,988 | **33.37%** |
| **`t1_oil_temp_C`** | 35,928 | 23,940 | 11,988 | **33.37%** |
| **`t2_observed_W`** | 35,928 | 34,028 | 1,900 | **5.29%** |
| **`t2_power_W`** | 35,928 | 34,022 | 1,906 | **5.31%** |
| **`t2_voltage_V`** | 35,928 | 34,022 | 1,906 | **5.31%** |
| **`t2_current_A`** | 35,928 | 34,022 | 1,906 | **5.31%** |
| **`t2_pf`** | 35,928 | 34,022 | 1,906 | **5.31%** |
| **`t2_watts_per_frequency`** | 35,928 | 34,022 | 1,906 | **5.31%** |
| **`t2_frequency_Hz`** | 35,928 | 34,022 | 1,906 | **5.31%** |
| **`t2_load_pct`** | 35,928 | 34,022 | 1,906 | **5.31%** |
| **`t2_oil_temp_C`** | 35,928 | 34,022 | 1,906 | **5.31%** |
| **`t3_observed_W`** | 35,928 | 34,178 | 1,750 | **4.87%** |
| **`t3_power_W`** | 35,928 | 34,172 | 1,756 | **4.89%** |
| **`t3_voltage_V`** | 35,928 | 34,172 | 1,756 | **4.89%** |
| **`t3_current_A`** | 35,928 | 34,172 | 1,756 | **4.89%** |
| **`t3_pf`** | 35,928 | 34,172 | 1,756 | **4.89%** |
| **`t3_watts_per_frequency`** | 35,928 | 34,172 | 1,756 | **4.89%** |
| **`t3_frequency_Hz`** | 35,928 | 34,172 | 1,756 | **4.89%** |
| **`t3_load_pct`** | 35,928 | 34,172 | 1,756 | **4.89%** |
| **`t3_oil_temp_C`** | 35,928 | 34,172 | 1,756 | **4.89%** |

---

## 3. Sensor Outage Patterns & Gaps

### 3.1 Major Telemetry Outages Identified

1. **Transformer T1 Major Outage (11,325 continuous hours = ~472 days):**
   - **Start:** `2015-04-21 14:00:00`
   - **End:** `2016-08-05 10:00:00`
   - **Description:** T1 suffered a continuous 15.5-month monitoring blackout. During this extended period, both T2 and T3 continued to operate and record normal load.
   - **Impact on ML:** Standard time-series lagging (e.g. `df['t1_power_W'].shift(1)`) across the whole dataframe will create data leakage by bridging a 15-month gap between April 2015 and August 2016 as if it were a single hour. Continuous sequences must be partitioned by segment.

2. **Synchronized T2 & T3 Substation Outage (1,054 continuous hours = ~44 days):**
   - **Start:** `2015-09-23 14:00:00`
   - **End:** `2015-11-06 11:00:00`
   - **Description:** T2 and T3 simultaneous telemetry outage. Note that T1 was already offline during this entire period, resulting in a total substation data blackout during October 2015.

3. **Trailing 6-Hour Inconsistency (`2017-12-31 18:00` to `2017-12-31 23:00`):**
   - On the final 6 rows of the dataset (New Year's Eve), `t1_observed_W`, `t2_observed_W`, and `t3_observed_W` are populated, but `power_W`, `current_A`, `load_pct`, and `oil_temp_C` are completely `NaN`.
   - This indicates a script or logging cutoff during festival uplift post-processing in the raw data generation pipeline.

---

## 4. Signal Invariance & Multicollinearity Findings

### 4.1 Invariant (Zero-Variance) Features
The following telemetry channels are completely static constants across all 35,928 hours:
- `t1_voltage_V`, `t2_voltage_V`, `t3_voltage_V` $\equiv 415.0\text{ V}$ ($\sigma = 0.00$)
- `t1_pf`, `t2_pf`, `t3_pf` $\equiv 0.90$ ($\sigma = 0.00$)
- `t1_frequency_Hz`, `t2_frequency_Hz`, `t3_frequency_Hz` $\equiv 50.00\text{ Hz}$ ($\sigma = 0.00$)

**Recommendation:** These zero-variance features provide zero predictive signal for machine learning and must be excluded from feature matrices ($X$) to prevent rank deficiency and tree split dilution.

### 4.2 Exact Deterministic Multicollinearity
1. **Frequency Normalization:**
   $$\text{watts\_per\_frequency} = \frac{\text{power\_W}}{50.00}$$
   `watts_per_frequency` is an exact collinear duplicate of active power.
2. **Current to Power Proportionality:**
   Because 3-phase line voltage ($V = 415\text{ V}$) and power factor ($\cos\phi = 0.90$) are constants:
   $$P = \sqrt{3} \times V_L \times I_L \times \cos\phi = \sqrt{3} \times 415 \times 0.90 \times I = 646.9042 \times I$$
   $$I = \frac{P}{646.9042}$$
   Current ($I$) is perfectly collinear ($r = 1.0000$) with active power ($P$).
3. **Load Percentage:**
   $$\text{load\_pct} = \frac{\text{power\_W}}{\text{Rated\_Watts}} \times 100$$
   `load_pct` is simply active power normalized by rated capacity ($r = 1.0000$).

---

## 5. Physical Limits & Operational Anomalies

| Subsystem | Metric | Minimum | Maximum | Realistic Operational Bound | Anomaly Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ambient** | `ambient_temperature_C` | 3.20°C | 44.80°C | 0°C to 50°C | Within valid physical meteorological range. |
| **T1** | `t1_oil_temp_C` | 18.70°C | 167.30°C | 20°C to 110°C (IEEE C57.91) | **Severe thermal anomaly:** 167.3°C is dangerously above flashpoint/degradation thresholds. |
| **T1** | `t1_load_pct` | 0.00% | 241.20% | 0% to 150% (emergency) | **Extreme overload:** 241.2% indicates extreme peak surge. |
| **T2** | `t2_oil_temp_C` | 23.20°C | 110.90°C | 20°C to 110°C | High, but aligns with observed 146% overload. |
| **T2** | `t2_load_pct` | 0.00% | 146.00% | 0% to 150% | High load, consistent with commercial/industrial feeder. |
| **T3** | `t3_oil_temp_C` | 21.00°C | 67.90°C | 20°C to 100°C | Well within normal thermal envelope. |
| **T3** | `t3_load_pct` | 0.00% | 78.70% | 0% to 100% | Conservative, never exceeded rated limit. |

---

## 6. Preprocessing Recommendations for Milestone 2

1. **Retain Raw Dataset Unchanged:** Stored raw CSV remains pristine at `data/raw/transformer_hourly_festival_with_temperature.csv`.
2. **Exclude Invariant Columns:** Drop `voltage_V`, `pf`, `frequency_Hz`, and `watts_per_frequency` from ML feature inputs.
3. **Strict Policy on `uplift_pct`:** Preserve in historical files but completely isolate from ML feature sets.
4. **Segment-Aware Feature Engineering:** Generate rolling averages and lag features only within contiguous continuous blocks (or enforce timestamp delta checks $\Delta t = 1\text{ hr}$) to prevent leakage across the 15-month T1 outage.
5. **Separate Modeling per Transformer:** Because T1, T2, and T3 serve distinct load categories with differing capacities and missingness profiles, models should be trained per transformer (or conditioned on transformer ID with separate time-series splits).

