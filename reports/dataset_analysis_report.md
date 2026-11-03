# Transformer Telemetry Dataset Analysis & Exploratory Findings
**Project:** AI-Based Transformer Load, Overload and Efficiency Prediction System  
**Milestone:** Milestone 1 – Dataset Analysis  
**Dataset File:** `data/raw/transformer_hourly_festival_with_temperature.csv`  
**Analysis Date:** 2026-11-02  
**Status:** Complete & Verified  

---

## 1. Dataset Overview

The historical dataset comprises recorded hourly telemetry from a distribution substation monitoring three electrical transformers over a continuous span of **4 years, 1 month, and 6 days**.

### 1.1 Structural Properties
- **Total Records (Rows):** 35,928
- **Total Features (Columns):** 32
- **Date Range:** `2013-11-26 00:00:00 IST` to `2017-12-31 23:00:00 IST`
- **Sampling Frequency:** Strict hourly frequency ($1\text{ hour} = 3,600\text{ seconds}$). Exactly 35,927 consecutive 1-hour step deltas.
- **Duplicate Records:** 0 full duplicate rows; 0 duplicate timestamp entries.
- **Substation Units Monitored:** 3 distribution transformers (designated as **T1**, **T2**, and **T3**).

### 1.2 Column Taxonomy & Data Types

| Group | Features | Data Type | Physical Dimension / Meaning |
| :--- | :--- | :--- | :--- |
| **Temporal Index** | `datetime_IST`, `timestamp` | `object` (string), `int64` | Timestamp in Indian Standard Time (IST) & Unix epoch |
| **Environmental** | `ambient_temperature_C` | `float64` | Ambient weather temperature in degrees Celsius (°C) |
| **Calendar / Festival**| `festival`, `uplift_pct` | `object` (string), `int64` | Indian festival classification and simulated uplift percentage |
| **Transformer T1** | `t1_observed_W`, `t1_power_W`, `t1_voltage_V`, `t1_current_A`, `t1_pf`, `t1_watts_per_frequency`, `t1_frequency_Hz`, `t1_load_pct`, `t1_oil_temp_C` | `float64` | T1 electrical, thermal, and load percentage measurements |
| **Transformer T2** | `t2_observed_W`, `t2_power_W`, `t2_voltage_V`, `t2_current_A`, `t2_pf`, `t2_watts_per_frequency`, `t2_frequency_Hz`, `t2_load_pct`, `t2_oil_temp_C` | `float64` | T2 electrical, thermal, and load percentage measurements |
| **Transformer T3** | `t3_observed_W`, `t3_power_W`, `t3_voltage_V`, `t3_current_A`, `t3_pf`, `t3_watts_per_frequency`, `t3_frequency_Hz`, `t3_load_pct`, `t3_oil_temp_C` | `float64` | T3 electrical, thermal, and load percentage measurements |

---

## 2. Transformer Subsystem Analysis

Each transformer exhibits distinctive electrical load profiles, capacity utilization, and thermal dynamics.

### 2.1 Transformer 1 (T1) Analysis
- **Available Parameters (9):** `observed_W`, `power_W`, `voltage_V`, `current_A`, `pf`, `watts_per_frequency`, `frequency_Hz`, `load_pct`, `oil_temp_C`.
- **Valid Recorded Hours:** 23,940 hours (66.63% of timeline; 11,988 missing hours).
- **Operating Load Statistics:**
  - Active Power ($P$): Mean = 53.31 kW, Median = 28.19 kW, Max = **683.74 kW**.
  - Current ($I$): Mean = 82.41 A, Median = 43.57 A, Max = **1,056.92 A**.
  - Load Percentage: Mean = 18.80%, Median = 9.90%, Max = **241.20%**.
  - Top-Oil Temperature: Mean = 42.13°C, Median = 41.00°C, Max = **167.30°C**.
- **Electrical Characterization:**
  - Implied Rated Active Capacity: $\approx 283.5\text{ kW}$ ($\approx 315\text{ kVA}$ at $0.90\text{ pf}$).
  - Overload Profile: Overloaded ($> 100\%$) for 134 hours; severely overloaded ($> 120\%$) for 70 hours.
  - Thermal Stress: Top-oil temperature peaked at 167.30°C during extreme load events, indicating critical thermal stress.

### 2.2 Transformer 2 (T2) Analysis
- **Available Parameters (9):** Identical parameter set to T1.
- **Valid Recorded Hours:** 34,022 hours (94.69% of timeline; 1,906 missing hours).
- **Operating Load Statistics:**
  - Active Power ($P$): Mean = 127.42 kW, Median = 95.66 kW, Max = **414.01 kW**.
  - Current ($I$): Mean = 196.96 A, Median = 147.87 A, Max = **639.97 A**.
  - Load Percentage: Mean = 44.94%, Median = 33.70%, Max = **146.00%**.
  - Top-Oil Temperature: Mean = 50.74°C, Median = 50.30°C, Max = **110.90°C**.
- **Electrical Characterization:**
  - Implied Rated Active Capacity: $\approx 283.5\text{ kW}$ ($\approx 315\text{ kVA}$ at $0.90\text{ pf}$).
  - Overload Profile: High base utilization; overloaded ($> 100\%$) for **513 hours** (1.43% of runtime); severely overloaded ($> 120\%$) for 101 hours.
  - Thermal Stress: Sustained high operating temperature ($50.74^\circ\text{C}$ mean), correlating with continuous commercial/urban demand.

### 2.3 Transformer 3 (T3) Analysis
- **Available Parameters (9):** Identical parameter set to T1 and T2.
- **Valid Recorded Hours:** 34,172 hours (95.11% of timeline; 1,756 missing hours).
- **Operating Load Statistics:**
  - Active Power ($P$): Mean = 73.92 kW, Median = 70.58 kW, Max = **177.03 kW**.
  - Current ($I$): Mean = 114.26 A, Median = 109.11 A, Max = **273.66 A**.
  - Load Percentage: Mean = 32.85%, Median = 31.40%, Max = **78.70%**.
  - Top-Oil Temperature: Mean = 44.99°C, Median = 46.70°C, Max = **67.90°C**.
- **Electrical Characterization:**
  - Implied Rated Active Capacity: $\approx 225.0\text{ kW}$ ($\approx 250\text{ kVA}$ at $0.90\text{ pf}$).
  - Overload Profile: **0 hours of overload**. Maximum load never exceeded 78.70%.
  - Thermal Stress: Mild, controlled thermal operating range (21.0°C to 67.9°C). Highly stable baseline feeder.

### 2.4 Side-by-Side Comparison Matrix

| Operational Metric | Transformer T1 | Transformer T2 | Transformer T3 | Unit |
| :--- | :--- | :--- | :--- | :--- |
| **Implied Transformer Rating** | ~315 kVA (283.5 kW) | ~315 kVA (283.5 kW) | ~250 kVA (225.0 kW) | kVA / kW |
| **Telemetry Availability** | 66.63% (23,940 hrs) | 94.69% (34,022 hrs) | 95.11% (34,172 hrs) | % (Hours) |
| **Mean Active Power** | 53.31 | 127.42 | 73.92 | kW |
| **Peak Active Power** | 683.74 | 414.01 | 177.03 | kW |
| **Mean Load Percentage** | 18.80% | 44.94% | 32.85% | % |
| **Peak Load Percentage** | **241.20%** | **146.00%** | **78.70%** | % |
| **Overload Count (>100%)** | 134 | 513 | 0 | Hours |
| **Severe Overload (>120%)** | 70 | 101 | 0 | Hours |
| **Mean Oil Temperature** | 42.13 | 50.74 | 44.99 | °C |
| **Max Oil Temperature** | 167.30 | 110.90 | 67.90 | °C |
| **Correlation with Ambient Temp** | 0.8728 | 0.7132 | 0.9482 | Pearson $r$ |

---

## 3. Festival Operational Dynamics

The dataset tags specific dates with major Indian festivals. In strict adherence to project scope rules:
- **`uplift_pct` was NOT used to infer or generate load metrics.**
- Historical comparisons are computed strictly from recorded telemetry (`power_W` conditioned on `festival`).

### 3.1 Festival Breakdown & Empirical Uplift
- **Normal Day Runtime:** 35,574 hours (99.01%)
- **Festival Runtime:** 354 hours (0.99%) across 7 festival categories:
  - Christmas (60 hours), New Year (54 hours), Pongal/Sankranti (48 hours), Holi (48 hours), Eid-ul-Fitr (48 hours), Dussehra (48 hours), Diwali (48 hours).

| Transformer | Normal Day Mean Power (kW) | Festival Day Mean Power (kW) | Empirical Load Uplift (%) | Observed Behavioral Pattern |
| :--- | :--- | :--- | :--- | :--- |
| **T1** | 53.40 kW | 45.35 kW | **-15.07%** | Institutional/commercial load reduction during holidays |
| **T2** | 126.89 kW | 180.02 kW | **+41.87%** | Substantial evening residential & illumination load surge |
| **T3** | 73.63 kW | 103.37 kW | **+40.40%** | Marked residential/public festive demand increase |

The empirical divergence between T1 (which decreases during festivals) and T2/T3 (which experience a >40% load surge) reflects distinct feeder types: T1 likely supplies educational/commercial facilities that close during holidays, whereas T2 and T3 feed residential and commercial consumer areas.

---

## 4. Visualizations Summary

All analysis charts were generated at high resolution (200 DPI) and are stored in the project workspace:

1. **Hourly Load Trends:** [`visualizations/01_hourly_load_trends.png`](file:///c:/Users/tponv/transformer-ai-monitoring/visualizations/01_hourly_load_trends.png)
   - Visualizes diurnal load profiles (00:00 to 23:00 IST), interquartile ranges (IQR 25th-75th percentile), and peak demand hours (18:00 - 22:00 IST).
2. **Daily and Monthly Trends:** [`visualizations/02_daily_monthly_load_trends.png`](file:///c:/Users/tponv/transformer-ai-monitoring/visualizations/02_daily_monthly_load_trends.png)
   - Illustrates multi-year seasonal progression and the 30-day moving average demand, clearly highlighting the mid-2015 to mid-2016 T1 outage.
3. **Festival vs Non-Festival Comparison:** [`visualizations/03_festival_vs_non_festival.png`](file:///c:/Users/tponv/transformer-ai-monitoring/visualizations/03_festival_vs_non_festival.png)
   - Direct hourly profile overlays contrasting festival days against normal operating baselines.
4. **Transformer-Wise Distribution:** [`visualizations/04_transformer_wise_comparison.png`](file:///c:/Users/tponv/transformer-ai-monitoring/visualizations/04_transformer_wise_comparison.png)
   - Boxplots of load percentage and oil temperature against standard IEEE limits (100% capacity, 120% overload, 90°C oil warning).
5. **Missing Values Timeline:** [`visualizations/05_missing_values_timeline.png`](file:///c:/Users/tponv/transformer-ai-monitoring/visualizations/05_missing_values_timeline.png)
   - Temporal heatmap displaying daily missingness rates, illustrating the 15.5-month gap in T1 and synchronized T2/T3 substation outage.
6. **Correlation Matrices:** [`visualizations/06_correlation_heatmaps.png`](file:///c:/Users/tponv/transformer-ai-monitoring/visualizations/06_correlation_heatmaps.png)
   - Correlation heatmaps quantifying inter-sensor relationships between active power, current, oil temperature, and ambient temperature.

---

## 5. Machine Learning Readiness & Modeling Formulation

### 5.1 Target Variables for Next-Hour Prediction
For next-hour forecasting ($t+1$), the following target variables are well-defined:
1. **Load Demand Forecasting (Regression):**
   - Target: `power_W(t+1)` or `load_pct(t+1)`
   - Primary metric: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), $R^2$.
2. **Overload Classification:**
   - Target: Binary indicator $\mathbb{I}(\text{load\_pct}(t+1) > 100.0\%)$.
   - Primary metric: F1-Score, Precision-Recall AUC (critical due to class imbalance: 0.37% overload in T1, 1.43% in T2).

### 5.2 Recommended Feature Engineering Pipeline
- **Temporal & Cyclical Features:**
  - Hour of day ($\sin / \cos$ transformation: $\sin(2\pi h/24)$, $\cos(2\pi h/24)$).
  - Day of week ($\sin / \cos$ transformation: $\sin(2\pi d/7)$, $\cos(2\pi d/7)$).
  - Month of year ($\sin / \cos$ transformation: $\sin(2\pi m/12)$, $\cos(2\pi m/12)$).
  - Is weekend indicator (`is_weekend` $\in \{0, 1\}$).
- **Festival Indicator:**
  - One-hot or categorical encoding of `festival` (Normal day, Diwali, Pongal, Christmas, etc.) or binary `is_festival`.
  - **Explicit Exclusion:** `uplift_pct` will NEVER be used as an input feature.
- **Meteorological Predictor:**
  - `ambient_temperature_C` (and 1h, 2h ambient temperature changes).
- **Lagged Sensor Features (Autoregressive):**
  - Load lags: $\text{power}(t), \text{power}(t-1), \text{power}(t-2), \text{power}(t-23), \text{power}(t-24)$.
  - Oil temperature lags: $\text{oil\_temp}(t), \text{oil\_temp}(t-1)$.
  - Rolling statistics: 6-hour and 24-hour rolling mean, standard deviation, min, max.

### 5.3 Leakage Prevention Constraint
> [!IMPORTANT]
> Because T1 has a 11,325-hour gap and T2/T3 have a 1,054-hour gap, naive row-based lagging (`shift(1)`) will incorrectly pair measurements across months of missing time. Feature engineering must strictly enforce sequence continuity checks:
> $$\Delta t = t_i - t_{i-1} == 1\text{ hour}$$
> Any row where lag origin exceeds $\Delta t = 1\text{ hr}$ must have invalid lags marked as NaN and excluded from training chunks.

### 5.4 Dataset Limitations & Uncalculable Quantities

> [!CAUTION]
> **What CANNOT be calculated from the available CSV dataset:**
> 1. **Transformer Efficiency ($\eta$):**
>    Transformer efficiency is defined as $\eta = \frac{P_{out}}{P_{out} + P_{core} + \left(\frac{I}{I_{rated}}\right)^2 P_{cu}}$. The dataset provides active output load ($P_{out}$), but contains **no core/iron loss ($P_{core}$)** parameters and **no full-load copper loss ($P_{cu}$)** parameters. Efficiency cannot be calculated without assuming or obtaining manufacturer nameplate loss parameters.
> 2. **Voltage Sags / Voltage Swells:**
>    Voltage is recorded as a static constant $415.0\text{ V}$ ($\sigma = 0.00$). Dynamic voltage fluctuations cannot be modeled.
> 3. **Power Factor Variations & Reactive Power (VAR):**
>    Power factor is a constant $0.90$. Dynamic reactive power dynamics cannot be modeled.
> 4. **Winding Hotspot Temperature & Health Degradation:**
>    Only top-oil temperature is recorded. Internal winding hotspot temperature ($T_{hs}$) and paper insulation loss-of-life cannot be measured without thermal model constants from IEEE C57.91.

