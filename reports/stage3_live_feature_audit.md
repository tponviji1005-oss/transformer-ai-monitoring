# Milestone 5 Stage 3: Live ML Feature Generation — Architecture & Feasibility Audit Report

**Document ID:** `AUDIT-M5-S3-001-REV2`  
**Revision:** 2.0 (Reconciled & Authoritative)  
**Date:** 2026-11-03  
**Status:** COMPLETED & RECONCILED — READY FOR IMPLEMENTATION REVIEW  
**Scope:** Stage 3 Technical Audit, Exact 39-Feature Reconciliation, Hardware Mapping, Missing Input Policy, and Implementation Readiness.

---

## 1. Executive Summary & Ownership Boundaries

### 1.1 Project Ownership & Roles
- **Original Project Concept, Planning & Physical Hardware:** Managed and owned by collaborator in Madurai, Tamil Nadu.
- **Software Architecture, Machine Learning, Backend, Dashboard & Deployment:** Developed and deployed remotely from Chennai.
- **Physical Hardware Deployed in Madurai:**
  - Microcontroller: **ESP32 DOIT DEVKIT V1 (38-pin)**
  - Power Meter: **Single-phase PZEM-004T V4.0** (UART2: RX GPIO16, TX GPIO17)
  - Temperature Sensor: **MAX6675 K-type thermocouple amplifier** (SPI: SCK GPIO18, CS GPIO5, SO GPIO19)
  - Sampling Frequency: Sensor readings every 2 seconds via UART/SPI to the local serial monitor (Wi-Fi/HTTP API integration scheduled in later stages).
  - Confirmed Hardware Signals: Voltage ($V$), Current ($A$), Active Power ($W$), Energy ($kWh$), Frequency ($Hz$), Power Factor ($\cos\phi$), and Transformer Oil Temperature ($^\circ\text{C}$).

### 1.2 Binding Architectural Decisions
In accordance with user decisions, the following strict constraints govern Stage 3:
1. **Zero Temperature Substitution:** Oil temperature must **never** be substituted for ambient temperature.
2. **No Weather API Substitution Yet:** Weather APIs will **not** be integrated as an ambient substitute at this stage. Ambient temperature remains an explicit, configurable field; if missing, inference must be marked strictly **UNAVAILABLE**.
3. **Mandatory Nameplate Configuration:** Transformer rating ($S_{\text{rated}}$ in kVA) and phase configuration remain **unconfigured** until the hardware owner in Madurai provides verified nameplate specifications. Load percentage will not be calculated without confirmed nameplate information.
4. **Zero Synthetic / Invented Specifications:** No hardware ratings, capacities, or sensor values may be assumed or fabricated.

---

## 2. Reconciliation of the 39-Feature Breakdown

### 2.1 Resolution of Category Discrepancy
An audit of previous project documentation identified a taxonomic categorization discrepancy between Milestone 2 and the initial Stage 3 draft:

- **Milestone 2 Approved Breakdown:**  
  $$\mathbf{6 + 10 + 8 + 7 + 8 = 39}$$
  - **6** Instantaneous Telemetry & Physical Gradients (`power_W`, `oil_temp_C`, `load_pct`, `ambient_temp_C`, `temp_gradient`, `temp_gradient_lag_1h`)
  - **10** Calendar & Cyclical Encodings (4 base units + 6 trigonometric encodings)
  - **8** Festival Indicators (1 binary flag + 7 one-hot dummies)
  - **7** Pure Autoregressive Lags (3 active load lags + 2 oil temp lags + 2 ambient temp lags)
  - **8** Rolling Statistical Windows (4 load 24h + 2 load 168h + 1 ambient 24h + 1 oil temp 24h)

- **Stage 3 Draft Presentation:**  
  $$\mathbf{4 + 10 + 8 + 9 + 8 = 39}$$
  - The draft isolated 4 raw telemetry channels and grouped the 2 thermal gradient features (`temp_gradient`, `temp_gradient_lag_1h`) under the lag category ($7 + 2 = 9$).

**Reconciliation Outcome:**  
The exact set of 39 features is **100% identical** across `src/preprocessing/feature_engineering.py`, `src/preprocessing/preprocessor.py`, `data/processed/*.csv`, and all eight trained `.joblib` model artifacts (`feature_names_in_`).  
The authoritative Milestone 2 breakdown **$6 + 10 + 8 + 7 + 8 = 39$** is formally adopted as the official classification standard.

```
+---------------------------------------------------------------------------------------------------+
|                               RECONCILED 39-FEATURE TAXONOMY                                      |
+--------------------+----------------------------+-----------------------+-------------------------+
| Category           | Feature Count              | Direct Hardware Match | Requires State/External |
+--------------------+----------------------------+-----------------------+-------------------------+
| Telemetry & Grad.  | 6 features                 | 2 / 6                 | 4 (Ambient, Load %, dT) |
| Calendar/Cyclical  | 10 features                | 0 (Time-derived)      | 10 (Timestamp in IST)   |
| Festival Encoding  | 8 features                 | 0 (Calendar Table)    | 8 (Holiday Calendar)    |
| Autoregressive Lag | 7 features                 | 0 (Historical State)  | 7 (1h, 24h, 168h Lags)  |
| Rolling Windows    | 8 features                 | 0 (Historical State)  | 8 (24h, 168h Windows)   |
+--------------------+----------------------------+-----------------------+-------------------------+
| Total              | 39 features                | 2 direct channels     | 37 state/derived/gaps   |
+--------------------+----------------------------+-----------------------+-------------------------+
```

---

## 3. Authoritative 39-Feature List in Exact Model Order

Every one of the eight trained models (`t1_rf_power`, `t1_rf_load`, `t2_rf_power`, `t2_rf_load`, `t3_rf_power`, `t3_rf_load`, `t2_overload_classifier`, `t1_anomaly_detector`) requires the features in the exact indexed order below:

| Index | Feature Name | Official Category | Mathematical Definition / Calculation | Live Data Source | Operational Status |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | `power_W` | Instantaneous Telemetry | Active real power at hour $t$ | `hourly_telemetry.power_W` | **Available** |
| **2** | `oil_temp_C` | Instantaneous Telemetry | Top oil temperature at hour $t$ | `hourly_telemetry.oil_temperature_C` | **Available** |
| **3** | `load_pct` | Instantaneous Telemetry | $P(t) / (S_{\text{rated}} \times \cos\phi) \times 100\%$ | Derived from $P(t)$ & Nameplate kVA | **BLOCKED** (Nameplate unverified) |
| **4** | `ambient_temp_C` | Instantaneous Telemetry | Outdoor ambient temperature at hour $t$ | `hourly_telemetry.ambient_temperature_C` | **BLOCKED** (No ambient sensor) |
| **5** | `hour` | Calendar / Temporal | $\text{hour} \in [0, 23]$ in IST | `timestamp` localized to `Asia/Kolkata` | **Available** |
| **6** | `day_of_week` | Calendar / Temporal | $\text{day} \in [0, 6]$ (Mon=0, Sun=6) in IST | `timestamp` localized to `Asia/Kolkata` | **Available** |
| **7** | `month` | Calendar / Temporal | $\text{month} \in [1, 12]$ in IST | `timestamp` localized to `Asia/Kolkata` | **Available** |
| **8** | `is_weekend` | Calendar / Temporal | $\mathbb{I}(\text{day\_of\_week} \in \{5, 6\})$ | Derived from `day_of_week` | **Available** |
| **9** | `hour_sin` | Cyclical Encoding | $\sin(2\pi \times \text{hour} / 24.0)$ | Deterministic trigonometric math | **Available** |
| **10** | `hour_cos` | Cyclical Encoding | $\cos(2\pi \times \text{hour} / 24.0)$ | Deterministic trigonometric math | **Available** |
| **11** | `day_sin` | Cyclical Encoding | $\sin(2\pi \times \text{day\_of\_week} / 7.0)$ | Deterministic trigonometric math | **Available** |
| **12** | `day_cos` | Cyclical Encoding | $\cos(2\pi \times \text{day\_of\_week} / 7.0)$ | Deterministic trigonometric math | **Available** |
| **13** | `month_sin` | Cyclical Encoding | $\sin(2\pi \times (\text{month} - 1) / 12.0)$ | Deterministic trigonometric math | **Available** |
| **14** | `month_cos` | Cyclical Encoding | $\cos(2\pi \times (\text{month} - 1) / 12.0)$ | Deterministic trigonometric math | **Available** |
| **15** | `is_festival` | Festival Indicator | $\mathbb{I}(\text{festival} \ne \text{'Normal day'})$ | Indian holiday calendar lookup | **Available** |
| **16** | `fest_christmas` | Festival Indicator | $\mathbb{I}(\text{festival} == \text{'Christmas'})$ | December 25 annual lookup | **Available** |
| **17** | `fest_diwali` | Festival Indicator | $\mathbb{I}(\text{festival} == \text{'Diwali'})$ | Hindu lunar calendar lookup | **Available** |
| **18** | `fest_dussehra` | Festival Indicator | $\mathbb{I}(\text{festival} == \text{'Dussehra'})$ | Hindu lunar calendar lookup | **Available** |
| **19** | `fest_eid-ul-fitr` | Festival Indicator | $\mathbb{I}(\text{festival} == \text{'Eid-ul-Fitr'})$ | Islamic lunar calendar lookup | **Available** |
| **20** | `fest_holi` | Festival Indicator | $\mathbb{I}(\text{festival} == \text{'Holi'})$ | Hindu lunar calendar lookup | **Available** |
| **21** | `fest_new_year` | Festival Indicator | $\mathbb{I}(\text{festival} == \text{'New Year'})$ | January 1 annual lookup | **Available** |
| **22** | `fest_pongal_sankranti` | Festival Indicator | $\mathbb{I}(\text{festival} == \text{'Pongal/Sankranti'})$ | January 14/15 annual lookup | **Available** |
| **23** | `load_lag_1h` | Autoregressive Lag | $P(t - 1\text{h})$ | `hourly_telemetry` at $t - 1\text{h}$ | **Available** ($\ge 1\text{h}$ history) |
| **24** | `load_lag_24h` | Autoregressive Lag | $P(t - 24\text{h})$ | `hourly_telemetry` at $t - 24\text{h}$ | **Available** ($\ge 24\text{h}$ history) |
| **25** | `load_lag_168h` | Autoregressive Lag | $P(t - 168\text{h})$ | `hourly_telemetry` at $t - 168\text{h}$ | **Available** ($\ge 168\text{h}$ history) |
| **26** | `oil_temp_lag_1h` | Autoregressive Lag | $T_{\text{oil}}(t - 1\text{h})$ | `hourly_telemetry` at $t - 1\text{h}$ | **Available** ($\ge 1\text{h}$ history) |
| **27** | `oil_temp_lag_24h` | Autoregressive Lag | $T_{\text{oil}}(t - 24\text{h})$ | `hourly_telemetry` at $t - 24\text{h}$ | **Available** ($\ge 24\text{h}$ history) |
| **28** | `ambient_temp_lag_1h` | Autoregressive Lag | $T_{\text{amb}}(t - 1\text{h})$ | `hourly_telemetry` at $t - 1\text{h}$ | **BLOCKED** (No ambient sensor) |
| **29** | `ambient_temp_lag_24h` | Autoregressive Lag | $T_{\text{amb}}(t - 24\text{h})$ | `hourly_telemetry` at $t - 24\text{h}$ | **BLOCKED** (No ambient sensor) |
| **30** | `temp_gradient` | Physical Gradient | $T_{\text{oil}}(t) - T_{\text{amb}}(t)$ | $T_{\text{oil}}(t)$ minus $T_{\text{amb}}(t)$ | **BLOCKED** (No ambient sensor) |
| **31** | `temp_gradient_lag_1h` | Physical Gradient | $T_{\text{oil}}(t-1\text{h}) - T_{\text{amb}}(t-1\text{h})$ | Gradient at $t - 1\text{h}$ | **BLOCKED** (No ambient sensor) |
| **32** | `load_rolling_mean_24h` | Rolling Window | Mean $P$ over past 24h ($\ge 18\text{h}$) | `hourly_telemetry` $[t-24\text{h}, t-1\text{h}]$ | **Available** ($\ge 18\text{h}$ history) |
| **33** | `load_rolling_std_24h` | Rolling Window | Std $P$ over past 24h ($\ge 18\text{h}$) | `hourly_telemetry` $[t-24\text{h}, t-1\text{h}]$ | **Available** ($\ge 18\text{h}$ history) |
| **34** | `load_rolling_min_24h` | Rolling Window | Min $P$ over past 24h ($\ge 18\text{h}$) | `hourly_telemetry` $[t-24\text{h}, t-1\text{h}]$ | **Available** ($\ge 18\text{h}$ history) |
| **35** | `load_rolling_max_24h` | Rolling Window | Max $P$ over past 24h ($\ge 18\text{h}$) | `hourly_telemetry` $[t-24\text{h}, t-1\text{h}]$ | **Available** ($\ge 18\text{h}$ history) |
| **36** | `load_rolling_mean_168h` | Rolling Window | Mean $P$ over past 168h ($\ge 84\text{h}$) | `hourly_telemetry` $[t-168\text{h}, t-1\text{h}]$ | **Available** ($\ge 84\text{h}$ history) |
| **37** | `load_rolling_std_168h` | Rolling Window | Std $P$ over past 168h ($\ge 84\text{h}$) | `hourly_telemetry` $[t-168\text{h}, t-1\text{h}]$ | **Available** ($\ge 84\text{h}$ history) |
| **38** | `ambient_rolling_mean_24h` | Rolling Window | Mean $T_{\text{amb}}$ past 24h ($\ge 18\text{h}$) | `hourly_telemetry` $[t-24\text{h}, t-1\text{h}]$ | **BLOCKED** (No ambient sensor) |
| **39** | `oil_temp_rolling_mean_24h` | Rolling Window | Mean $T_{\text{oil}}$ past 24h ($\ge 18\text{h}$) | `hourly_telemetry` $[t-24\text{h}, t-1\text{h}]$ | **Available** ($\ge 18\text{h}$ history) |

---

## 4. Deep-Dive: Missing Input Blockers & Governed Policies

### 4.1 Ambient Temperature Dependency (6 Features Blocked)
The following 6 features require ambient temperature:
1. `ambient_temp_C` (Feature #4)
2. `ambient_temp_lag_1h` (Feature #28)
3. `ambient_temp_lag_24h` (Feature #29)
4. `temp_gradient` (Feature #30)
5. `temp_gradient_lag_1h` (Feature #31)
6. `ambient_rolling_mean_24h` (Feature #38)

**Governing Policy & Impact:**
- **Zero Substitution:** Oil temperature is **never** substituted for ambient temperature. Under load, transformer oil runs $15^\circ\text{C}$ to $40^\circ\text{C}$ higher than ambient. Substituting oil temperature would force `temp_gradient` to zero, completely invalidating the thermal models.
- **No External API Substitution Yet:** As decided, third-party weather APIs will not be plugged in at this time.
- **Handling in Live Feature Generator:**
  - If `hourly_telemetry.ambient_temperature_C` is `NULL`, all 6 ambient features evaluate strictly to `None` (`NaN`).
  - Feature generator marks overall vector readiness as **`STATUS_BLOCKED_MISSING_AMBIENT`**.
  - Any downstream ML inference call must detect this status and cleanly return:
    `{"status": "unavailable", "reason": "Ambient temperature channel is missing. Physical ambient sensor or confirmed input required."}`.

---

### 4.2 Transformer Nameplate Rating & Load Percentage (1 Feature Blocked)
Feature #3 is `load_pct`:
$$\text{load\_pct} = \frac{P_{\text{active}}}{S_{\text{rated}} \times \cos\phi_{\text{rated}}} \times 100\%$$

**Model Weight & Importance:**
In the trained models, `load_pct` carries substantial predictive importance:
- **T1 Power Regressor:** Rank #2 feature (**19.74% importance**).
- **T2 Power Regressor:** Rank #2 feature (**5.09% importance**).
- **T3 Power Regressor:** Rank #4 feature (**0.90% importance**).

**Governing Policy & Impact:**
- The hardware measures electrical parameters ($V, I, P, Q, \text{pf}$), but cannot detect the physical rating of the transformer iron core and winding.
- **Strict Rule:** Transformer rating ($S_{\text{rated}}$ in kVA), phase configuration (1-phase vs. 3-phase), and rated power factor remain **mandatory and unconfigured** until confirmed by the hardware owner in Madurai.
- **Handling in Live Feature Generator:**
  - If a transformer's configuration is unverified, `load_pct` evaluates strictly to `None`.
  - Feature generator marks overall vector readiness as **`STATUS_BLOCKED_MISSING_NAMEPLATE`**.
  - Inference cannot run until the nameplate rating is formally entered into system settings.

---

## 5. Temporal Verification: Timezones, Calendar & Cold Start

### 5.1 Historical IST Timestamp Handling
- **Historical Ground Truth:** The raw dataset used `datetime_IST` (Indian Standard Time, UTC+05:30).
- **Live System Requirement:** SQLite and FastAPI record timestamps in UTC.
- **Verification:**
  To guarantee exact mathematical equivalence with historical training:
  1. The live feature generator must convert UTC timestamps to `Asia/Kolkata` using Python's standard `zoneinfo.ZoneInfo("Asia/Kolkata")`:
     ```python
     dt_ist = utc_timestamp.astimezone(ZoneInfo("Asia/Kolkata"))
     ```
  2. `hour`, `day_of_week`, `month`, and `is_weekend` are extracted from `dt_ist`.
  3. Cyclical sine and cosine encodings use identical mathematical denominators:
     $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24.0}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24.0}\right)$$
     $$\text{day\_sin} = \sin\left(\frac{2\pi \cdot \text{day}}{7.0}\right), \quad \text{day\_cos} = \cos\left(\frac{2\pi \cdot \text{day}}{7.0}\right)$$
     $$\text{month\_sin} = \sin\left(\frac{2\pi \cdot (\text{month} - 1)}{12.0}\right), \quad \text{month\_cos} = \cos\left(\frac{2\pi \cdot (\text{month} - 1)}{12.0}\right)$$
  4. Note that `month` uses `month - 1` so that January maps to angle 0, matching `src/preprocessing/feature_engineering.py`.

### 5.2 Festival Encoding
- **Historical Ground Truth:** The training pipeline used an explicit list of 7 festivals: `Christmas`, `Diwali`, `Dussehra`, `Eid-ul-Fitr`, `Holi`, `New Year`, `Pongal/Sankranti`.
- **Live Implementation:**
  A deterministic date-lookup table maps Indian holiday calendar dates (in `Asia/Kolkata`) to festival names.
  - Normal day: `is_festival = 0`, all 7 `fest_*` features = `0`.
  - Festival day: `is_festival = 1`, matching `fest_<name> = 1`, other 6 `fest_*` features = `0`.
  - Exactly 8 binary indicators are populated, matching model order.

### 5.3 Verification of the 168-Hour History Requirement
- **Lag Dependency:** Feature #25 (`load_lag_168h`) is defined as active power at $t - 168\text{ hours}$ (exactly 7 days ago).
- **Rolling Dependencies:** Features #36 & #37 (`load_rolling_mean_168h`, `load_rolling_std_168h`) cover the window $[t - 168\text{h}, t - 1\text{h}]$ with $\text{min\_periods} = 84$ (requiring at least 84 valid hours).
- **Operational Proof:**
  - Even though the rolling mean can mathematically compute with 84 valid hours, `load_lag_168h` strictly requires the observation at $t - 168\text{h}$.
  - Therefore, full 39-feature generation is impossible during the first 168 hours (7 days) of a newly commissioned transformer.
  - The live feature generator must enforce a **Cold-Start Lifecycle**:
    - Hours $0 \le t < 1$: `STATUS_COLD_START_NO_LAGS`
    - Hours $1 \le t < 18$: `STATUS_COLD_START_SHORT_HISTORY`
    - Hours $18 \le t < 168$: `STATUS_COLD_START_ACCUMULATING_7DAY_BUFFER`
    - Hours $t \ge 168$: `STATUS_READY_FOR_INFERENCE` (subject to ambient & nameplate availability).

### 5.4 Exact Timestamp-Delta vs. `.shift(k)` Alignment
In training, Pandas `.shift(k)` was used because historical outages were pre-filtered or contiguous.  
In live operation, if network or power drops for 3 hours, row index $-1$ is $t - 4\text{h}$, not $t - 1\text{h}$.  
**The live feature generator must query by exact timestamp difference:**
- $t - 1\text{h} = t - \text{Timedelta(hours=1)}$
- $t - 24\text{h} = t - \text{Timedelta(hours=24)}$
- $t - 168\text{h} = t - \text{Timedelta(hours=168)}$

If no record exists for that exact hour, the lag feature evaluates strictly to `None` (missing). No forward-filling, backward-filling, or synthetic substitution is permitted.

---

## 6. Implementation Readiness Architecture for Stage 3

When authorized to implement Stage 3, the codebase will be structured as follows without altering any completed milestones:

```
[Hourly Telemetry (SQLite)] ──> [Feature Extraction Service] ──> [Live Feature Store (SQLite)]
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
     [Transformer Config]                             [Indian Festival Table]
  (kVA, pf, Nameplate Verified)                      (Annual Holiday Calendar)
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                             [Data Readiness Evaluator]
                                         │
        ┌────────────────────────────────┼──────────────────────────────┐
        ▼                                ▼                              ▼
[STATUS_READY]               [STATUS_BLOCKED_AMBIENT]         [STATUS_COLD_START]
(All 39 Features Valid)      (Ambient Temp = None)            (History < 168 Hours)
```

### 6.1 Planned Components
1. **`src/backend/feature_service.py`:**
   - Pure, stateless feature calculation functions matching `FeatureEngineer`.
   - Timezone localization to `Asia/Kolkata`.
   - Exact timestamp-delta query engine for lags and rolling windows.
   - Comprehensive readiness validator returning detailed missing-feature diagnostics.
2. **`LiveFeatures` ORM Model (`src/backend/database.py`):**
   - SQLite table `live_features` storing the 39 generated values, calculation timestamp, transformer ID, readiness status, and missing feature list.
3. **Pydantic Schemas (`src/backend/schemas.py`):**
   - `FeatureVectorResponse`: Complete 39-feature schema matching model expectations.
   - `FeatureReadinessResponse`: Status indicator (`READY`, `BLOCKED_MISSING_AMBIENT`, `BLOCKED_MISSING_NAMEPLATE`, `COLD_START`).
4. **FastAPI Endpoints (`src/backend/main.py`):**
   - `POST /api/features/{transformer_id}/generate`: Generates and persists the 39-feature vector for a completed hour.
   - `GET /api/features/{transformer_id}/latest`: Returns the latest feature vector and readiness status.
   - `GET /api/features/{transformer_id}/readiness`: Inspects whether prerequisites for inference are met.

---

## 7. Implementation-Readiness Checklist

Before Stage 3 code can be written, verify the readiness items below:

| Checklist Item | Requirement | Verification Status | Action Required |
| :--- | :--- | :---: | :--- |
| **1. 39-Feature List & Order** | Exactly matches all 8 trained `.joblib` model `feature_names_in_` | **VERIFIED** | None (100% match confirmed) |
| **2. Feature Breakdown** | Official $6 + 10 + 8 + 7 + 8 = 39$ reconciled | **VERIFIED** | Adopted in documentation |
| **3. Timezone Logic** | Localization to `Asia/Kolkata` before calendar/festival calculation | **VERIFIED** | Ready to implement via `zoneinfo` |
| **4. Rolling & Lag Alignment** | Exact timestamp-delta lookup; zero row-index `.shift()` | **VERIFIED** | Ready to implement |
| **5. 168-Hour Buffer Rule** | Enforce cold-start handling for $< 168\text{h}$ of history | **VERIFIED** | Ready to implement |
| **6. Ambient Temperature Policy**| Never substitute oil temp; never use weather API; evaluate to `None` | **CONFIRMED** | Governed by user decision |
| **7. Nameplate Capacity Policy**| Mandatory & unconfigured until hardware owner confirms; evaluate to `None` | **CONFIRMED** | Governed by user decision |
| **8. Zero File Mutation Rule** | No changes to ML models, raw CSVs, or Milestones 1–4 code | **VERIFIED** | Fully respected |

---

---

## 8. Implementation & Verification Summary

Milestone 5 Stage 3 (Live ML Feature Generation) has been **fully implemented and verified**.

### 8.1 Delivered Artifacts

1. **`src/backend/feature_service.py`**:
   - `FEATURE_NAMES_39`: Authoritative 39-feature vector matching all 8 models.
   - `FESTIVAL_LIST` and `FESTIVAL_CALENDAR`: Indian festival calendar (2013–2027) reproducing `FeatureEngineer`.
   - `localize_to_ist()`: `Asia/Kolkata` (UTC+05:30) timezone conversion with DST-free offset.
   - `compute_calendar_cyclical_features()`: 10 cyclical calendar components (`hour_sin`, `hour_cos`, `day_sin`, `day_cos`, `month_sin`, `month_cos`).
   - `compute_festival_features()`: Binary festival indicator and 7 one-hot dummies.
   - `extract_lags()`: Exact timestamp delta queries for 1h, 24h, and 168h lags (never bridging missing gaps).
   - `extract_rolling_windows()`: Rolling windows over past $[t-W, t-1\text{h}]$, strictly excluding time $t$.
   - `generate_live_features()`: Complete pipeline generating, validating, and persisting live 39-feature vectors.
   - `get_feature_readiness()`: Diagnostic service reporting gating statuses (`READY_FOR_INFERENCE`, `STATUS_COLD_START_INSUFFICIENT_HISTORY`, `STATUS_BLOCKED_MISSING_AMBIENT`, `STATUS_BLOCKED_MISSING_NAMEPLATE`).
   - `set_transformer_spec()`, `get_transformer_spec()`, `reset_transformer_specs()`: Verified nameplate registry.

2. **Database Table `live_features` (`src/backend/database.py`)**:
   - Complete 39-feature column schema with composite unique index `(transformer_id, hour_timestamp)`.
   - Stores calculation metadata, readiness status flag (`is_ready_for_inference`), missing features JSON array, and diagnostic reason text.

3. **FastAPI Endpoints (`src/backend/main.py`)**:
   - `POST /api/features/{transformer_id}/generate`: Generates/updates 39-feature vector for specified or latest completed hour.
   - `GET /api/features/{transformer_id}/latest`: Returns the latest feature vector and readiness status.
   - `GET /api/features/{transformer_id}/readiness`: Returns inference gating diagnostics and missing prerequisites.
   - `POST /api/transformers/{transformer_id}/specs`: Secure endpoint for hardware owner to configure physical transformer nameplate parameters.

4. **Comprehensive Test Suite (`tests/test_feature_service.py`)**:
   - 12 comprehensive unit tests covering all edge cases, exact historical parity, and API endpoints.

---

## 9. Test Verification Results

All 29 tests across the entire backend repository pass with 100% success:

| Test Module | Tests | Status | Key Verifications |
| :--- | :---: | :---: | :--- |
| `tests/test_backend_api.py` | 9 | **PASS** | Telemetry ingestion, health API, SQLite storage, boundary validation |
| `tests/test_aggregation_service.py` | 8 | **PASS** | Hourly averages, cold start, current hour exclusion, idempotent updates |
| `tests/test_feature_service.py` | 12 | **PASS** | 39-feature ordering, cyclical math, festival dummies, exact lags, missing ambient gating, unverified nameplate gating, cold-start lifecycle, historical numerical parity against real raw data |
| **Total** | **29** | **PASS** | **Zero failures, zero regressions** |

### Numerical Parity Test Outcome
In `test_11_numerical_parity_with_historical_feature_engineer`, an uninterrupted 169-hour sequence from raw historical dataset `transformer_hourly_festival_with_temperature.csv` was processed through `generate_live_features()`. Every single one of the 39 generated features matched the historical `FeatureEngineer` output from `data/processed/t1_train.csv` with zero errors within strict tolerances ($< 0.001$ for float features, $< 0.05\%$ for rounded historical `load_pct`).

---

*Milestone 5 Stage 3 is complete, validated, and ready for Stage 4 (Real-Time ML Inference & Overload Scoring).*
