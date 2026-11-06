"""
Live ML Feature Generation Service.
Constructs the authoritative 39-feature vector required by all eight trained Random Forest models,
overload classifiers, and anomaly detectors.

Key Operational Guarantees:
1. Exact Feature Ordering: Preserves the exact 39-feature names and sequence (1 to 39).
2. Timezone Normalization: Converts timestamps to Asia/Kolkata (IST, UTC+05:30) before extracting
   temporal, cyclical, and festival features.
3. Exact Timestamp-Delta Lags: Fetches lags by exact datetime subtraction (t - 1h, t - 24h, t - 168h).
   Never calculates across missing-hour gaps.
4. Cold-Start Life-Cycle: Strictly enforces the 168-hour history requirement.
5. Zero Ambient Substitution: Never substitutes oil temperature for ambient temperature.
   Never uses third-party weather APIs unless explicitly configured.
6. Nameplate Integrity: Never calculates load percentage without confirmed nameplate capacity.
7. Model Safety: Blocks inference whenever feature vectors are incomplete or contain NaNs/None.
"""

from collections import OrderedDict
from datetime import datetime, timedelta, timezone
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo
import numpy as np
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from src.backend.config import settings
from src.backend.database import HourlyTelemetry, LiveFeatures

logger = logging.getLogger("transformer_backend.feature_service")

# Timezone standard
IST = ZoneInfo("Asia/Kolkata")

# Exact Authoritative 39-Feature List in Model Order
FEATURE_NAMES_39: List[str] = [
    "power_W",
    "oil_temp_C",
    "load_pct",
    "ambient_temp_C",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
    "is_festival",
    "fest_christmas",
    "fest_diwali",
    "fest_dussehra",
    "fest_eid-ul-fitr",
    "fest_holi",
    "fest_new_year",
    "fest_pongal_sankranti",
    "load_lag_1h",
    "load_lag_24h",
    "load_lag_168h",
    "oil_temp_lag_1h",
    "oil_temp_lag_24h",
    "ambient_temp_lag_1h",
    "ambient_temp_lag_24h",
    "temp_gradient",
    "temp_gradient_lag_1h",
    "load_rolling_mean_24h",
    "load_rolling_std_24h",
    "load_rolling_min_24h",
    "load_rolling_max_24h",
    "load_rolling_mean_168h",
    "load_rolling_std_168h",
    "ambient_rolling_mean_24h",
    "oil_temp_rolling_mean_24h",
]

# Supported Festivals List from FeatureEngineer
FESTIVAL_LIST: List[str] = [
    "Christmas",
    "Diwali",
    "Dussehra",
    "Eid-ul-Fitr",
    "Holi",
    "New Year",
    "Pongal/Sankranti",
]

# Verified Indian Holiday Calendar Table (Historical 2013-2017 & Modern 2025-2027)
FESTIVAL_CALENDAR: Dict[str, str] = {
    # 2013-2014
    "2013-12-25": "Christmas",
    "2013-12-26": "Christmas",
    "2013-12-31": "New Year",
    "2014-01-01": "New Year",
    "2014-01-14": "Pongal/Sankranti",
    "2014-01-15": "Pongal/Sankranti",
    "2014-03-17": "Holi",
    "2014-03-18": "Holi",
    "2014-07-29": "Eid-ul-Fitr",
    "2014-07-30": "Eid-ul-Fitr",
    "2014-10-03": "Dussehra",
    "2014-10-04": "Dussehra",
    "2014-10-23": "Diwali",
    "2014-10-24": "Diwali",
    # 2014-2015
    "2014-12-25": "Christmas",
    "2014-12-26": "Christmas",
    "2014-12-31": "New Year",
    "2015-01-01": "New Year",
    "2015-01-15": "Pongal/Sankranti",
    "2015-01-16": "Pongal/Sankranti",
    "2015-03-06": "Holi",
    "2015-03-07": "Holi",
    "2015-07-18": "Eid-ul-Fitr",
    "2015-07-19": "Eid-ul-Fitr",
    "2015-10-22": "Dussehra",
    "2015-10-23": "Dussehra",
    "2015-11-11": "Diwali",
    "2015-11-12": "Diwali",
    # 2015-2016
    "2015-12-25": "Christmas",
    "2015-12-26": "Christmas",
    "2015-12-31": "New Year",
    "2016-01-01": "New Year",
    "2016-01-15": "Pongal/Sankranti",
    "2016-01-16": "Pongal/Sankranti",
    "2016-03-24": "Holi",
    "2016-03-25": "Holi",
    "2016-07-07": "Eid-ul-Fitr",
    "2016-07-08": "Eid-ul-Fitr",
    "2016-10-11": "Dussehra",
    "2016-10-12": "Dussehra",
    "2016-10-30": "Diwali",
    "2016-10-31": "Diwali",
    # 2016-2017
    "2016-12-25": "Christmas",
    "2016-12-26": "Christmas",
    "2016-12-31": "New Year",
    "2017-01-01": "New Year",
    "2017-01-14": "Pongal/Sankranti",
    "2017-01-15": "Pongal/Sankranti",
    "2017-03-13": "Holi",
    "2017-03-14": "Holi",
    "2017-06-26": "Eid-ul-Fitr",
    "2017-06-27": "Eid-ul-Fitr",
    "2017-09-30": "Dussehra",
    "2017-10-01": "Dussehra",
    "2017-10-19": "Diwali",
    "2017-10-20": "Diwali",
    "2017-12-25": "Christmas",
    "2017-12-26": "Christmas",
    "2017-12-31": "New Year",
    # 2025-2027 Projections
    "2025-01-01": "New Year",
    "2025-01-14": "Pongal/Sankranti",
    "2025-03-14": "Holi",
    "2025-03-31": "Eid-ul-Fitr",
    "2025-10-02": "Dussehra",
    "2025-10-20": "Diwali",
    "2025-12-25": "Christmas",
    "2026-01-01": "New Year",
    "2026-01-14": "Pongal/Sankranti",
    "2026-01-15": "Pongal/Sankranti",
    "2026-03-04": "Holi",
    "2026-03-20": "Eid-ul-Fitr",
    "2026-10-20": "Dussehra",
    "2026-11-08": "Diwali",
    "2026-12-25": "Christmas",
    "2027-01-01": "New Year",
    "2027-01-14": "Pongal/Sankranti",
}

# In-memory registry of confirmed transformer nameplate specifications
# Stored per transformer ID, defaulting to unconfigured/unverified
_TRANSFORMER_SPECS: Dict[str, Dict[str, Any]] = {
    "T1": {"rated_kVA": None, "phases": None, "rated_pf": 0.90, "verified": False},
    "T2": {"rated_kVA": None, "phases": None, "rated_pf": 0.90, "verified": False},
    "T3": {"rated_kVA": None, "phases": None, "rated_pf": 0.90, "verified": False},
}


def get_transformer_spec(transformer_id: str) -> Dict[str, Any]:
    """Retrieves current nameplate configuration for a transformer."""
    tid = transformer_id.strip().upper()
    return _TRANSFORMER_SPECS.get(
        tid,
        {"rated_kVA": None, "phases": None, "rated_pf": 0.90, "verified": False},
    )


def set_transformer_spec(
    transformer_id: str,
    rated_kVA: float,
    phases: int = 1,
    rated_pf: float = 0.90,
    verified: bool = True,
) -> None:
    """
    Registers confirmed physical transformer nameplate specifications.
    Must ONLY be invoked when hardware owner verifies actual nameplate rating.
    """
    tid = transformer_id.strip().upper()
    _TRANSFORMER_SPECS[tid] = {
        "rated_kVA": float(rated_kVA),
        "phases": int(phases),
        "rated_pf": float(rated_pf),
        "verified": bool(verified),
    }
    logger.info(
        f"Updated nameplate specs for {tid}: {rated_kVA} kVA, {phases}-phase, pf={rated_pf}, verified={verified}"
    )


def reset_transformer_specs() -> None:
    """Resets transformer specs back to unconfigured state (for unit testing and safety)."""
    for tid in _TRANSFORMER_SPECS:
        _TRANSFORMER_SPECS[tid] = {
            "rated_kVA": None,
            "phases": None,
            "rated_pf": 0.90,
            "verified": False,
        }


def localize_to_ist(dt: datetime) -> datetime:
    """
    Converts a UTC or naive timestamp to Asia/Kolkata (IST, UTC+05:30).
    Historical ML models were trained on datetime_IST; temporal features MUST use IST.
    """
    if dt.tzinfo is None:
        # Naive datetime from SQLite or payload: assume UTC
        utc_dt = dt.replace(tzinfo=timezone.utc)
    else:
        utc_dt = dt.astimezone(timezone.utc)
    return utc_dt.astimezone(IST)


def compute_calendar_cyclical_features(dt_ist: datetime) -> Dict[str, float]:
    """
    Computes 10 calendar and cyclical trigonometric features matching FeatureEngineer:
    - hour (0-23)
    - day_of_week (0-6, Monday=0)
    - month (1-12)
    - is_weekend (1 if Sat/Sun, else 0)
    - hour_sin, hour_cos (24h cycle)
    - day_sin, day_cos (7d cycle)
    - month_sin, month_cos (12m cycle, angle from month-1)
    """
    hour = float(dt_ist.hour)
    day_of_week = float(dt_ist.weekday())
    month = float(dt_ist.month)
    is_weekend = 1.0 if day_of_week in (5.0, 6.0) else 0.0

    hour_sin = float(np.sin(2 * np.pi * hour / 24.0))
    hour_cos = float(np.cos(2 * np.pi * hour / 24.0))
    day_sin = float(np.sin(2 * np.pi * day_of_week / 7.0))
    day_cos = float(np.cos(2 * np.pi * day_of_week / 7.0))
    month_sin = float(np.sin(2 * np.pi * (month - 1.0) / 12.0))
    month_cos = float(np.cos(2 * np.pi * (month - 1.0) / 12.0))

    return {
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": is_weekend,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "day_sin": day_sin,
        "day_cos": day_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
    }


def compute_festival_features(dt_ist: datetime) -> Dict[str, float]:
    """
    Computes 8 festival indicators (1 binary is_festival + 7 one-hot dummies)
    matching FeatureEngineer and historical festival dictionary.
    """
    date_str = dt_ist.strftime("%Y-%m-%d")
    matched_festival = FESTIVAL_CALENDAR.get(date_str, "Normal day")

    is_festival = 1.0 if matched_festival != "Normal day" else 0.0
    festival_features = {"is_festival": is_festival}

    for fest in FESTIVAL_LIST:
        clean_name = fest.replace("/", "_").replace(" ", "_").lower()
        col_name = f"fest_{clean_name}"
        festival_features[col_name] = 1.0 if matched_festival == fest else 0.0

    return festival_features


def extract_lags(
    db: Session,
    transformer_id: str,
    target_hour: datetime,
) -> Dict[str, Optional[float]]:
    """
    Extracts autoregressive lags by EXACT timestamp delta:
    - 1h lag: target_hour - 1 hour
    - 24h lag: target_hour - 24 hours
    - 168h lag: target_hour - 168 hours
    
    If the exact matching hour is missing from hourly_telemetry,
    evaluates strictly to None. Never steps across missing gaps.
    """
    tid = transformer_id.strip().upper()
    th = target_hour.replace(tzinfo=None) if target_hour.tzinfo else target_hour
    t_1h = th - timedelta(hours=1)
    t_24h = th - timedelta(hours=24)
    t_168h = th - timedelta(hours=168)

    # Query exact matching records from hourly_telemetry
    lags_query = (
        select(HourlyTelemetry)
        .where(
            HourlyTelemetry.transformer_id == tid,
            HourlyTelemetry.hour_start.in_([t_1h, t_24h, t_168h]),
        )
    )
    records = {r.hour_start: r for r in db.execute(lags_query).scalars().all()}

    rec_1h = records.get(t_1h)
    rec_24h = records.get(t_24h)
    rec_168h = records.get(t_168h)

    load_lag_1h = float(rec_1h.power_W) if rec_1h else None
    oil_temp_lag_1h = float(rec_1h.oil_temperature_C) if rec_1h else None
    ambient_temp_lag_1h = (
        float(rec_1h.ambient_temperature_C)
        if (rec_1h and rec_1h.ambient_temperature_C is not None)
        else None
    )

    load_lag_24h = float(rec_24h.power_W) if rec_24h else None
    oil_temp_lag_24h = float(rec_24h.oil_temperature_C) if rec_24h else None
    ambient_temp_lag_24h = (
        float(rec_24h.ambient_temperature_C)
        if (rec_24h and rec_24h.ambient_temperature_C is not None)
        else None
    )

    load_lag_168h = float(rec_168h.power_W) if rec_168h else None

    # Thermal gradient lag at t-1h: oil_temp_lag_1h - ambient_temp_lag_1h
    if oil_temp_lag_1h is not None and ambient_temp_lag_1h is not None:
        temp_gradient_lag_1h = oil_temp_lag_1h - ambient_temp_lag_1h
    else:
        temp_gradient_lag_1h = None

    return {
        "load_lag_1h": load_lag_1h,
        "load_lag_24h": load_lag_24h,
        "load_lag_168h": load_lag_168h,
        "oil_temp_lag_1h": oil_temp_lag_1h,
        "oil_temp_lag_24h": oil_temp_lag_24h,
        "ambient_temp_lag_1h": ambient_temp_lag_1h,
        "ambient_temp_lag_24h": ambient_temp_lag_24h,
        "temp_gradient_lag_1h": temp_gradient_lag_1h,
    }


def extract_rolling_windows(
    db: Session,
    transformer_id: str,
    target_hour: datetime,
    min_periods_24: int = 18,
    min_periods_168: int = 84,
) -> Dict[str, Optional[float]]:
    """
    Computes rolling statistics strictly over past windows [target_hour - W, target_hour - 1h].
    The current observation at target_hour is strictly excluded (reproducing shift(1)).
    Requires at least min_periods valid observations; otherwise evaluates strictly to None.
    """
    tid = transformer_id.strip().upper()
    th = target_hour.replace(tzinfo=None) if target_hour.tzinfo else target_hour
    start_24h = th - timedelta(hours=24)
    start_168h = th - timedelta(hours=168)
    end_past = th - timedelta(hours=1)

    # Query all historical records in the maximum 168h lookback window
    query = (
        select(HourlyTelemetry)
        .where(
            HourlyTelemetry.transformer_id == tid,
            HourlyTelemetry.hour_start >= start_168h,
            HourlyTelemetry.hour_start <= end_past,
        )
        .order_by(HourlyTelemetry.hour_start.asc())
    )
    history_records = list(db.execute(query).scalars().all())

    # Filter records within the 24h past window
    recs_24h = [r for r in history_records if r.hour_start >= start_24h]

    # --- 24-Hour Power Rolling Windows ---
    powers_24 = [float(r.power_W) for r in recs_24h if r.power_W is not None]
    if len(powers_24) >= min_periods_24:
        load_rolling_mean_24h = float(np.mean(powers_24))
        load_rolling_std_24h = float(np.std(powers_24, ddof=1))
        load_rolling_min_24h = float(np.min(powers_24))
        load_rolling_max_24h = float(np.max(powers_24))
    else:
        load_rolling_mean_24h = None
        load_rolling_std_24h = None
        load_rolling_min_24h = None
        load_rolling_max_24h = None

    # --- 24-Hour Oil Temperature Rolling Window ---
    oil_temps_24 = [float(r.oil_temperature_C) for r in recs_24h if r.oil_temperature_C is not None]
    if len(oil_temps_24) >= min_periods_24:
        oil_temp_rolling_mean_24h = float(np.mean(oil_temps_24))
    else:
        oil_temp_rolling_mean_24h = None

    # --- 24-Hour Ambient Temperature Rolling Window ---
    ambient_temps_24 = [
        float(r.ambient_temperature_C)
        for r in recs_24h
        if r.ambient_temperature_C is not None
    ]
    if len(ambient_temps_24) >= min_periods_24:
        ambient_rolling_mean_24h = float(np.mean(ambient_temps_24))
    else:
        ambient_rolling_mean_24h = None

    # --- 168-Hour Power Rolling Windows ---
    powers_168 = [float(r.power_W) for r in history_records if r.power_W is not None]
    if len(powers_168) >= min_periods_168:
        load_rolling_mean_168h = float(np.mean(powers_168))
        load_rolling_std_168h = float(np.std(powers_168, ddof=1))
    else:
        load_rolling_mean_168h = None
        load_rolling_std_168h = None

    return {
        "load_rolling_mean_24h": load_rolling_mean_24h,
        "load_rolling_std_24h": load_rolling_std_24h,
        "load_rolling_min_24h": load_rolling_min_24h,
        "load_rolling_max_24h": load_rolling_max_24h,
        "oil_temp_rolling_mean_24h": oil_temp_rolling_mean_24h,
        "ambient_rolling_mean_24h": ambient_rolling_mean_24h,
        "load_rolling_mean_168h": load_rolling_mean_168h,
        "load_rolling_std_168h": load_rolling_std_168h,
    }


def identify_affected_downstream_timestamps(
    db: Session,
    transformer_id: str,
    modified_hour: datetime,
) -> List[datetime]:
    """
    Identifies all future feature timestamps affected when hour H is modified:
    - Target hour H itself: direct sensor measurements (power, temp, voltage, current) changed.
    - 1-hour lag: H + 1h
    - 24-hour lag: H + 24h
    - 168-hour lag: H + 168h
    - 24-hour rolling window [T-24h, T-1h]: includes H for all T in [H+1h, H+24h]
    - 168-hour rolling window [T-168h, T-1h]: includes H for all T in [H+1h, H+168h]
    - Thermal gradient lag: H + 1h
    
    Returns all existing hourly timestamps in [H, H + 168h] in strictly ascending chronological order.
    """
    tid = transformer_id.strip().upper()
    h_clean = modified_hour.replace(minute=0, second=0, microsecond=0, tzinfo=None)
    max_affected = h_clean + timedelta(hours=168)

    query = (
        select(HourlyTelemetry.hour_start)
        .where(
            HourlyTelemetry.transformer_id == tid,
            HourlyTelemetry.hour_start >= h_clean,
            HourlyTelemetry.hour_start <= max_affected,
        )
        .order_by(HourlyTelemetry.hour_start.asc())
    )
    return list(db.execute(query).scalars().all())


def generate_live_features(
    db: Session,
    transformer_id: str,
    target_hour: Optional[datetime] = None,
) -> LiveFeatures:
    """
    Assembles the exact 39-feature vector for a completed hour from hourly_telemetry.
    Validates completeness, identifies missing features, records readiness status,
    and idempotently commits the feature vector to SQLite table live_features.

    Args:
        db: Active SQLAlchemy transactional database session.
        transformer_id: Identifier (e.g. 'T1', 'T2', 'T3').
        target_hour: The specific completed hour start. If None, uses the latest
                     available completed hour for that transformer.

    Returns:
        LiveFeatures model instance containing features and readiness metadata.
    """
    clean_tid = transformer_id.strip().upper()

    # 1. Resolve Target Hour
    if target_hour is None:
        latest_hourly = (
            db.query(HourlyTelemetry)
            .filter(HourlyTelemetry.transformer_id == clean_tid)
            .order_by(desc(HourlyTelemetry.hour_start))
            .first()
        )
        if not latest_hourly:
            raise ValueError(f"No hourly telemetry found for transformer {clean_tid}.")
        target_hour = latest_hourly.hour_start
        current_obs = latest_hourly
    else:
        th_query = target_hour.replace(tzinfo=None) if target_hour.tzinfo else target_hour
        current_obs = (
            db.query(HourlyTelemetry)
            .filter(
                HourlyTelemetry.transformer_id == clean_tid,
                HourlyTelemetry.hour_start == th_query,
            )
            .first()
        )
        if not current_obs:
            raise ValueError(
                f"Hourly observation for {clean_tid} at {target_hour.isoformat()} does not exist."
            )

    # 2. Direct Telemetry Features
    power_W = float(current_obs.power_W)
    oil_temp_C = float(current_obs.oil_temperature_C)
    ambient_temp_C = (
        float(current_obs.ambient_temperature_C)
        if current_obs.ambient_temperature_C is not None
        else None
    )

    # Thermal gradient at time t: T_oil - T_amb
    if oil_temp_C is not None and ambient_temp_C is not None:
        temp_gradient = oil_temp_C - ambient_temp_C
    else:
        temp_gradient = None

    # Load percentage calculation
    spec = get_transformer_spec(clean_tid)
    if spec.get("verified") and spec.get("rated_kVA"):
        rated_kVA = float(spec["rated_kVA"])
        rated_pf = float(spec.get("rated_pf", 0.90))
        # Active capacity = S_rated (kVA) * 1000 * pf
        active_capacity_W = rated_kVA * 1000.0 * rated_pf
        load_pct = (power_W / active_capacity_W) * 100.0 if active_capacity_W > 0 else None
    else:
        # Strictly unconfigured/unverified -> None
        load_pct = None

    # 3. Calendar & Cyclical Features (IST localized)
    dt_ist = localize_to_ist(target_hour)
    cal_features = compute_calendar_cyclical_features(dt_ist)
    fest_features = compute_festival_features(dt_ist)

    # 4. Autoregressive Lags & Rolling Windows
    lag_features = extract_lags(db=db, transformer_id=clean_tid, target_hour=target_hour)
    rolling_features = extract_rolling_windows(
        db=db, transformer_id=clean_tid, target_hour=target_hour
    )

    # 5. Assemble Dictionary in Exact 39 Model Order
    raw_feature_map = {
        "power_W": power_W,
        "oil_temp_C": oil_temp_C,
        "load_pct": load_pct,
        "ambient_temp_C": ambient_temp_C,
        "hour": cal_features["hour"],
        "day_of_week": cal_features["day_of_week"],
        "month": cal_features["month"],
        "is_weekend": cal_features["is_weekend"],
        "hour_sin": cal_features["hour_sin"],
        "hour_cos": cal_features["hour_cos"],
        "day_sin": cal_features["day_sin"],
        "day_cos": cal_features["day_cos"],
        "month_sin": cal_features["month_sin"],
        "month_cos": cal_features["month_cos"],
        "is_festival": fest_features["is_festival"],
        "fest_christmas": fest_features["fest_christmas"],
        "fest_diwali": fest_features["fest_diwali"],
        "fest_dussehra": fest_features["fest_dussehra"],
        "fest_eid-ul-fitr": fest_features["fest_eid-ul-fitr"],
        "fest_holi": fest_features["fest_holi"],
        "fest_new_year": fest_features["fest_new_year"],
        "fest_pongal_sankranti": fest_features["fest_pongal_sankranti"],
        "load_lag_1h": lag_features["load_lag_1h"],
        "load_lag_24h": lag_features["load_lag_24h"],
        "load_lag_168h": lag_features["load_lag_168h"],
        "oil_temp_lag_1h": lag_features["oil_temp_lag_1h"],
        "oil_temp_lag_24h": lag_features["oil_temp_lag_24h"],
        "ambient_temp_lag_1h": lag_features["ambient_temp_lag_1h"],
        "ambient_temp_lag_24h": lag_features["ambient_temp_lag_24h"],
        "temp_gradient": temp_gradient,
        "temp_gradient_lag_1h": lag_features["temp_gradient_lag_1h"],
        "load_rolling_mean_24h": rolling_features["load_rolling_mean_24h"],
        "load_rolling_std_24h": rolling_features["load_rolling_std_24h"],
        "load_rolling_min_24h": rolling_features["load_rolling_min_24h"],
        "load_rolling_max_24h": rolling_features["load_rolling_max_24h"],
        "load_rolling_mean_168h": rolling_features["load_rolling_mean_168h"],
        "load_rolling_std_168h": rolling_features["load_rolling_std_168h"],
        "ambient_rolling_mean_24h": rolling_features["ambient_rolling_mean_24h"],
        "oil_temp_rolling_mean_24h": rolling_features["oil_temp_rolling_mean_24h"],
    }

    # Verify all 39 keys are present
    assert len(raw_feature_map) == 39, f"Feature count mismatch: {len(raw_feature_map)} != 39"

    # 6. Readiness Evaluation & Missing Data Diagnostics
    missing_features = [k for k in FEATURE_NAMES_39 if raw_feature_map[k] is None]

    if not missing_features:
        readiness_status = "READY_FOR_INFERENCE"
        is_ready = 1
        status_reason = "All 39 features successfully generated and validated. Ready for inference."
    else:
        is_ready = 0
        reasons = []
        if any("ambient" in f or "gradient" in f for f in missing_features):
            reasons.append(
                "Ambient temperature channel is missing/unconfigured (affects 6 features). "
                "Never substitute oil temperature."
            )
        if "load_pct" in missing_features:
            reasons.append(
                "Transformer nameplate capacity is unverified/unconfigured. "
                "Load percentage cannot be calculated."
            )
        if "load_lag_168h" in missing_features or "load_rolling_mean_168h" in missing_features:
            reasons.append(
                "Insufficient historical telemetry (<168 hours of continuous uptime). "
                "Cold-start buffer is accumulating."
            )
        if any(f in ("load_lag_1h", "load_lag_24h") for f in missing_features):
            reasons.append(
                "Missing-hour gap detected in historical telemetry. "
                "Lags cannot be computed across gaps."
            )

        status_reason = " | ".join(reasons)

        if "load_lag_168h" in missing_features or "load_rolling_mean_168h" in missing_features:
            readiness_status = "STATUS_COLD_START_INSUFFICIENT_HISTORY"
        elif "ambient_temp_C" in missing_features:
            readiness_status = "STATUS_BLOCKED_MISSING_AMBIENT"
        elif "load_pct" in missing_features:
            readiness_status = "STATUS_BLOCKED_MISSING_NAMEPLATE"
        else:
            readiness_status = "STATUS_INCOMPLETE_FEATURES"

    # 7. Idempotent Database Upsert in live_features
    th_store = target_hour.replace(tzinfo=None) if target_hour.tzinfo else target_hour
    existing = (
        db.query(LiveFeatures)
        .filter(
            LiveFeatures.transformer_id == clean_tid,
            LiveFeatures.hour_timestamp == th_store,
        )
        .first()
    )

    if existing:
        # Update existing record
        existing.readiness_status = readiness_status
        existing.is_ready_for_inference = is_ready
        existing.missing_features = json.dumps(missing_features)
        existing.status_reason = status_reason
        for k in FEATURE_NAMES_39:
            val = raw_feature_map[k]
            # Map hyphenated fest_eid-ul-fitr to ORM attribute fest_eid_ul_fitr
            attr_name = "fest_eid_ul_fitr" if k == "fest_eid-ul-fitr" else k
            setattr(existing, attr_name, val)
        db_record = existing
    else:
        # Create new record
        orm_kwargs = {
            "transformer_id": clean_tid,
            "hour_timestamp": th_store,
            "readiness_status": readiness_status,
            "is_ready_for_inference": is_ready,
            "missing_features": json.dumps(missing_features),
            "status_reason": status_reason,
            "created_at": datetime.now(timezone.utc),
        }
        for k in FEATURE_NAMES_39:
            val = raw_feature_map[k]
            attr_name = "fest_eid_ul_fitr" if k == "fest_eid-ul-fitr" else k
            orm_kwargs[attr_name] = val

        db_record = LiveFeatures(**orm_kwargs)
        db.add(db_record)

    db.commit()
    db.refresh(db_record)

    logger.info(
        f"Generated 39-feature vector for {clean_tid} at {target_hour.isoformat()} "
        f"- Status: {readiness_status} (Missing: {len(missing_features)} features)"
    )
    return db_record


def get_latest_features(
    db: Session,
    transformer_id: str,
) -> Optional[LiveFeatures]:
    """Retrieves the latest generated feature vector for a transformer."""
    clean_tid = transformer_id.strip().upper()
    return (
        db.query(LiveFeatures)
        .filter(LiveFeatures.transformer_id == clean_tid)
        .order_by(desc(LiveFeatures.hour_timestamp))
        .first()
    )


def get_feature_readiness(
    db: Session,
    transformer_id: str,
    target_hour: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Evaluates whether live features are available and ready for inference,
    returning detailed diagnostics and blocker status without altering data.
    """
    clean_tid = transformer_id.strip().upper()

    if target_hour is None:
        latest = get_latest_features(db, clean_tid)
        if not latest:
            # Check hourly telemetry availability
            hourly_count = (
                db.query(HourlyTelemetry)
                .filter(HourlyTelemetry.transformer_id == clean_tid)
                .count()
            )
            spec = get_transformer_spec(clean_tid)
            return {
                "transformer_id": clean_tid,
                "hour_timestamp": None,
                "readiness_status": "NO_FEATURES_GENERATED",
                "is_ready_for_inference": False,
                "total_features_count": 39,
                "valid_features_count": 0,
                "missing_features_count": 39,
                "missing_features": FEATURE_NAMES_39,
                "hourly_telemetry_hours_recorded": hourly_count,
                "nameplate_configured": bool(spec.get("verified")),
                "status_reason": (
                    f"No features generated yet for {clean_tid}. "
                    f"Hourly telemetry records present: {hourly_count}."
                ),
            }
        record = latest
    else:
        record = (
            db.query(LiveFeatures)
            .filter(
                LiveFeatures.transformer_id == clean_tid,
                LiveFeatures.hour_timestamp == target_hour,
            )
            .first()
        )
        if not record:
            return {
                "transformer_id": clean_tid,
                "hour_timestamp": target_hour.isoformat(),
                "readiness_status": "FEATURE_VECTOR_NOT_FOUND",
                "is_ready_for_inference": False,
                "total_features_count": 39,
                "valid_features_count": 0,
                "missing_features_count": 39,
                "missing_features": FEATURE_NAMES_39,
                "status_reason": f"No feature vector calculated for {clean_tid} at {target_hour.isoformat()}.",
            }

    missing_list = json.loads(record.missing_features) if record.missing_features else []
    return {
        "transformer_id": record.transformer_id,
        "hour_timestamp": record.hour_timestamp.isoformat(),
        "readiness_status": record.readiness_status,
        "is_ready_for_inference": bool(record.is_ready_for_inference),
        "total_features_count": 39,
        "valid_features_count": 39 - len(missing_list),
        "missing_features_count": len(missing_list),
        "missing_features": missing_list,
        "status_reason": record.status_reason,
        "features": record.to_feature_dict(),
    }
