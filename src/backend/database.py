"""
Database Layer for Transformer Monitoring System.
Implements SQLAlchemy engine, session management, and RawTelemetry ORM table
for persisting high-frequency ESP32 sensor observations in SQLite.
"""

from datetime import datetime, timezone
import os
from typing import Any, Dict, Generator
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from src.backend.config import settings

# Ensure data directory exists
os.makedirs(settings.DATA_DIR, exist_ok=True)

# SQLite engine configuration
# check_same_thread=False allows FastAPI multi-threaded request processing
# timeout=30.0 prevents database locked errors during concurrent test execution
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30.0} if settings.DATABASE_URL.startswith("sqlite") else {},
    echo=settings.DEBUG,
)

if settings.DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class RawTelemetry(Base):
    """
    Stores raw, unaggregated high-frequency sensor readings received from ESP32.
    Preserves exact hardware timestamps and physical measurements without modification.
    """
    __tablename__ = "raw_telemetry"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    transformer_id = Column(String(20), index=True, nullable=False)
    timestamp = Column(DateTime, index=True, nullable=False)
    voltage_V = Column(Float, nullable=False)
    current_A = Column(Float, nullable=False)
    power_W = Column(Float, nullable=False)
    energy = Column(Float, nullable=True)
    frequency_Hz = Column(Float, nullable=False)
    power_factor = Column(Float, nullable=False)
    oil_temperature_C = Column(Float, nullable=False)
    ambient_temperature_C = Column(Float, nullable=True)  # Strictly nullable; never fabricated
    received_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    def to_dict(self) -> dict:
        """Converts model instance to dictionary representation."""
        return {
            "id": self.id,
            "transformer_id": self.transformer_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "voltage_V": self.voltage_V,
            "current_A": self.current_A,
            "power_W": self.power_W,
            "energy": self.energy,
            "frequency_Hz": self.frequency_Hz,
            "power_factor": self.power_factor,
            "oil_temperature_C": self.oil_temperature_C,
            "ambient_temperature_C": self.ambient_temperature_C,
            "received_at": self.received_at.isoformat() if self.received_at else None,
        }


class HourlyTelemetry(Base):
    """
    Stores 1-hour aggregated telemetry for a transformer.
    Derived from raw high-frequency ESP32 sensor observations.
    Each record represents a completed 60-minute window [hour_start, hour_end).
    Enforces uniqueness on (transformer_id, hour_start) for idempotent aggregation.
    """
    __tablename__ = "hourly_telemetry"
    __table_args__ = (
        UniqueConstraint("transformer_id", "hour_start", name="uq_hourly_transformer_hour"),
    )

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    transformer_id = Column(String(20), index=True, nullable=False)
    hour_start = Column(DateTime, index=True, nullable=False)
    hour_end = Column(DateTime, nullable=False)

    # Hourly arithmetic averages across raw readings
    voltage_V = Column(Float, nullable=False)
    current_A = Column(Float, nullable=False)
    power_W = Column(Float, nullable=False)
    frequency_Hz = Column(Float, nullable=False)
    power_factor = Column(Float, nullable=False)
    oil_temperature_C = Column(Float, nullable=False)
    ambient_temperature_C = Column(Float, nullable=True)  # Strictly nullable; never substituted with oil temp

    reading_count = Column(Integer, nullable=False)
    aggregation_status = Column(String(20), default="completed", nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    def to_dict(self) -> dict:
        """Converts model instance to dictionary representation."""
        return {
            "id": self.id,
            "transformer_id": self.transformer_id,
            "hour_start": self.hour_start.isoformat() if self.hour_start else None,
            "hour_end": self.hour_end.isoformat() if self.hour_end else None,
            "voltage_V": self.voltage_V,
            "current_A": self.current_A,
            "power_W": self.power_W,
            "frequency_Hz": self.frequency_Hz,
            "power_factor": self.power_factor,
            "oil_temperature_C": self.oil_temperature_C,
            "ambient_temperature_C": self.ambient_temperature_C,
            "reading_count": self.reading_count,
            "aggregation_status": self.aggregation_status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class LiveFeatures(Base):
    """
    Stores 39-feature ML vectors calculated from completed hourly observations.
    Preserves exact 39-feature names, ordering, readiness status, and missing feature diagnostics.
    Enforces uniqueness on (transformer_id, hour_timestamp) for idempotent feature storage.
    """
    __tablename__ = "live_features"
    __table_args__ = (
        UniqueConstraint("transformer_id", "hour_timestamp", name="uq_live_features_transformer_hour"),
    )

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    transformer_id = Column(String(20), index=True, nullable=False)
    hour_timestamp = Column(DateTime, index=True, nullable=False)

    # Operational Readiness Status
    readiness_status = Column(String(50), nullable=False)  # e.g. READY_FOR_INFERENCE, BLOCKED_MISSING_AMBIENT
    is_ready_for_inference = Column(Integer, default=0, nullable=False)  # 1 if all 39 valid, else 0
    missing_features = Column(String, nullable=True)  # JSON-encoded array of missing features
    status_reason = Column(String, nullable=True)

    # The 39 Features in Exact Trained Model Order
    power_W = Column(Float, nullable=True)
    oil_temp_C = Column(Float, nullable=True)
    load_pct = Column(Float, nullable=True)
    ambient_temp_C = Column(Float, nullable=True)
    hour = Column(Float, nullable=True)
    day_of_week = Column(Float, nullable=True)
    month = Column(Float, nullable=True)
    is_weekend = Column(Float, nullable=True)
    hour_sin = Column(Float, nullable=True)
    hour_cos = Column(Float, nullable=True)
    day_sin = Column(Float, nullable=True)
    day_cos = Column(Float, nullable=True)
    month_sin = Column(Float, nullable=True)
    month_cos = Column(Float, nullable=True)
    is_festival = Column(Float, nullable=True)
    fest_christmas = Column(Float, nullable=True)
    fest_diwali = Column(Float, nullable=True)
    fest_dussehra = Column(Float, nullable=True)
    fest_eid_ul_fitr = Column("fest_eid-ul-fitr", Float, nullable=True)
    fest_holi = Column(Float, nullable=True)
    fest_new_year = Column(Float, nullable=True)
    fest_pongal_sankranti = Column(Float, nullable=True)
    load_lag_1h = Column(Float, nullable=True)
    load_lag_24h = Column(Float, nullable=True)
    load_lag_168h = Column(Float, nullable=True)
    oil_temp_lag_1h = Column(Float, nullable=True)
    oil_temp_lag_24h = Column(Float, nullable=True)
    ambient_temp_lag_1h = Column(Float, nullable=True)
    ambient_temp_lag_24h = Column(Float, nullable=True)
    temp_gradient = Column(Float, nullable=True)
    temp_gradient_lag_1h = Column(Float, nullable=True)
    load_rolling_mean_24h = Column(Float, nullable=True)
    load_rolling_std_24h = Column(Float, nullable=True)
    load_rolling_min_24h = Column(Float, nullable=True)
    load_rolling_max_24h = Column(Float, nullable=True)
    load_rolling_mean_168h = Column(Float, nullable=True)
    load_rolling_std_168h = Column(Float, nullable=True)
    ambient_rolling_mean_24h = Column(Float, nullable=True)
    oil_temp_rolling_mean_24h = Column(Float, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    def to_feature_dict(self) -> dict:
        """Returns the 39 features in the exact authoritative sequence required by trained models."""
        return {
            "power_W": self.power_W,
            "oil_temp_C": self.oil_temp_C,
            "load_pct": self.load_pct,
            "ambient_temp_C": self.ambient_temp_C,
            "hour": self.hour,
            "day_of_week": self.day_of_week,
            "month": self.month,
            "is_weekend": self.is_weekend,
            "hour_sin": self.hour_sin,
            "hour_cos": self.hour_cos,
            "day_sin": self.day_sin,
            "day_cos": self.day_cos,
            "month_sin": self.month_sin,
            "month_cos": self.month_cos,
            "is_festival": self.is_festival,
            "fest_christmas": self.fest_christmas,
            "fest_diwali": self.fest_diwali,
            "fest_dussehra": self.fest_dussehra,
            "fest_eid-ul-fitr": self.fest_eid_ul_fitr,
            "fest_holi": self.fest_holi,
            "fest_new_year": self.fest_new_year,
            "fest_pongal_sankranti": self.fest_pongal_sankranti,
            "load_lag_1h": self.load_lag_1h,
            "load_lag_24h": self.load_lag_24h,
            "load_lag_168h": self.load_lag_168h,
            "oil_temp_lag_1h": self.oil_temp_lag_1h,
            "oil_temp_lag_24h": self.oil_temp_lag_24h,
            "ambient_temp_lag_1h": self.ambient_temp_lag_1h,
            "ambient_temp_lag_24h": self.ambient_temp_lag_24h,
            "temp_gradient": self.temp_gradient,
            "temp_gradient_lag_1h": self.temp_gradient_lag_1h,
            "load_rolling_mean_24h": self.load_rolling_mean_24h,
            "load_rolling_std_24h": self.load_rolling_std_24h,
            "load_rolling_min_24h": self.load_rolling_min_24h,
            "load_rolling_max_24h": self.load_rolling_max_24h,
            "load_rolling_mean_168h": self.load_rolling_mean_168h,
            "load_rolling_std_168h": self.load_rolling_std_168h,
            "ambient_rolling_mean_24h": self.ambient_rolling_mean_24h,
            "oil_temp_rolling_mean_24h": self.oil_temp_rolling_mean_24h,
        }


class PredictionRecord(Base):
    """
    Stores next-hour ML regression forecasts, overload predictions, and anomaly alerts.
    Includes power forecast, load forecast, overload probabilities/anomaly indicators,
    readiness gating status, and standardized advisory operator actions.
    Enforces uniqueness on (transformer_id, feature_hour_timestamp) for idempotency.
    """
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("transformer_id", "feature_hour_timestamp", name="uq_predictions_transformer_hour"),
    )

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    transformer_id = Column(String(20), index=True, nullable=False)
    feature_hour_timestamp = Column(DateTime, index=True, nullable=False)   # Hour t (observation)
    target_hour_timestamp = Column(DateTime, index=True, nullable=False)    # Hour t + 1h (forecast)

    status = Column(String(30), nullable=False)                            # SUCCESS or UNAVAILABLE
    status_reason = Column(String, nullable=True)

    # Forecasts (Hour t+1)
    predicted_power_W = Column(Float, nullable=True)
    predicted_load_pct = Column(Float, nullable=True)

    # Overload / Anomaly Signals
    overload_probability = Column(Float, nullable=True)                     # T2 only (probability of overload)
    is_overload_predicted = Column(Integer, nullable=True)                  # T2 only (1 if prob >= 0.3382)
    anomaly_score = Column(Float, nullable=True)                           # T1 only (anomaly risk index in [0, 1])
    is_anomaly_detected = Column(Integer, nullable=True)                    # T1 only (1 if anomaly, 0 otherwise)

    # Unified Advisory Alert
    alert_level = Column(String(20), nullable=True)                        # NORMAL, WARNING, CRITICAL
    alert_type = Column(String(40), nullable=True)                         # NORMAL_OPERATION, PREDICTED_OVERLOAD, etc.
    alert_reason = Column(String, nullable=True)
    recommended_action = Column(String, nullable=True)                     # Strictly advisory operator instructions

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "transformer_id": self.transformer_id,
            "feature_hour_timestamp": self.feature_hour_timestamp.isoformat() if self.feature_hour_timestamp else None,
            "target_hour_timestamp": self.target_hour_timestamp.isoformat() if self.target_hour_timestamp else None,
            "status": self.status,
            "status_reason": self.status_reason,
            "predicted_power_W": self.predicted_power_W,
            "predicted_load_pct": self.predicted_load_pct,
            "overload_probability": self.overload_probability,
            "is_overload_predicted": bool(self.is_overload_predicted) if self.is_overload_predicted is not None else None,
            "anomaly_score": self.anomaly_score,
            "is_anomaly_detected": bool(self.is_anomaly_detected) if self.is_anomaly_detected is not None else None,
            "alert_level": self.alert_level,
            "alert_type": self.alert_type,
            "alert_reason": self.alert_reason,
            "recommended_action": self.recommended_action,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


def init_db() -> None:
    """Initializes SQLite database tables."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that provides a transactional database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

