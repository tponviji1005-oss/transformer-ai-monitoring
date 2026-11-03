"""
Database Layer for Transformer Monitoring System.
Implements SQLAlchemy engine, session management, and RawTelemetry ORM table
for persisting high-frequency ESP32 sensor observations in SQLite.
"""

from datetime import datetime, timezone
import os
from typing import Generator
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from src.backend.config import settings

# Ensure data directory exists
os.makedirs(settings.DATA_DIR, exist_ok=True)

# SQLite engine configuration
# check_same_thread=False allows FastAPI multi-threaded request processing
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {},
    echo=settings.DEBUG,
)

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

