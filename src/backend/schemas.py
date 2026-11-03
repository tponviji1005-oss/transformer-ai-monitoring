"""
Pydantic Schemas for Ingestion, Validation, and API Responses.
Strictly validates incoming sensor signals against electrical and physical boundaries.
Does not invent missing sensor values and keeps ambient temperature explicitly nullable.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class TelemetryIngestRequest(BaseModel):
    """
    Validation schema for incoming ESP32 sensor observations.
    Accepts raw 2-second telemetry packets from remote microcontrollers.
    """
    transformer_id: str = Field(
        ...,
        description="Transformer identifier (e.g. 'T1', 'T2', 'T3')",
        examples=["T2"],
        min_length=1,
        max_length=20,
    )
    timestamp: datetime = Field(
        ...,
        description="Measurement timestamp in ISO-8601 format (e.g. '2026-11-02T23:15:00')",
        examples=["2026-11-02T23:15:00"],
    )
    voltage_V: float = Field(
        ...,
        description="RMS AC voltage in Volts (valid range: 0.0 to 1000.0 V)",
        examples=[415.0],
        ge=0.0,
        le=1000.0,
    )
    current_A: float = Field(
        ...,
        description="RMS AC current in Amperes (valid range: 0.0 to 2000.0 A)",
        examples=[125.4],
        ge=0.0,
        le=2000.0,
    )
    power_W: float = Field(
        ...,
        description="Active real power in Watts (valid range: 0.0 to 1,000,000.0 W)",
        examples=[85200.0],
        ge=0.0,
        le=1_000_000.0,
    )
    energy: Optional[float] = Field(
        default=None,
        description="Cumulative electrical energy in kWh (optional / nullable)",
        examples=[1452.8],
        ge=0.0,
    )
    frequency_Hz: float = Field(
        ...,
        description="Grid AC frequency in Hertz (valid operational range: 40.0 to 65.0 Hz)",
        examples=[50.02],
        ge=40.0,
        le=65.0,
    )
    power_factor: float = Field(
        ...,
        description="Power factor (cos phi, valid range: -1.0 to 1.0)",
        examples=[0.92],
        ge=-1.0,
        le=1.0,
    )
    oil_temperature_C: float = Field(
        ...,
        description="Transformer top oil temperature in °C via MAX6675 (valid range: -20.0 to 200.0 °C)",
        examples=[58.4],
        ge=-20.0,
        le=200.0,
    )
    ambient_temperature_C: Optional[float] = Field(
        default=None,
        description="Outdoor ambient temperature in °C (optional / nullable; never substituted with oil temp)",
        examples=[32.1],
        ge=-30.0,
        le=70.0,
    )

    @field_validator("transformer_id")
    @classmethod
    def normalize_transformer_id(cls, v: str) -> str:
        clean = v.strip().upper()
        if not clean:
            raise ValueError("transformer_id cannot be empty")
        return clean


class TelemetryIngestResponse(BaseModel):
    """Structured response returned upon successful raw telemetry ingestion."""
    status: str = Field(default="success", description="Ingestion status indicator")
    message: str = Field(..., description="Human-readable status summary")
    record_id: int = Field(..., description="Unique auto-incremented database record ID")
    transformer_id: str = Field(..., description="Normalized transformer identifier")
    timestamp: datetime = Field(..., description="Original measurement timestamp recorded")
    ambient_temperature_supplied: bool = Field(
        ...,
        description="Explicit indicator whether physical ambient temperature was supplied or remains null"
    )


class HealthResponse(BaseModel):
    """API health and operational readiness response."""
    status: str = Field(default="healthy", description="Overall service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    database_connected: bool = Field(..., description="Database connectivity status")
    timestamp: datetime = Field(..., description="Current server UTC timestamp")
    total_raw_readings: int = Field(..., description="Total count of raw sensor records stored")


class HourlyTelemetryResponse(BaseModel):
    """Schema for 1-hour aggregated transformer telemetry."""
    id: int = Field(..., description="Hourly record database ID")
    transformer_id: str = Field(..., description="Transformer identifier")
    hour_start: datetime = Field(..., description="Start of 1-hour window (inclusive)")
    hour_end: datetime = Field(..., description="End of 1-hour window (exclusive)")
    voltage_V: float = Field(..., description="Hourly average voltage in Volts")
    current_A: float = Field(..., description="Hourly average current in Amperes")
    power_W: float = Field(..., description="Hourly average active power in Watts")
    frequency_Hz: float = Field(..., description="Hourly average frequency in Hz")
    power_factor: float = Field(..., description="Hourly average power factor")
    oil_temperature_C: float = Field(..., description="Hourly average top oil temperature in °C")
    ambient_temperature_C: Optional[float] = Field(
        default=None,
        description="Hourly average outdoor ambient temperature in °C (strictly null if no ambient readings were provided)"
    )
    reading_count: int = Field(..., description="Number of raw 2-second readings aggregated in this hour")
    aggregation_status: str = Field(default="completed", description="Aggregation status flag")
    created_at: datetime = Field(..., description="Record creation UTC timestamp")

    model_config = {"from_attributes": True}


class AggregationRunRequest(BaseModel):
    """Optional parameters for triggering manual hourly aggregation."""
    transformer_id: Optional[str] = Field(
        default=None,
        description="Optional filter to aggregate only a specific transformer (e.g. 'T1')",
        examples=["T1"],
    )
    reference_time: Optional[datetime] = Field(
        default=None,
        description="Optional reference timestamp for completed hour cutoff (defaults to server UTC now)",
        examples=["2026-11-03T01:00:00"],
    )


class AggregationRunResponse(BaseModel):
    """Summary response returned from triggering hourly aggregation."""
    status: str = Field(default="success", description="Execution status")
    records_created: int = Field(..., description="Number of new hourly records created")
    records_updated: int = Field(..., description="Number of existing hourly records updated")
    total_processed: int = Field(..., description="Total hourly records created or updated")
    cutoff_time: str = Field(..., description="Cutoff timestamp strictly before which hours were processed")
    message: str = Field(..., description="Human-readable result summary")


class HourlyHistoryResponse(BaseModel):
    """Paginated collection of hourly aggregated records for a transformer."""
    transformer_id: str = Field(..., description="Transformer identifier")
    total_records: int = Field(..., description="Total matching hourly records available")
    limit: int = Field(..., description="Page size limit applied")
    offset: int = Field(..., description="Page offset applied")
    records: List[HourlyTelemetryResponse] = Field(..., description="Chronological hourly aggregated records")

