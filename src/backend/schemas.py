"""
Pydantic Schemas for Ingestion, Validation, and API Responses.
Strictly validates incoming sensor signals against electrical and physical boundaries.
Does not invent missing sensor values and keeps ambient temperature explicitly nullable.
"""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class TelemetryIngestRequest(BaseModel):
    """
    Validation schema for incoming ESP32 sensor observations.
    Accepts raw 2-second telemetry packets from remote microcontrollers.
    """
    reading_id: Optional[str] = Field(
        default=None,
        description="Unique ESP32 reading identifier (<transformer_id>-<boot_id>-<seq>)",
        examples=["T2-B94A7C10-00000042"],
        max_length=64,
    )
    sequence_number: Optional[int] = Field(
        default=None,
        description="Monotonically increasing sequence number per boot session",
        examples=[42],
        ge=1,
    )
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
    status: str = Field(default="success", description="Ingestion status indicator ('success' or 'duplicate')")
    message: str = Field(..., description="Human-readable status summary")
    record_id: Optional[int] = Field(default=None, description="Unique database record ID (None if duplicate)")
    transformer_id: str = Field(..., description="Normalized transformer identifier")
    timestamp: datetime = Field(..., description="Original measurement timestamp recorded")
    reading_id: Optional[str] = Field(default=None, description="ESP32 reading identifier recorded")
    ambient_temperature_supplied: bool = Field(
        ...,
        description="Explicit indicator whether physical ambient temperature was supplied or remains null"
    )


class TelemetryBatchIngestRequest(BaseModel):
    """Schema for micro-batch telemetry ingestion from ESP32 buffer flush."""
    readings: List[TelemetryIngestRequest] = Field(
        ...,
        description="List of telemetry readings to ingest in chronological order",
        min_length=1,
        max_length=120,
    )


class TelemetryBatchIngestResponse(BaseModel):
    """Structured response for batch telemetry ingestion."""
    status: str = Field(default="success", description="Overall batch ingestion status")
    total_received: int = Field(..., description="Total readings received in batch")
    inserted_count: int = Field(..., description="Number of newly inserted readings")
    duplicate_count: int = Field(..., description="Number of duplicate readings acknowledged")
    record_ids: List[int] = Field(default_factory=list, description="IDs of newly inserted records")


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


class FeatureGenerateRequest(BaseModel):
    """Optional parameters for generating live features."""
    hour_timestamp: Optional[datetime] = Field(
        default=None,
        description="Target completed hour timestamp. If None, uses latest available completed hour.",
        examples=["2026-11-03T18:00:00"],
    )


class FeatureVectorResponse(BaseModel):
    """Detailed response containing the complete 39-feature vector and readiness metadata."""
    transformer_id: str = Field(..., description="Transformer identifier")
    hour_timestamp: datetime = Field(..., description="Timestamp of the observation hour")
    readiness_status: str = Field(..., description="Operational readiness status")
    is_ready_for_inference: bool = Field(..., description="True if all 39 features are non-null and valid")
    status_reason: Optional[str] = Field(default=None, description="Diagnostics explaining why features are complete or blocked")
    missing_features: List[str] = Field(default_factory=list, description="List of feature names that are currently None/missing")
    features: Dict[str, Optional[float]] = Field(..., description="The authoritative 39-feature dictionary in exact model order")
    created_at: datetime = Field(..., description="Feature vector generation timestamp")


class FeatureReadinessResponse(BaseModel):
    """Diagnostic response evaluating readiness for ML inference."""
    transformer_id: str = Field(..., description="Transformer identifier")
    hour_timestamp: Optional[str] = Field(default=None, description="Target observation hour timestamp")
    readiness_status: str = Field(..., description="Current status code")
    is_ready_for_inference: bool = Field(..., description="True if inference is permitted")
    total_features_count: int = Field(default=39, description="Total features in the standard vector")
    valid_features_count: int = Field(..., description="Count of non-null valid features")
    missing_features_count: int = Field(..., description="Count of missing/null features")
    missing_features: List[str] = Field(..., description="List of missing feature names")
    status_reason: Optional[str] = Field(default=None, description="Explanatory failure reason or readiness confirmation")
    features: Optional[Dict[str, Optional[float]]] = Field(default=None, description="Feature dictionary if generated")


class TransformerSpecUpdateRequest(BaseModel):
    """Request model for confirming physical transformer nameplate ratings."""
    rated_kVA: float = Field(..., gt=0.0, description="Rated apparent power capacity in kVA", examples=[50.0])
    phases: int = Field(default=1, ge=1, le=3, description="Phase configuration (1 or 3)", examples=[1])
    rated_pf: float = Field(default=0.90, ge=0.1, le=1.0, description="Rated power factor", examples=[0.90])
    verified: bool = Field(default=True, description="Explicit confirmation of verified nameplate specs")


class PredictionRequest(BaseModel):
    """Optional parameters for triggering next-hour inference."""
    hour_timestamp: Optional[datetime] = Field(
        default=None,
        description="Target observation hour timestamp. If None, uses latest completed hour with generated features.",
        examples=["2026-11-03T18:00:00"],
    )


class PredictionResponse(BaseModel):
    """Structured response containing next-hour regression forecast and advisory alert."""
    transformer_id: str = Field(..., description="Transformer identifier")
    feature_hour_timestamp: datetime = Field(..., description="Timestamp of observation hour t")
    target_hour_timestamp: datetime = Field(..., description="Timestamp of forecast hour t+1h")
    status: str = Field(..., description="Inference status: SUCCESS or UNAVAILABLE")
    status_reason: Optional[str] = Field(default=None, description="Diagnostic reason if unavailable or operational note")

    # Forecasts (Hour t+1)
    predicted_power_W: Optional[float] = Field(default=None, description="Predicted active power for hour t+1 in Watts")
    predicted_load_pct: Optional[float] = Field(default=None, description="Predicted load percentage for hour t+1")

    # Overload / Anomaly Signals
    overload_probability: Optional[float] = Field(default=None, description="T2 only: predicted probability of overload in [0, 1]")
    is_overload_predicted: Optional[bool] = Field(default=None, description="T2 only: true if overload_probability >= 0.3382")
    anomaly_score: Optional[float] = Field(default=None, description="T1 only: anomaly risk index in [0, 1]")
    is_anomaly_detected: Optional[bool] = Field(default=None, description="T1 only: true if out-of-distribution anomaly detected")

    # Unified Advisory Alert
    alert_level: Optional[str] = Field(default=None, description="Advisory risk level: NORMAL, WARNING, CRITICAL")
    alert_type: Optional[str] = Field(default=None, description="Advisory alert category")
    alert_reason: Optional[str] = Field(default=None, description="Explanation of triggered alert")
    recommended_action: Optional[str] = Field(default=None, description="Advisory recommendations for substation operator review")

    created_at: datetime = Field(..., description="Prediction generation timestamp")


class PredictionHistoryResponse(BaseModel):
    """Paginated collection of prediction records for a transformer."""
    transformer_id: str = Field(..., description="Transformer identifier")
    total_records: int = Field(..., description="Total matching prediction records available")
    limit: int = Field(..., description="Page size limit applied")
    offset: int = Field(..., description="Page offset applied")
    records: List[PredictionResponse] = Field(..., description="Chronological prediction records")


