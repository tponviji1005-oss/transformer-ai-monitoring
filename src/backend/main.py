"""
FastAPI Application Entrypoint for Transformer Telemetry Ingestion.
Exposes:
- GET  /api/health: Service status, database connectivity, and record counts.
- POST /api/telemetry/ingest: Validates and persists raw ESP32 telemetry packets.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from typing import AsyncGenerator, Optional
from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.backend.config import settings
from src.backend.database import HourlyTelemetry, RawTelemetry, get_db, init_db
from src.backend.schemas import (
    AggregationRunRequest,
    AggregationRunResponse,
    HealthResponse,
    HourlyHistoryResponse,
    HourlyTelemetryResponse,
    TelemetryIngestRequest,
    TelemetryIngestResponse,
)
from src.backend.aggregation_service import (
    aggregate_hourly_telemetry,
    get_hourly_history,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("transformer_backend")


# Initialize database tables on module import
init_db()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="REST API for remote ESP32 transformer telemetry ingestion and monitoring.",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS for local development, web dashboards, and ESP32 clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
def root_info() -> dict:
    """Root metadata endpoint."""
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": f"{settings.API_PREFIX}/health",
        "ingest": f"{settings.API_PREFIX}/telemetry/ingest",
        "aggregation_run": f"{settings.API_PREFIX}/aggregation/run",
        "aggregation_history": f"{settings.API_PREFIX}/aggregation/{{transformer_id}}/history",
    }


@app.get(
    f"{settings.API_PREFIX}/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["System"],
    summary="Health and Operational Check",
)
def get_health_status(db: Session = Depends(get_db)) -> HealthResponse:
    """
    Returns system operational health, verifies SQLite database connectivity,
    and reports total raw sensor records logged to date.
    """
    try:
        count = db.scalar(select(func.count(RawTelemetry.id))) or 0
        db_connected = True
    except Exception as exc:
        logger.error(f"Database health check failed: {exc}", exc_info=True)
        db_connected = False
        count = 0

    return HealthResponse(
        status="healthy" if db_connected else "degraded",
        service=settings.PROJECT_NAME,
        version=settings.VERSION,
        database_connected=db_connected,
        timestamp=datetime.now(timezone.utc),
        total_raw_readings=count,
    )


@app.post(
    f"{settings.API_PREFIX}/telemetry/ingest",
    response_model=TelemetryIngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Telemetry"],
    summary="Ingest Raw ESP32 Sensor Observation",
)
def ingest_raw_telemetry(
    payload: TelemetryIngestRequest,
    db: Session = Depends(get_db),
) -> TelemetryIngestResponse:
    """
    Ingests, validates, and stores a raw 2-second telemetry reading from the ESP32.
    
    Data Handling Policy:
    - Stores exact physical measurements with timestamp.
    - If ambient_temperature_C is null/missing, it is strictly stored as NULL.
    - Ambient temperature is never substituted with oil temperature.
    - Does not aggregate readings or calculate load percentage at this stage.
    """
    try:
        record = RawTelemetry(
            transformer_id=payload.transformer_id,
            timestamp=payload.timestamp,
            voltage_V=payload.voltage_V,
            current_A=payload.current_A,
            power_W=payload.power_W,
            energy=payload.energy,
            frequency_Hz=payload.frequency_Hz,
            power_factor=payload.power_factor,
            oil_temperature_C=payload.oil_temperature_C,
            ambient_temperature_C=payload.ambient_temperature_C,
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        ambient_supplied = payload.ambient_temperature_C is not None

        return TelemetryIngestResponse(
            status="success",
            message=f"Raw telemetry for {payload.transformer_id} persisted successfully.",
            record_id=record.id,
            transformer_id=record.transformer_id,
            timestamp=record.timestamp,
            ambient_temperature_supplied=ambient_supplied,
        )
    except Exception as exc:
        db.rollback()
        logger.error(f"Failed to persist telemetry payload: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database persistence failure: {str(exc)}",
        )


@app.post(
    f"{settings.API_PREFIX}/aggregation/run",
    response_model=AggregationRunResponse,
    status_code=status.HTTP_200_OK,
    tags=["Aggregation"],
    summary="Run Hourly Telemetry Aggregation",
)
def run_aggregation(
    payload: Optional[AggregationRunRequest] = None,
    db: Session = Depends(get_db),
) -> AggregationRunResponse:
    """
    Executes hourly aggregation for completed hours.
    Idempotent: running multiple times updates existing records without duplication.
    Incomplete current hour is strictly excluded.
    """
    try:
        transformer_id = payload.transformer_id if payload else None
        reference_time = payload.reference_time if payload else None
        result = aggregate_hourly_telemetry(
            db=db,
            transformer_id=transformer_id,
            reference_time=reference_time,
        )
        return AggregationRunResponse(**result)
    except Exception as exc:
        db.rollback()
        logger.error(f"Hourly aggregation execution failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Aggregation execution failure: {str(exc)}",
        )


@app.get(
    f"{settings.API_PREFIX}/aggregation/{{transformer_id}}/history",
    response_model=HourlyHistoryResponse,
    status_code=status.HTTP_200_OK,
    tags=["Aggregation"],
    summary="Get Hourly Aggregated History for a Transformer",
)
def get_aggregation_history(
    transformer_id: str,
    start_time: Optional[datetime] = Query(default=None, description="Filter records starting from this timestamp"),
    end_time: Optional[datetime] = Query(default=None, description="Filter records up to this timestamp"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum number of hourly records to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
) -> HourlyHistoryResponse:
    """
    Retrieves chronologically sorted hourly aggregated history for a given transformer.
    """
    try:
        total, records = get_hourly_history(
            db=db,
            transformer_id=transformer_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=offset,
        )
        return HourlyHistoryResponse(
            transformer_id=transformer_id.strip().upper(),
            total_records=total,
            limit=limit,
            offset=offset,
            records=[HourlyTelemetryResponse.model_validate(r) for r in records],
        )
    except Exception as exc:
        logger.error(f"Failed to fetch hourly history: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch aggregation history: {str(exc)}",
        )
