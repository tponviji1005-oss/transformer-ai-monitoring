"""
FastAPI Application Entrypoint for Transformer Telemetry Ingestion.
Exposes:
- GET  /api/health: Service status, database connectivity, and record counts.
- POST /api/telemetry/ingest: Validates and persists raw ESP32 telemetry packets.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import os
import secrets
from typing import AsyncGenerator, Optional
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.backend.config import settings
from src.backend.database import HourlyTelemetry, LiveFeatures, PredictionRecord, RawTelemetry, get_db, init_db
from src.backend.schemas import (
    AggregationRunRequest,
    AggregationRunResponse,
    FeatureGenerateRequest,
    FeatureReadinessResponse,
    FeatureVectorResponse,
    HealthResponse,
    HourlyHistoryResponse,
    HourlyTelemetryResponse,
    PredictionHistoryResponse,
    PredictionRequest,
    PredictionResponse,
    TelemetryBatchIngestRequest,
    TelemetryBatchIngestResponse,
    TelemetryIngestRequest,
    TelemetryIngestResponse,
    TransformerSpecUpdateRequest,
)
from src.backend.aggregation_service import (
    aggregate_hourly_telemetry,
    get_hourly_history,
)
from src.backend.feature_service import (
    generate_live_features,
    get_feature_readiness,
    get_latest_features,
    get_transformer_spec,
    set_transformer_spec,
)
from src.backend.inference_service import (
    execute_live_inference,
    get_latest_prediction,
    get_prediction_history,
)
from src.backend.scheduler import background_scheduler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("transformer_backend")

# Initialize database tables on module import
init_db()


def verify_device_api_key(
    x_device_api_key: Optional[str] = Header(None, alias="X-Device-API-Key"),
) -> str:
    """
    Verifies that incoming requests contain a valid X-Device-API-Key header.
    Rejects unauthorized access with HTTP 401.
    Uses constant-time comparison via secrets.compare_digest.
    Fails safely if DEVICE_API_KEY is missing or empty in configuration.
    """
    configured_key = (settings.DEVICE_API_KEY or "").strip()
    if not configured_key:
        logger.error("Authentication rejected: DEVICE_API_KEY is missing or empty on the server.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed: server has no valid DEVICE_API_KEY configured.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not x_device_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Device-API-Key header: Missing required X-Device-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not secrets.compare_digest(x_device_api_key.encode("utf-8"), configured_key.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid X-Device-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return x_device_api_key


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context: validates configuration on startup."""
    # Startup validation: enforce security rules in production
    settings.validate_security_configuration()
    if settings.SCHEDULER_ENABLED:
        background_scheduler.start()
    yield
    # Shutdown handling
    if background_scheduler._running:
        background_scheduler.stop()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="REST API for remote ESP32 transformer telemetry ingestion and monitoring.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
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
        "dashboard": "/dashboard",
        "health": f"{settings.API_PREFIX}/health",
        "ingest": f"{settings.API_PREFIX}/telemetry/ingest",
        "ingest_batch": f"{settings.API_PREFIX}/telemetry/ingest-batch",
        "aggregation_run": f"{settings.API_PREFIX}/aggregation/run",
        "aggregation_history": f"{settings.API_PREFIX}/aggregation/{{transformer_id}}/history",
        "feature_generate": f"{settings.API_PREFIX}/features/{{transformer_id}}/generate",
        "feature_latest": f"{settings.API_PREFIX}/features/{{transformer_id}}/latest",
        "feature_readiness": f"{settings.API_PREFIX}/features/{{transformer_id}}/readiness",
        "predictions_predict": f"{settings.API_PREFIX}/predictions/{{transformer_id}}/predict",
        "predictions_latest": f"{settings.API_PREFIX}/predictions/{{transformer_id}}/latest",
        "predictions_history": f"{settings.API_PREFIX}/predictions/{{transformer_id}}/history",
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
    response: Response = Response(),
    db: Session = Depends(get_db),
    api_key: Optional[str] = Depends(verify_device_api_key),
) -> TelemetryIngestResponse:
    """
    Ingests, validates, and stores a raw 2-second telemetry reading from the ESP32.
    
    Data Handling Policy:
    - Verifies device API key via X-Device-API-Key header.
    - Idempotent deduplication: if reading_id is already present, returns HTTP 200 without inserting duplicate.
    - Stores exact physical measurements with timestamp.
    - If ambient_temperature_C is null/missing, it is strictly stored as NULL.
    - Ambient temperature is never substituted with oil temperature.
    - Does not aggregate readings or calculate load percentage at this stage.
    """
    try:
        # Atomic deduplication check
        if payload.reading_id:
            existing = (
                db.query(RawTelemetry)
                .filter(RawTelemetry.reading_id == payload.reading_id)
                .first()
            )
            if existing is not None:
                if response is not None:
                    response.status_code = status.HTTP_200_OK
                return TelemetryIngestResponse(
                    status="duplicate",
                    message=f"Duplicate reading {payload.reading_id} already recorded.",
                    record_id=existing.id,
                    transformer_id=existing.transformer_id,
                    timestamp=existing.timestamp,
                    reading_id=existing.reading_id,
                    ambient_temperature_supplied=existing.ambient_temperature_C is not None,
                )

        record = RawTelemetry(
            reading_id=payload.reading_id,
            sequence_number=payload.sequence_number,
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
            reading_id=record.reading_id,
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
    f"{settings.API_PREFIX}/telemetry/ingest-batch",
    response_model=TelemetryBatchIngestResponse,
    status_code=status.HTTP_200_OK,
    tags=["Telemetry"],
    summary="Batch Ingest ESP32 Sensor Observations (Buffer Flush)",
)
def ingest_batch_telemetry(
    payload: TelemetryBatchIngestRequest,
    db: Session = Depends(get_db),
    api_key: Optional[str] = Depends(verify_device_api_key),
) -> TelemetryBatchIngestResponse:
    """
    Ingests a chronological batch of telemetry readings from the ESP32 ring buffer.
    Performs atomic per-reading deduplication, ensuring network reconnect flushes
    do not insert duplicate records.
    """
    inserted_ids = []
    duplicate_count = 0

    try:
        for item in payload.readings:
            if item.reading_id:
                existing = (
                    db.query(RawTelemetry)
                    .filter(RawTelemetry.reading_id == item.reading_id)
                    .first()
                )
                if existing is not None:
                    duplicate_count += 1
                    continue

            record = RawTelemetry(
                reading_id=item.reading_id,
                sequence_number=item.sequence_number,
                transformer_id=item.transformer_id,
                timestamp=item.timestamp,
                voltage_V=item.voltage_V,
                current_A=item.current_A,
                power_W=item.power_W,
                energy=item.energy,
                frequency_Hz=item.frequency_Hz,
                power_factor=item.power_factor,
                oil_temperature_C=item.oil_temperature_C,
                ambient_temperature_C=item.ambient_temperature_C,
            )
            db.add(record)
            db.flush()
            inserted_ids.append(record.id)

        db.commit()

        return TelemetryBatchIngestResponse(
            status="success",
            total_received=len(payload.readings),
            inserted_count=len(inserted_ids),
            duplicate_count=duplicate_count,
            record_ids=inserted_ids,
        )
    except Exception as exc:
        db.rollback()
        logger.error(f"Failed to persist batch telemetry: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch persistence failure: {str(exc)}",
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


@app.post(
    f"{settings.API_PREFIX}/features/{{transformer_id}}/generate",
    response_model=FeatureVectorResponse,
    status_code=status.HTTP_200_OK,
    tags=["Features"],
    summary="Generate 39-Feature Vector for an Observation Hour",
)
def generate_features_endpoint(
    transformer_id: str,
    payload: Optional[FeatureGenerateRequest] = None,
    db: Session = Depends(get_db),
) -> FeatureVectorResponse:
    """
    Assembles the exact 39-feature vector for the specified (or latest) completed hour.
    Enforces exact timestamp delta lookups, IST timezone localization, and 168h cold-start buffer.
    Flags missing ambient temperature and unverified nameplate capacity without substituting data.
    """
    try:
        hour_ts = payload.hour_timestamp if payload else None
        record = generate_live_features(db=db, transformer_id=transformer_id, target_hour=hour_ts)
        missing_list = json.loads(record.missing_features) if record.missing_features else []
        return FeatureVectorResponse(
            transformer_id=record.transformer_id,
            hour_timestamp=record.hour_timestamp,
            readiness_status=record.readiness_status,
            is_ready_for_inference=bool(record.is_ready_for_inference),
            status_reason=record.status_reason,
            missing_features=missing_list,
            features=record.to_feature_dict(),
            created_at=record.created_at,
        )
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(val_err))
    except Exception as exc:
        db.rollback()
        logger.error(f"Live feature generation failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Live feature generation error: {str(exc)}",
        )


@app.get(
    f"{settings.API_PREFIX}/features/{{transformer_id}}/latest",
    response_model=FeatureVectorResponse,
    status_code=status.HTTP_200_OK,
    tags=["Features"],
    summary="Get Latest 39-Feature Vector for a Transformer",
)
def get_latest_features_endpoint(
    transformer_id: str,
    db: Session = Depends(get_db),
) -> FeatureVectorResponse:
    """Retrieves the most recent generated 39-feature vector for a transformer."""
    record = get_latest_features(db=db, transformer_id=transformer_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No feature vector has been generated yet for transformer {transformer_id}.",
        )
    missing_list = json.loads(record.missing_features) if record.missing_features else []
    return FeatureVectorResponse(
        transformer_id=record.transformer_id,
        hour_timestamp=record.hour_timestamp,
        readiness_status=record.readiness_status,
        is_ready_for_inference=bool(record.is_ready_for_inference),
        status_reason=record.status_reason,
        missing_features=missing_list,
        features=record.to_feature_dict(),
        created_at=record.created_at,
    )


@app.get(
    f"{settings.API_PREFIX}/features/{{transformer_id}}/readiness",
    response_model=FeatureReadinessResponse,
    status_code=status.HTTP_200_OK,
    tags=["Features"],
    summary="Check Operational Feature Readiness for Inference",
)
def check_feature_readiness_endpoint(
    transformer_id: str,
    hour_timestamp: Optional[datetime] = Query(default=None, description="Specific hour to check"),
    db: Session = Depends(get_db),
) -> FeatureReadinessResponse:
    """Evaluates whether all 39 features are valid or details blockers preventing inference."""
    try:
        readiness = get_feature_readiness(db=db, transformer_id=transformer_id, target_hour=hour_timestamp)
        return FeatureReadinessResponse(**readiness)
    except Exception as exc:
        logger.error(f"Failed to check feature readiness: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature readiness check failed: {str(exc)}",
        )


@app.post(
    f"{settings.API_PREFIX}/transformers/{{transformer_id}}/specs",
    status_code=status.HTTP_200_OK,
    tags=["Configuration"],
    summary="Update Confirmed Transformer Nameplate Specifications",
)
def update_transformer_specs_endpoint(
    transformer_id: str,
    specs: TransformerSpecUpdateRequest,
) -> dict:
    """
    Registers confirmed physical nameplate capacity.
    Must ONLY be updated when hardware owner confirms verified transformer ratings.
    """
    set_transformer_spec(
        transformer_id=transformer_id,
        rated_kVA=specs.rated_kVA,
        phases=specs.phases,
        rated_pf=specs.rated_pf,
        verified=specs.verified,
    )
    return {
        "status": "success",
        "transformer_id": transformer_id.strip().upper(),
        "specs": get_transformer_spec(transformer_id),
        "message": f"Nameplate specifications for {transformer_id.strip().upper()} registered successfully.",
    }


# =====================================================================
# Milestone 5 Stage 4: Live ML Inference and Prediction Endpoints
# =====================================================================

@app.post(
    f"{settings.API_PREFIX}/predictions/{{transformer_id}}/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
    summary="Generate Next-Hour ML Forecast and Advisory Overload Alert",
)
def predict_next_hour_endpoint(
    transformer_id: str,
    payload: Optional[PredictionRequest] = None,
    db: Session = Depends(get_db),
) -> PredictionResponse:
    """
    Executes next-hour machine learning forecasting and overload/anomaly detection:
    - Enforces Stage 3 feature readiness check (is_ready_for_inference == 1).
    - If prerequisites fail, marks status UNAVAILABLE with diagnostic reason. Zero synthetic data.
    - If ready, predicts next-hour active power (W) and load percentage (%).
    - Runs unit-specific detection:
      * T1: Isolation Forest anomaly risk score (flags out-of-distribution patterns as LOAD_ANOMALY).
      * T2: Random Forest overload classifier probability (selected threshold 0.3382).
      * T3: ThresholdMonitor evaluation.
    - Synthesizes standardized, purely advisory operator alert via UnifiedAlertEngine.
    - Idempotently stores result in `predictions` SQLite table.
    """
    try:
        hour_ts = payload.hour_timestamp if payload else None
        record = execute_live_inference(
            db=db,
            transformer_id=transformer_id,
            hour_timestamp=hour_ts,
        )
        return PredictionResponse(
            transformer_id=record.transformer_id,
            feature_hour_timestamp=record.feature_hour_timestamp,
            target_hour_timestamp=record.target_hour_timestamp,
            status=record.status,
            status_reason=record.status_reason,
            predicted_power_W=record.predicted_power_W,
            predicted_load_pct=record.predicted_load_pct,
            overload_probability=record.overload_probability,
            is_overload_predicted=bool(record.is_overload_predicted) if record.is_overload_predicted is not None else None,
            anomaly_score=record.anomaly_score,
            is_anomaly_detected=bool(record.is_anomaly_detected) if record.is_anomaly_detected is not None else None,
            alert_level=record.alert_level,
            alert_type=record.alert_type,
            alert_reason=record.alert_reason,
            recommended_action=record.recommended_action,
            created_at=record.created_at,
        )
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(val_err))
    except Exception as exc:
        db.rollback()
        logger.error(f"Live prediction failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Live prediction error: {str(exc)}",
        )


@app.get(
    f"{settings.API_PREFIX}/predictions/{{transformer_id}}/latest",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
    summary="Get Latest Next-Hour ML Forecast for a Transformer",
)
def get_latest_prediction_endpoint(
    transformer_id: str,
    db: Session = Depends(get_db),
) -> PredictionResponse:
    """Retrieves the most recent prediction and advisory alert for a transformer."""
    record = get_latest_prediction(db=db, transformer_id=transformer_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No prediction record has been generated yet for transformer {transformer_id}.",
        )
    return PredictionResponse(
        transformer_id=record.transformer_id,
        feature_hour_timestamp=record.feature_hour_timestamp,
        target_hour_timestamp=record.target_hour_timestamp,
        status=record.status,
        status_reason=record.status_reason,
        predicted_power_W=record.predicted_power_W,
        predicted_load_pct=record.predicted_load_pct,
        overload_probability=record.overload_probability,
        is_overload_predicted=bool(record.is_overload_predicted) if record.is_overload_predicted is not None else None,
        anomaly_score=record.anomaly_score,
        is_anomaly_detected=bool(record.is_anomaly_detected) if record.is_anomaly_detected is not None else None,
        alert_level=record.alert_level,
        alert_type=record.alert_type,
        alert_reason=record.alert_reason,
        recommended_action=record.recommended_action,
        created_at=record.created_at,
    )


@app.get(
    f"{settings.API_PREFIX}/predictions/{{transformer_id}}/history",
    response_model=PredictionHistoryResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
    summary="Get Chronological ML Prediction History",
)
def get_prediction_history_endpoint(
    transformer_id: str,
    start_time: Optional[datetime] = Query(default=None, description="Filter predictions from this observation hour"),
    end_time: Optional[datetime] = Query(default=None, description="Filter predictions up to this observation hour"),
    limit: int = Query(default=50, ge=1, le=500, description="Page size limit"),
    offset: int = Query(default=0, ge=0, description="Page offset"),
    db: Session = Depends(get_db),
) -> PredictionHistoryResponse:
    """Returns paginated historical predictions for a transformer."""
    clean_tid = transformer_id.strip().upper()
    total_count, records = get_prediction_history(
        db=db,
        transformer_id=clean_tid,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        offset=offset,
    )
    items = [
        PredictionResponse(
            transformer_id=r.transformer_id,
            feature_hour_timestamp=r.feature_hour_timestamp,
            target_hour_timestamp=r.target_hour_timestamp,
            status=r.status,
            status_reason=r.status_reason,
            predicted_power_W=r.predicted_power_W,
            predicted_load_pct=r.predicted_load_pct,
            overload_probability=r.overload_probability,
            is_overload_predicted=bool(r.is_overload_predicted) if r.is_overload_predicted is not None else None,
            anomaly_score=r.anomaly_score,
            is_anomaly_detected=bool(r.is_anomaly_detected) if r.is_anomaly_detected is not None else None,
            alert_level=r.alert_level,
            alert_type=r.alert_type,
            alert_reason=r.alert_reason,
            recommended_action=r.recommended_action,
            created_at=r.created_at,
        )
        for r in records
    ]
    return PredictionHistoryResponse(
        transformer_id=clean_tid,
        total_records=total_count,
        limit=limit,
        offset=offset,
        records=items,
    )


# Mount compiled React dashboard if dist directory exists
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist"))
if os.path.exists(frontend_dist):
    import os
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    assets_dir = os.path.join(frontend_dist, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/dashboard", tags=["Frontend"])
    def serve_dashboard():
        """Serves the React Monitoring & Predictive Maintenance Dashboard."""
        return FileResponse(os.path.join(frontend_dist, "index.html"))

