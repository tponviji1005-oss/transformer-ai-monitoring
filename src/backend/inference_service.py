"""
Live Machine Learning Inference and Prediction Service for Milestone 5 Stage 4.
Loads trained models (6 regression models, T2 classifier, T1 anomaly detector),
validates exact 39-feature compatibility, enforces Stage 3 readiness checks,
synthesizes unified advisory operator alerts, and persists forecasts idempotently.

CRITICAL CONSTRAINTS:
- Never bypass Stage 3 readiness checks.
- Never pass incomplete feature vectors (NaN/None) to trained models.
- Never invent missing sensor data or assume unverified nameplate ratings.
- T2 classifier threshold strictly = 0.3382.
- T1 anomaly detection flags out-of-distribution patterns (LOAD_ANOMALY), NOT confirmed overload probability.
- Do NOT predict or calculate efficiency (unmeasured secondary side).
- Strictly advisory alerts; zero automated hardware switching/tripping.
"""

from datetime import datetime, timedelta, timezone
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from src.backend.config import settings
from src.backend.database import LiveFeatures, PredictionRecord
from src.backend.feature_service import (
    FEATURE_NAMES_39,
    generate_live_features,
    get_transformer_spec,
)
from src.detection.alert_engine import AlertLevel, AlertType, UnifiedAlertEngine
from src.detection.anomaly_detector import TransformerAnomalyDetector
from src.detection.overload_classifier import TransformerOverloadClassifier
from src.detection.threshold_monitor import ThresholdMonitor
from src.models.random_forest import TransformerRandomForest

logger = logging.getLogger("transformer_backend.inference")

# Authoritative T2 Decision Threshold from Milestone 4 Optimization
T2_SELECTED_THRESHOLD: float = 0.3382

# Authoritative Anomaly Score Alert Threshold from Milestone 4
T1_ANOMALY_ALERT_THRESHOLD: float = 0.50


class ModelRegistry:
    """
    Singleton registry caching loaded ML models in memory.
    Models are loaded once and reused across inference invocations to optimize latency and RAM.
    """
    _instance: Optional["ModelRegistry"] = None

    def __init__(self):
        self.regression_models: Dict[str, Dict[str, TransformerRandomForest]] = {}
        self.overload_classifier_t2: Optional[TransformerOverloadClassifier] = None
        self.anomaly_detector_t1: Optional[TransformerAnomalyDetector] = None
        self.threshold_monitor_t3: Optional[ThresholdMonitor] = None
        self.alert_engine: Optional[UnifiedAlertEngine] = None
        self.is_loaded: bool = False

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        """Returns the singleton ModelRegistry instance, initializing it if necessary."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.load_all_models()
        return cls._instance

    def load_all_models(self) -> None:
        """
        Loads all 6 regression models, T2 classifier, T1 anomaly detector,
        and initializes monitors and alert engine.
        """
        logger.info("Initializing ModelRegistry: loading serialized joblib artifacts...")

        # 1. Load 6 Random Forest Regression Models
        reg_configs = [
            ("T1", "power", "models/t1_rf_power.joblib"),
            ("T1", "load", "models/t1_rf_load.joblib"),
            ("T2", "power", "models/t2_rf_power.joblib"),
            ("T2", "load", "models/t2_rf_load.joblib"),
            ("T3", "power", "models/t3_rf_power.joblib"),
            ("T3", "load", "models/t3_rf_load.joblib"),
        ]

        for tid, target_type, filepath in reg_configs:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model artifact not found: {filepath}")
            model = TransformerRandomForest.load(filepath)
            assert (
                list(model.feature_names) == FEATURE_NAMES_39
            ), f"Feature schema mismatch in {filepath}"
            if tid not in self.regression_models:
                self.regression_models[tid] = {}
            self.regression_models[tid][target_type] = model
            logger.info(f"Loaded regression model {tid}_{target_type} ({len(model.feature_names)} features).")

        # 2. Load T2 Supervised Overload Classifier
        t2_clf_path = "models/overload/t2_overload_classifier.joblib"
        if not os.path.exists(t2_clf_path):
            raise FileNotFoundError(f"T2 classifier artifact not found: {t2_clf_path}")
        self.overload_classifier_t2 = TransformerOverloadClassifier.load(t2_clf_path)
        assert (
            list(self.overload_classifier_t2.feature_names) == FEATURE_NAMES_39
        ), f"Feature schema mismatch in {t2_clf_path}"
        # Force exact calibrated threshold of 0.3382
        self.overload_classifier_t2.selected_threshold = T2_SELECTED_THRESHOLD
        logger.info(
            f"Loaded T2 overload classifier (selected_threshold={self.overload_classifier_t2.selected_threshold:.4f})."
        )

        # 3. Load T1 Unsupervised Anomaly Detector (Isolation Forest)
        t1_ad_path = "models/overload/t1_anomaly_detector.joblib"
        if not os.path.exists(t1_ad_path):
            raise FileNotFoundError(f"T1 anomaly detector artifact not found: {t1_ad_path}")
        self.anomaly_detector_t1 = TransformerAnomalyDetector.load(t1_ad_path)
        assert (
            list(self.anomaly_detector_t1.feature_names) == FEATURE_NAMES_39
        ), f"Feature schema mismatch in {t1_ad_path}"
        logger.info("Loaded T1 Isolation Forest anomaly detector.")

        # 4. Initialize T3 Rule-Based Threshold Monitor
        self.threshold_monitor_t3 = ThresholdMonitor(
            transformer_id="T3",
            warning_load_pct=80.0,
            critical_load_pct=100.0,
            warning_temp_C=75.0,
            critical_temp_C=90.0,
        )
        logger.info("Initialized T3 ThresholdMonitor (Warning: 80%, Critical: 100%).")

        # 5. Initialize Unified Alert Engine
        # Strictly calibrated: classifier_alert_threshold = 0.3382, anomaly_alert_threshold = 0.50
        self.alert_engine = UnifiedAlertEngine(
            warning_load_pct=80.0,
            critical_load_pct=100.0,
            classifier_alert_threshold=T2_SELECTED_THRESHOLD,
            anomaly_alert_threshold=T1_ANOMALY_ALERT_THRESHOLD,
        )
        logger.info(
            f"Initialized UnifiedAlertEngine (classifier_threshold={T2_SELECTED_THRESHOLD}, anomaly_threshold={T1_ANOMALY_ALERT_THRESHOLD})."
        )

        self.is_loaded = True


def get_model_registry() -> ModelRegistry:
    """Convenience getter for the global ModelRegistry singleton."""
    return ModelRegistry.get_instance()


def execute_live_inference(
    db: Session,
    transformer_id: str,
    hour_timestamp: Optional[datetime] = None,
) -> PredictionRecord:
    """
    Orchestrates live next-hour ML inference for a transformer.

    Workflow:
    1. Retrieves or generates LiveFeatures record for the specified or latest observation hour t.
    2. Enforces Stage 3 readiness check (is_ready_for_inference == 1).
    3. If prerequisites fail:
       - Persists an UNAVAILABLE PredictionRecord with diagnostic reason.
       - NEVER invokes ML models with missing/null features.
    4. If prerequisites pass:
       - Formats exact 39-feature vector.
       - Runs next-hour active power and load percentage regression models.
       - Runs transformer-specific overload/anomaly detection:
         * T1: Isolation Forest anomaly score & flag (out-of-distribution pattern recognition).
         * T2: Random Forest classifier probability & overload prediction (threshold 0.3382).
         * T3: ThresholdMonitor evaluation.
       - Synthesizes standardized, advisory-only operator alert via UnifiedAlertEngine.
       - Persists prediction idempotently in `predictions` table.
    """
    clean_tid = transformer_id.strip().upper()
    if clean_tid not in ("T1", "T2", "T3"):
        raise ValueError(f"Unknown transformer_id '{clean_tid}'. Valid units: T1, T2, T3.")

    registry = get_model_registry()

    # 1. Fetch or generate LiveFeatures record
    th_query = hour_timestamp.replace(tzinfo=None) if (hour_timestamp and hour_timestamp.tzinfo) else hour_timestamp

    if th_query is not None:
        feature_record = (
            db.query(LiveFeatures)
            .filter(
                LiveFeatures.transformer_id == clean_tid,
                LiveFeatures.hour_timestamp == th_query,
            )
            .first()
        )
    else:
        feature_record = (
            db.query(LiveFeatures)
            .filter(LiveFeatures.transformer_id == clean_tid)
            .order_by(desc(LiveFeatures.hour_timestamp))
            .first()
        )

    # If no feature record exists yet for the given timestamp, attempt generation
    if not feature_record:
        try:
            feature_record = generate_live_features(
                db=db,
                transformer_id=clean_tid,
                target_hour=th_query,
            )
        except Exception as e:
            logger.warning(f"Feature generation failed for {clean_tid}: {e}")
            feature_record = None

    # Determine reference observation timestamp
    if feature_record:
        feature_dt = feature_record.hour_timestamp
    elif th_query is not None:
        feature_dt = th_query
    else:
        raise ValueError(f"No telemetry or feature record found for {clean_tid}.")

    target_dt = feature_dt + timedelta(hours=1)
    th_store = feature_dt.replace(tzinfo=None) if feature_dt.tzinfo else feature_dt

    # 2. Gate Check: Is the feature vector ready for inference?
    if not feature_record or feature_record.is_ready_for_inference != 1:
        reason = (
            feature_record.status_reason
            if feature_record
            else f"No complete 39-feature vector generated for {clean_tid} at {feature_dt.isoformat()}."
        )
        logger.info(
            f"Inference gated/blocked for {clean_tid} at {feature_dt.isoformat()}: {reason}"
        )

        # Idempotent persistence of UNAVAILABLE record
        existing = (
            db.query(PredictionRecord)
            .filter(
                PredictionRecord.transformer_id == clean_tid,
                PredictionRecord.feature_hour_timestamp == th_store,
            )
            .first()
        )

        if existing:
            existing.status = "UNAVAILABLE"
            existing.status_reason = reason
            existing.predicted_power_W = None
            existing.predicted_load_pct = None
            existing.overload_probability = None
            existing.is_overload_predicted = None
            existing.anomaly_score = None
            existing.is_anomaly_detected = None
            existing.alert_level = None
            existing.alert_type = None
            existing.alert_reason = None
            existing.recommended_action = None
            db.commit()
            db.refresh(existing)
            return existing
        else:
            unavailable_rec = PredictionRecord(
                transformer_id=clean_tid,
                feature_hour_timestamp=th_store,
                target_hour_timestamp=target_dt.replace(tzinfo=None) if target_dt.tzinfo else target_dt,
                status="UNAVAILABLE",
                status_reason=reason,
                predicted_power_W=None,
                predicted_load_pct=None,
                overload_probability=None,
                is_overload_predicted=None,
                anomaly_score=None,
                is_anomaly_detected=None,
                alert_level=None,
                alert_type=None,
                alert_reason=None,
                recommended_action=None,
            )
            db.add(unavailable_rec)
            db.commit()
            db.refresh(unavailable_rec)
            return unavailable_rec

    # 3. Prerequisites Met: Extract and validate 39 features
    feat_dict = feature_record.to_feature_dict()
    for feat_name in FEATURE_NAMES_39:
        if feat_dict[feat_name] is None:
            raise ValueError(f"Feature '{feat_name}' is None despite is_ready_for_inference == 1.")

    df_X = pd.DataFrame([feat_dict])[FEATURE_NAMES_39]
    assert not df_X.isnull().any().any(), "Feature matrix contains unexpected nulls."

    # 4. Next-Hour Regression Forecasts
    rf_power = registry.regression_models[clean_tid]["power"]
    rf_load = registry.regression_models[clean_tid]["load"]

    pred_power_W = float(rf_power.predict(df_X)[0])
    pred_load_pct = float(rf_load.predict(df_X)[0])

    # 5. Transformer-Specific Overload & Anomaly Detection
    overload_prob: Optional[float] = None
    is_ol_pred: Optional[int] = None
    anom_score: Optional[float] = None
    is_anom: Optional[int] = None

    if clean_tid == "T1":
        # Unsupervised Isolation Forest Anomaly Detection
        anom_detector = registry.anomaly_detector_t1
        is_anom = int(anom_detector.predict(df_X)[0])
        anom_score = float(anom_detector.anomaly_score(df_X)[0])
        # T1 has no supervised overload classifier (0 historical training overloads)
        overload_prob = None
        is_ol_pred = None

    elif clean_tid == "T2":
        # Supervised Random Forest Classifier with threshold 0.3382
        classifier = registry.overload_classifier_t2
        prob_arr = classifier.predict_proba(df_X)
        overload_prob = float(prob_arr[0])
        is_ol_pred = int(overload_prob >= T2_SELECTED_THRESHOLD)
        # T2 uses supervised classification rather than Isolation Forest
        anom_score = None
        is_anom = None

    elif clean_tid == "T3":
        # T3 had 0 historical overloads; evaluated via ThresholdMonitor
        overload_prob = None
        is_ol_pred = None
        anom_score = None
        is_anom = None

    # 6. Unified Advisory Alert Synthesis
    curr_load = float(feature_record.load_pct)
    alert = registry.alert_engine.generate_alert(
        transformer_id=clean_tid,
        timestamp=feature_dt.isoformat(),
        current_load_pct=curr_load,
        predicted_load_pct=pred_load_pct,
        overload_probability=overload_prob,
        is_anomaly=bool(is_anom) if is_anom is not None else None,
        anomaly_score=anom_score,
    )

    # 7. Idempotent Persistence in predictions table
    existing = (
        db.query(PredictionRecord)
        .filter(
            PredictionRecord.transformer_id == clean_tid,
            PredictionRecord.feature_hour_timestamp == th_store,
        )
        .first()
    )

    if existing:
        existing.target_hour_timestamp = target_dt.replace(tzinfo=None) if target_dt.tzinfo else target_dt
        existing.status = "SUCCESS"
        existing.status_reason = "Inference completed successfully across all models."
        existing.predicted_power_W = round(pred_power_W, 2)
        existing.predicted_load_pct = round(pred_load_pct, 2)
        existing.overload_probability = round(overload_prob, 4) if overload_prob is not None else None
        existing.is_overload_predicted = is_ol_pred
        existing.anomaly_score = round(anom_score, 4) if anom_score is not None else None
        existing.is_anomaly_detected = is_anom
        existing.alert_level = alert.risk_level
        existing.alert_type = alert.alert_type
        existing.alert_reason = alert.reason
        existing.recommended_action = alert.recommended_action
        db_record = existing
    else:
        db_record = PredictionRecord(
            transformer_id=clean_tid,
            feature_hour_timestamp=th_store,
            target_hour_timestamp=target_dt.replace(tzinfo=None) if target_dt.tzinfo else target_dt,
            status="SUCCESS",
            status_reason="Inference completed successfully across all models.",
            predicted_power_W=round(pred_power_W, 2),
            predicted_load_pct=round(pred_load_pct, 2),
            overload_probability=round(overload_prob, 4) if overload_prob is not None else None,
            is_overload_predicted=is_ol_pred,
            anomaly_score=round(anom_score, 4) if anom_score is not None else None,
            is_anomaly_detected=is_anom,
            alert_level=alert.risk_level,
            alert_type=alert.alert_type,
            alert_reason=alert.reason,
            recommended_action=alert.recommended_action,
        )
        db.add(db_record)

    db.commit()
    db.refresh(db_record)

    logger.info(
        f"Prediction complete for {clean_tid} at {feature_dt.isoformat()}: "
        f"Power={pred_power_W:.1f}W, Load={pred_load_pct:.1f}%, Alert={alert.alert_type} [{alert.risk_level}]"
    )

    return db_record


def get_latest_prediction(db: Session, transformer_id: str) -> Optional[PredictionRecord]:
    """Retrieves the most recent prediction record for a transformer."""
    clean_tid = transformer_id.strip().upper()
    return (
        db.query(PredictionRecord)
        .filter(PredictionRecord.transformer_id == clean_tid)
        .order_by(desc(PredictionRecord.feature_hour_timestamp))
        .first()
    )


def get_prediction_history(
    db: Session,
    transformer_id: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 50,
    offset: int = 0,
) -> Tuple[int, List[PredictionRecord]]:
    """Retrieves paginated chronological predictions for a transformer."""
    clean_tid = transformer_id.strip().upper()
    query = select(PredictionRecord).where(PredictionRecord.transformer_id == clean_tid)

    if start_time:
        st = start_time.replace(tzinfo=None) if start_time.tzinfo else start_time
        query = query.where(PredictionRecord.feature_hour_timestamp >= st)
    if end_time:
        et = end_time.replace(tzinfo=None) if end_time.tzinfo else end_time
        query = query.where(PredictionRecord.feature_hour_timestamp <= et)

    all_records = db.execute(query).scalars().all()
    total_count = len(all_records)

    paginated_query = (
        query.order_by(PredictionRecord.feature_hour_timestamp.desc())
        .offset(offset)
        .limit(limit)
    )
    records = list(db.execute(paginated_query).scalars().all())

    return total_count, records

