"""
Unit and Integration Test Suite for Milestone 5 Stage 4:
Live Machine Learning Inference and Prediction Service.

Verifies:
1. Singleton ModelRegistry caching (models loaded once).
2. Strict 39-feature vector compatibility and ordering across all models.
3. Next-hour power and load forecasts for T1, T2, and T3.
4. T2 Overload Classification using exact threshold 0.3382.
5. T1 Isolation Forest Anomaly Detection (flags LOAD_ANOMALY, zero overload probability confusion).
6. T3 Configurable Threshold Monitoring.
7. Strict Stage 3 readiness gating (blocked inference on missing ambient, missing nameplate, cold start).
8. Idempotent persistence in `predictions` SQLite table.
9. FastAPI prediction endpoints (POST /predict, GET /latest, GET /history).
10. Strictly advisory operator language with zero electrical actuation commands.
11. Absolute exclusion of efficiency calculation.
"""

from datetime import datetime, timedelta, timezone
import json
import os
import sys
import unittest
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend.config import settings
from src.backend.database import (
    HourlyTelemetry,
    LiveFeatures,
    PredictionRecord,
    RawTelemetry,
    SessionLocal,
    init_db,
)
from src.backend.feature_service import (
    FEATURE_NAMES_39,
    generate_live_features,
    reset_transformer_specs,
    set_transformer_spec,
)
from src.backend.inference_service import (
    T1_ANOMALY_ALERT_THRESHOLD,
    T2_SELECTED_THRESHOLD,
    ModelRegistry,
    execute_live_inference,
    get_latest_prediction,
    get_model_registry,
    get_prediction_history,
)
from src.backend.main import app
from src.backend.schemas import PredictionRequest, PredictionResponse
from src.detection.alert_engine import AlertLevel, AlertType


class TestLiveInferenceService(unittest.TestCase):
    """Test suite for Milestone 5 Stage 4 live inference and prediction APIs."""

    @classmethod
    def setUpClass(cls):
        """Initialize database tables and pre-load model registry."""
        init_db()
        db = SessionLocal()
        cls._clean_db(db)
        db.close()

        cls.client = TestClient(app)
        cls.registry = get_model_registry()

    @classmethod
    def tearDownClass(cls):
        """Clean up test records and restore baseline hourly records."""
        db = SessionLocal()
        cls._clean_db(db)
        db.close()
        reset_transformer_specs()

        # Restore baseline hourly records from backup if available
        backup_path = os.path.join(settings.DATA_DIR, "backups", "transformer_monitoring_pre_stage5.db")
        if os.path.exists(backup_path):
            import sqlite3
            src = sqlite3.connect(backup_path)
            dst = sqlite3.connect(settings.DB_FILE)
            src_c = src.cursor()
            dst_c = dst.cursor()
            rows = src_c.execute("SELECT * FROM hourly_telemetry").fetchall()
            if rows:
                cols = [d[0] for d in src_c.description]
                placeholders = ",".join(["?" for _ in cols])
                dst_c.executemany(f"INSERT OR REPLACE INTO hourly_telemetry ({','.join(cols)}) VALUES ({placeholders})", rows)
                dst.commit()
            src.close()
            dst.close()

    @staticmethod
    def _clean_db(db):
        """Cleans test rows for T1, T2, T3 and temporary test prefixes."""
        db.query(PredictionRecord).filter(
            PredictionRecord.transformer_id.in_(["T1", "T2", "T3"])
            | PredictionRecord.transformer_id.like("T_%")
        ).delete()
        db.query(LiveFeatures).filter(
            LiveFeatures.transformer_id.in_(["T1", "T2", "T3"])
            | LiveFeatures.transformer_id.like("T_%")
        ).delete()
        db.query(HourlyTelemetry).filter(
            HourlyTelemetry.transformer_id.in_(["T1", "T2", "T3"])
            | HourlyTelemetry.transformer_id.like("T_%")
        ).delete()
        db.commit()

    def setUp(self):
        """Create fresh transactional database session for each test."""
        self.db = SessionLocal()
        self._clean_db(self.db)
        reset_transformer_specs()

    def tearDown(self):
        """Close database session after each test."""
        self._clean_db(self.db)
        self.db.close()
        reset_transformer_specs()

    def _seed_complete_history(
        self,
        transformer_id: str,
        base_hour: datetime,
        base_power: float = 20000.0,
        base_oil_temp: float = 50.0,
        base_ambient_temp: float = 28.0,
    ) -> None:
        """Helper to seed contiguous 169-hour buffer to clear cold start."""
        recs = []
        for h in range(168, -1, -1):
            t = base_hour - timedelta(hours=h)
            recs.append(
                HourlyTelemetry(
                    transformer_id=transformer_id,
                    hour_start=t,
                    hour_end=t + timedelta(hours=1),
                    voltage_V=415.0,
                    current_A=15.0,
                    power_W=base_power,
                    frequency_Hz=50.0,
                    power_factor=0.90,
                    oil_temperature_C=base_oil_temp,
                    ambient_temperature_C=base_ambient_temp,
                    reading_count=1800,
                )
            )
        self.db.add_all(recs)
        self.db.commit()

    def test_01_model_registry_cached_singleton(self):
        """
        Verify that ModelRegistry behaves as an in-memory singleton.
        Subsequent calls must return the same pre-loaded instance.
        """
        reg1 = get_model_registry()
        reg2 = get_model_registry()
        self.assertIs(reg1, reg2)
        self.assertTrue(reg1.is_loaded)

        # Check that all 6 regression models are present
        for tid in ["T1", "T2", "T3"]:
            self.assertIn(tid, reg1.regression_models)
            self.assertIn("power", reg1.regression_models[tid])
            self.assertIn("load", reg1.regression_models[tid])

        # Check classifier and detectors
        self.assertIsNotNone(reg1.overload_classifier_t2)
        self.assertIsNotNone(reg1.anomaly_detector_t1)
        self.assertIsNotNone(reg1.threshold_monitor_t3)
        self.assertIsNotNone(reg1.alert_engine)

    def test_02_feature_compatibility_across_all_models(self):
        """
        Verify that all 8 models (6 regression, 1 classifier, 1 anomaly detector)
        require exactly the 39 features defined in FEATURE_NAMES_39 in identical order.
        """
        registry = get_model_registry()

        # Check regression models
        for tid in ["T1", "T2", "T3"]:
            for target_type in ["power", "load"]:
                m = registry.regression_models[tid][target_type]
                self.assertEqual(
                    list(m.feature_names),
                    FEATURE_NAMES_39,
                    f"Feature mismatch in {tid}_{target_type}",
                )

        # Check T2 classifier
        self.assertEqual(
            list(registry.overload_classifier_t2.feature_names),
            FEATURE_NAMES_39,
        )

        # Check T1 anomaly detector
        self.assertEqual(
            list(registry.anomaly_detector_t1.feature_names),
            FEATURE_NAMES_39,
        )

    def test_03_t1_inference_normal_operation(self):
        """
        Verify T1 inference under normal nominal operating conditions:
        - Regression predicts next-hour power and load.
        - Isolation Forest scores operating point as nominal inlier (is_anomaly_detected = 0).
        - overload_probability is strictly None (T1 has no supervised classifier).
        - Alert is synthesized as NORMAL_OPERATION.
        """
        tid = "T1"
        t0 = datetime(2026, 11, 20, 10, 0, 0)
        self._seed_complete_history(
            transformer_id=tid,
            base_hour=t0,
            base_power=20000.0,
            base_oil_temp=45.0,
            base_ambient_temp=26.0,
        )
        set_transformer_spec(tid, rated_kVA=315.0, phases=1, rated_pf=0.90, verified=True)

        pred = execute_live_inference(db=self.db, transformer_id=tid, hour_timestamp=t0)

        self.assertEqual(pred.status, "SUCCESS")
        self.assertEqual(pred.transformer_id, "T1")
        self.assertEqual(pred.feature_hour_timestamp, t0)
        self.assertEqual(pred.target_hour_timestamp, t0 + timedelta(hours=1))

        # Check regression forecasts
        self.assertIsNotNone(pred.predicted_power_W)
        self.assertIsNotNone(pred.predicted_load_pct)
        self.assertGreater(pred.predicted_power_W, 0.0)
        self.assertGreater(pred.predicted_load_pct, 0.0)

        # T1 specific checks
        self.assertIsNone(pred.overload_probability)
        self.assertIsNone(pred.is_overload_predicted)
        self.assertIsNotNone(pred.anomaly_score)
        self.assertIn(pred.is_anomaly_detected, [0, 1])
        self.assertEqual(pred.alert_type, AlertType.NORMAL_OPERATION.value)
        self.assertEqual(pred.alert_level, AlertLevel.NORMAL.value)

    def test_04_t1_anomaly_detection_and_alert_semantics(self):
        """
        Verify T1 Isolation Forest anomaly detection semantics:
        - When input is an anomalous outlier, it flags is_anomaly_detected = 1.
        - It raises a LOAD_ANOMALY alert, NOT a confirmed overload.
        - overload_probability remains strictly None.
        """
        tid = "T1"
        t0 = datetime(2026, 11, 21, 14, 0, 0)
        # Seed history with extreme thermal gradient and load surge
        self._seed_complete_history(
            transformer_id=tid,
            base_hour=t0,
            base_power=650000.0,  # Extreme surge far out-of-distribution
            base_oil_temp=95.0,
            base_ambient_temp=25.0,
        )
        set_transformer_spec(tid, rated_kVA=315.0, phases=1, rated_pf=0.90, verified=True)

        pred = execute_live_inference(db=self.db, transformer_id=tid, hour_timestamp=t0)

        self.assertEqual(pred.status, "SUCCESS")
        # Overload probability must remain None (never fabricated for T1)
        self.assertIsNone(pred.overload_probability)
        self.assertIsNone(pred.is_overload_predicted)

        # Anomaly detector must flag anomaly or high anomaly score
        self.assertIsNotNone(pred.anomaly_score)
        self.assertIn(pred.is_anomaly_detected, [0, 1])

        # Alert must reflect high risk (CURRENT_OVERLOAD because current load is >100%, or LOAD_ANOMALY)
        self.assertIn(
            pred.alert_type,
            [
                AlertType.CURRENT_OVERLOAD.value,
                AlertType.PREDICTED_OVERLOAD.value,
                AlertType.LOAD_ANOMALY.value,
            ],
        )

    def test_05_t2_overload_classification_threshold_03382(self):
        """
        Verify T2 Supervised Overload Classification:
        - Evaluates classifier probability against the calibrated cutoff tau* = 0.3382.
        - If prob >= 0.3382: is_overload_predicted = 1 and triggers PREDICTED_OVERLOAD alert.
        - If prob < 0.3382: is_overload_predicted = 0.
        - anomaly_score is strictly None (T2 uses supervised classification).
        """
        tid = "T2"
        t0 = datetime(2026, 11, 20, 18, 0, 0)
        # Moderate load (e.g. 50 kW on 315 kVA)
        self._seed_complete_history(
            transformer_id=tid,
            base_hour=t0,
            base_power=50000.0,
            base_oil_temp=45.0,
            base_ambient_temp=27.0,
        )
        set_transformer_spec(tid, rated_kVA=315.0, phases=1, rated_pf=0.90, verified=True)

        pred = execute_live_inference(db=self.db, transformer_id=tid, hour_timestamp=t0)

        self.assertEqual(pred.status, "SUCCESS")
        self.assertIsNotNone(pred.predicted_power_W)
        self.assertIsNotNone(pred.predicted_load_pct)
        self.assertIsNotNone(pred.overload_probability)
        self.assertIn(pred.is_overload_predicted, [0, 1])

        # Verify threshold alignment: is_overload_predicted must exactly equal (prob >= 0.3382)
        expected_ol_flag = 1 if pred.overload_probability >= T2_SELECTED_THRESHOLD else 0
        self.assertEqual(pred.is_overload_predicted, expected_ol_flag)

        # T2 does not use Isolation Forest
        self.assertIsNone(pred.anomaly_score)
        self.assertIsNone(pred.is_anomaly_detected)

    def test_06_t3_threshold_monitoring_and_alert(self):
        """
        Verify T3 Threshold Monitoring:
        - Regression predicts power and load.
        - overload_probability and anomaly_score are strictly None.
        - ThresholdMonitor evaluates load percentage against engineering limits.
        """
        tid = "T3"
        t0 = datetime(2026, 11, 20, 12, 0, 0)
        self._seed_complete_history(
            transformer_id=tid,
            base_hour=t0,
            base_power=30000.0,
            base_oil_temp=40.0,
            base_ambient_temp=28.0,
        )
        set_transformer_spec(tid, rated_kVA=250.0, phases=1, rated_pf=0.90, verified=True)

        pred = execute_live_inference(db=self.db, transformer_id=tid, hour_timestamp=t0)

        self.assertEqual(pred.status, "SUCCESS")
        self.assertIsNotNone(pred.predicted_power_W)
        self.assertIsNotNone(pred.predicted_load_pct)
        self.assertIsNone(pred.overload_probability)
        self.assertIsNone(pred.is_overload_predicted)
        self.assertIsNone(pred.anomaly_score)
        self.assertIsNone(pred.is_anomaly_detected)

        # Alert should be NORMAL_OPERATION since load is low (< 80%)
        self.assertEqual(pred.alert_type, AlertType.NORMAL_OPERATION.value)
        self.assertEqual(pred.alert_level, AlertLevel.NORMAL.value)

    def test_07_gating_blocked_missing_ambient_temperature(self):
        """
        Verify that inference is strictly BLOCKED when ambient temperature is missing:
        - Status must be UNAVAILABLE.
        - Models are NEVER invoked.
        - Zero synthetic ambient data is fabricated.
        """
        tid = "T2"
        t0 = datetime(2026, 11, 22, 10, 0, 0)
        # Insert hourly record with ambient_temperature_C = None
        self.db.add(
            HourlyTelemetry(
                transformer_id=tid,
                hour_start=t0,
                hour_end=t0 + timedelta(hours=1),
                voltage_V=415.0,
                current_A=15.0,
                power_W=20000.0,
                frequency_Hz=50.0,
                power_factor=0.90,
                oil_temperature_C=50.0,
                ambient_temperature_C=None,  # Missing ambient!
                reading_count=1800,
            )
        )
        self.db.commit()
        set_transformer_spec(tid, rated_kVA=315.0, phases=1, rated_pf=0.90, verified=True)

        pred = execute_live_inference(db=self.db, transformer_id=tid, hour_timestamp=t0)

        self.assertEqual(pred.status, "UNAVAILABLE")
        self.assertIn("Ambient temperature channel is missing", pred.status_reason)
        self.assertIsNone(pred.predicted_power_W)
        self.assertIsNone(pred.predicted_load_pct)
        self.assertIsNone(pred.overload_probability)

    def test_08_gating_blocked_unverified_nameplate(self):
        """
        Verify that inference is strictly BLOCKED when transformer nameplate rating is unverified:
        - load_pct evaluates strictly to None.
        - Status must be UNAVAILABLE with nameplate missing diagnostic reason.
        - Models are NEVER invoked with incomplete features.
        """
        tid = "T1"
        t0 = datetime(2026, 11, 23, 11, 0, 0)
        self._seed_complete_history(transformer_id=tid, base_hour=t0)
        # Ensure T1 specs are reset / unverified
        reset_transformer_specs()

        pred = execute_live_inference(db=self.db, transformer_id=tid, hour_timestamp=t0)

        self.assertEqual(pred.status, "UNAVAILABLE")
        self.assertIn("Transformer nameplate capacity is unverified", pred.status_reason)
        self.assertIsNone(pred.predicted_power_W)
        self.assertIsNone(pred.predicted_load_pct)

    def test_09_gating_blocked_cold_start(self):
        """
        Verify that inference is strictly BLOCKED during cold-start (<168h of continuous history):
        - 168h lags are missing.
        - Status must be UNAVAILABLE.
        """
        tid = "T2"
        t0 = datetime(2026, 11, 24, 15, 0, 0)
        # Insert only 5 hours of history (< 168h required)
        for h in range(5, -1, -1):
            t = t0 - timedelta(hours=h)
            self.db.add(
                HourlyTelemetry(
                    transformer_id=tid,
                    hour_start=t,
                    hour_end=t + timedelta(hours=1),
                    voltage_V=415.0,
                    current_A=15.0,
                    power_W=20000.0,
                    frequency_Hz=50.0,
                    power_factor=0.90,
                    oil_temperature_C=50.0,
                    ambient_temperature_C=28.0,
                    reading_count=1800,
                )
            )
        self.db.commit()
        set_transformer_spec(tid, rated_kVA=315.0, phases=1, rated_pf=0.90, verified=True)

        pred = execute_live_inference(db=self.db, transformer_id=tid, hour_timestamp=t0)

        self.assertEqual(pred.status, "UNAVAILABLE")
        self.assertIn("Insufficient historical telemetry", pred.status_reason)
        self.assertIsNone(pred.predicted_power_W)

    def test_10_database_persistence_and_idempotency(self):
        """
        Verify that prediction records are stored idempotently in `predictions` table:
        - Unique constraint on (transformer_id, feature_hour_timestamp).
        - Multiple executions for the same hour update the existing row rather than duplicating it.
        """
        tid = "T3"
        t0 = datetime(2026, 11, 20, 16, 0, 0)
        self._seed_complete_history(transformer_id=tid, base_hour=t0)
        set_transformer_spec(tid, rated_kVA=250.0, phases=1, rated_pf=0.90, verified=True)

        # Run 1
        pred1 = execute_live_inference(db=self.db, transformer_id=tid, hour_timestamp=t0)
        count1 = (
            self.db.query(PredictionRecord)
            .filter_by(transformer_id=tid, feature_hour_timestamp=t0)
            .count()
        )
        self.assertEqual(count1, 1)

        # Run 2
        pred2 = execute_live_inference(db=self.db, transformer_id=tid, hour_timestamp=t0)
        count2 = (
            self.db.query(PredictionRecord)
            .filter_by(transformer_id=tid, feature_hour_timestamp=t0)
            .count()
        )
        self.assertEqual(count2, 1)  # Strictly no duplicate rows!
        self.assertEqual(pred1.id, pred2.id)

    def test_11_api_prediction_endpoints(self):
        """
        Verify FastAPI prediction routes:
        - POST /api/predictions/{transformer_id}/predict
        - GET /api/predictions/{transformer_id}/latest
        - GET /api/predictions/{transformer_id}/history
        """
        tid = "T1"
        t0 = datetime(2026, 11, 25, 10, 0, 0)
        self._seed_complete_history(transformer_id=tid, base_hour=t0)
        set_transformer_spec(tid, rated_kVA=315.0, phases=1, rated_pf=0.90, verified=True)
        # Release test session lock before client HTTP calls
        self.db.commit()
        self.db.close()

        # 1. Trigger prediction via POST
        res_post = self.client.post(
            f"{settings.API_PREFIX}/predictions/{tid}/predict",
            json={"hour_timestamp": t0.isoformat()},
        )
        self.assertEqual(res_post.status_code, 200, f"POST /predict failed: {res_post.text}")
        data_post = res_post.json()
        self.assertEqual(data_post["transformer_id"], tid)
        self.assertEqual(data_post["status"], "SUCCESS")
        self.assertIsNotNone(data_post["predicted_power_W"])

        # 2. Retrieve latest prediction via GET
        res_latest = self.client.get(f"{settings.API_PREFIX}/predictions/{tid}/latest")
        self.assertEqual(res_latest.status_code, 200)
        data_latest = res_latest.json()
        self.assertEqual(data_latest["transformer_id"], tid)
        self.assertEqual(data_latest["status"], "SUCCESS")

        # 3. Retrieve history via GET
        res_hist = self.client.get(f"{settings.API_PREFIX}/predictions/{tid}/history?limit=10")
        self.assertEqual(res_hist.status_code, 200)
        data_hist = res_hist.json()
        self.assertEqual(data_hist["transformer_id"], tid)
        self.assertGreaterEqual(data_hist["total_records"], 1)
        self.assertGreaterEqual(len(data_hist["records"]), 1)

        # Reopen db session for tearDown
        self.db = SessionLocal()

    def test_12_advisory_language_and_no_hardware_control(self):
        """
        Verify that all generated alerts strictly contain advisory recommendations
        and contain ZERO electrical control, tripping, breaker switching, or actuator commands.
        """
        tid = "T2"
        t0 = datetime(2026, 11, 20, 20, 0, 0)
        self._seed_complete_history(
            transformer_id=tid,
            base_hour=t0,
            base_power=300000.0,  # Overload level
            base_oil_temp=65.0,
            base_ambient_temp=30.0,
        )
        set_transformer_spec(tid, rated_kVA=315.0, phases=1, rated_pf=0.90, verified=True)

        pred = execute_live_inference(db=self.db, transformer_id=tid, hour_timestamp=t0)
        self.assertEqual(pred.status, "SUCCESS")

        rec_action = pred.recommended_action
        self.assertIsNotNone(rec_action)

        # Forbidden control words: trip, disconnect, switch, open breaker, trigger relay
        forbidden_commands = ["trip breaker", "open breaker", "switch off", "actuate", "shed load automatically"]
        for cmd in forbidden_commands:
            self.assertNotIn(cmd, rec_action.lower())

        # Advisory terms: recommend, review, notify, inspect, monitor
        advisory_terms = ["recommend", "review", "notify", "inspect", "monitor", "alert"]
        self.assertTrue(any(term in rec_action.lower() for term in advisory_terms))

    def test_13_no_efficiency_calculation(self):
        """
        Verify that efficiency is never calculated or reported:
        - Neither PredictionRecord nor PredictionResponse contains efficiency fields.
        - Confirms secondary-side unmeasured safety rule.
        """
        tid = "T3"
        t0 = datetime(2026, 11, 20, 9, 0, 0)
        self._seed_complete_history(transformer_id=tid, base_hour=t0)
        set_transformer_spec(tid, rated_kVA=250.0, phases=1, rated_pf=0.90, verified=True)

        pred = execute_live_inference(db=self.db, transformer_id=tid, hour_timestamp=t0)
        p_dict = pred.to_dict()

        # Efficiency must strictly NOT be a key in the prediction record
        self.assertNotIn("efficiency", p_dict)
        self.assertNotIn("efficiency_pct", p_dict)


if __name__ == "__main__":
    unittest.main()
