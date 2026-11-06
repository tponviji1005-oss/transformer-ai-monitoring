"""
End-to-End Integration Test Runner for Transformer AI Monitoring System.
Executes the full pipeline on the isolated simulation database (data/simulation_telemetry.db):
1. Ingestion of raw telemetry (single and batch).
2. Deduplication verification (reading_id).
3. Hourly telemetry aggregation.
4. 39 ML feature generation.
5. Strict safety validations:
   - Missing ambient temperature handling
   - Missing nameplate capacity handling
   - Verification that unavailable predictions are NOT replaced with fabricated values
   - 168-hour history requirement & cold-start gating
   - Festival feature encoding with IST timezone conversion
6. Full ML inference (6 regression models, T2 overload classifier, T1 anomaly detector, T3 monitor).
7. Unified advisory alert generation.
8. Strict verification of production database isolation (production DB records untouched).
"""

from datetime import datetime, timedelta, timezone
import json
import math
import os
import random
import shutil
import sys
from typing import Any, Dict, List
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

# Set workspace root in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.backend.config import settings
from src.backend.database import (
    Base,
    HourlyTelemetry,
    LiveFeatures,
    PredictionRecord,
    RawTelemetry,
    SchedulerLock,
    get_db,
    run_safe_migrations,
)
from src.backend.main import app
from src.backend.aggregation_service import aggregate_hourly_telemetry
from src.backend.feature_service import (
    FEATURE_NAMES_39,
    compute_calendar_cyclical_features,
    compute_festival_features,
    generate_live_features,
    get_feature_readiness,
    get_transformer_spec,
    localize_to_ist,
    reset_transformer_specs,
    set_transformer_spec,
)
from src.backend.inference_service import execute_live_inference
from scripts.simulate_esp32_stream import enforce_safety_isolation


class E2EIntegrationTestRunner:
    def __init__(self):
        self.sim_db_path = os.path.join(BASE_DIR, "data", "simulation_telemetry.db")
        self.prod_db_path = os.path.join(BASE_DIR, "data", "transformer_monitoring.db")
        self.api_key = settings.DEVICE_API_KEY
        self.results: Dict[str, Any] = {
            "test_start_time": datetime.now(timezone.utc).isoformat(),
            "workflow_steps": {},
            "validations": {},
            "database_verification": {},
            "failed_cases": [],
            "hardware_confirmation_issues": [],
        }

    def setup_simulation_db(self):
        """Initializes a clean simulation database with all tables and safe migrations."""
        if os.path.exists(self.sim_db_path):
            os.remove(self.sim_db_path)

        self.sim_engine = create_engine(
            f"sqlite:///{self.sim_db_path}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(bind=self.sim_engine)
        run_safe_migrations(self.sim_engine)
        self.SimSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.sim_engine)

        # Override FastAPI dependency to route all requests to simulation DB
        def override_get_db():
            db = self.SimSessionLocal()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)

    def record_prod_db_state(self) -> Dict[str, int]:
        """Inspects and returns production DB record counts."""
        import sqlite3
        conn = sqlite3.connect(self.prod_db_path)
        c = conn.cursor()
        raw_count = c.execute("SELECT count(*) FROM raw_telemetry").fetchone()[0]
        hourly_count = c.execute("SELECT count(*) FROM hourly_telemetry").fetchone()[0]
        conn.close()
        return {"raw_telemetry": raw_count, "hourly_telemetry": hourly_count}

    def run_e2e_tests(self):
        print("=================================================================")
        print("STARTING E2E INTEGRATION TEST (SIMULATION DATABASE ISOLATION)")
        print("=================================================================")

        # Check production DB before start
        prod_before = self.record_prod_db_state()
        print(f"[PRE-TEST] Production DB state: {prod_before}")
        self.setup_simulation_db()

        # Step 1: Ingestion API (Single & Batch) + Deduplication
        self.test_telemetry_ingestion_and_deduplication()

        # Step 2: Hourly Telemetry Aggregation
        self.test_hourly_aggregation()

        # Step 3: Safety Validation - Missing Ambient Temp & Missing Nameplate
        self.test_safety_gating_missing_ambient_and_nameplate()

        # Step 4: Festival Feature Encoding & Timezone Math
        self.test_festival_features_and_timezone()

        # Step 5: Cold-Start vs Full 168-Hour Telemetry Accumulation
        self.test_168h_cold_start_and_model_inference()

        # Step 6: Overload Classification, Anomaly Detection & Unified Advisory Alerts
        self.test_alert_generation_and_detection()

        # Step 7: Verify Production DB Isolation
        prod_after = self.record_prod_db_state()
        print(f"[POST-TEST] Production DB state: {prod_after}")
        self.results["database_verification"]["prod_db_before"] = prod_before
        self.results["database_verification"]["prod_db_after"] = prod_after
        self.results["database_verification"]["isolation_preserved"] = (prod_before == prod_after)

        if prod_before != prod_after:
            self.results["failed_cases"].append("PRODUCTION DATABASE INTEGRITY VIOLATION: Records were modified!")

        # Hardware blockers
        self.results["hardware_confirmation_issues"] = [
            "Transformer official rated kVA capacity (T1, T2, T3) must be verified by hardware partner in Madurai.",
            "PZEM-004T CT clamp current direction arrow must point towards transformer load.",
            "MAX6675 thermocouple must be mounted in top oil thermowell with thermal grease.",
            "Second physical sensor for ambient temperature is not yet wired to hardware (firmware reports null).",
            "Network reachability and backend host IP must be configured in firmware config.h.",
        ]

        self.results["test_end_time"] = datetime.now(timezone.utc).isoformat()
        return self.results

    def test_telemetry_ingestion_and_deduplication(self):
        print("\n--- [STEP 1] Telemetry Ingestion & Deduplication ---")
        headers = {"X-Device-API-Key": self.api_key}

        # 1. Single reading ingestion
        t0 = datetime(2026, 11, 20, 10, 0, 0)
        single_payload = {
            "reading_id": "T1-BOOT001-00000001",
            "sequence_number": 1,
            "transformer_id": "T1",
            "timestamp": t0.isoformat(),
            "voltage_V": 230.5,
            "current_A": 22.4,
            "power_W": 5163.2,
            "energy": 12.45,
            "frequency_Hz": 50.02,
            "power_factor": 0.92,
            "oil_temperature_C": 52.3,
            "ambient_temperature_C": None,
        }
        res_single = self.client.post("/api/telemetry/ingest", json=single_payload, headers=headers)
        print(f"Single Ingest Response HTTP {res_single.status_code}: {res_single.json()}")
        assert res_single.status_code == 201, "Single ingest must return HTTP 201"

        # 2. Duplicate ingestion (same reading_id)
        res_dup = self.client.post("/api/telemetry/ingest", json=single_payload, headers=headers)
        print(f"Duplicate Ingest Response HTTP {res_dup.status_code}: {res_dup.json()}")
        assert res_dup.status_code == 200, "Duplicate ingest must return HTTP 200"
        assert res_dup.json()["status"] == "duplicate", "Duplicate status must be 'duplicate'"

        # 3. Batch ingestion (10 readings)
        batch_readings = []
        for i in range(2, 12):
            ts = t0 + timedelta(seconds=i * 2)
            batch_readings.append({
                "reading_id": f"T1-BOOT001-{i:08d}",
                "sequence_number": i,
                "transformer_id": "T1",
                "timestamp": ts.isoformat(),
                "voltage_V": 230.0 + random.uniform(-1, 1),
                "current_A": 22.0 + random.uniform(-0.5, 0.5),
                "power_W": 5060.0 + random.uniform(-50, 50),
                "energy": 12.5 + (i * 0.001),
                "frequency_Hz": 50.0,
                "power_factor": 0.92,
                "oil_temperature_C": 52.3 + (i * 0.01),
                "ambient_temperature_C": None,
            })
        res_batch = self.client.post("/api/telemetry/ingest-batch", json={"readings": batch_readings}, headers=headers)
        print(f"Batch Ingest Response HTTP {res_batch.status_code}: {res_batch.json()}")
        assert res_batch.status_code == 200, "Batch ingest must return HTTP 200"
        assert res_batch.json()["inserted_count"] == 10, "Batch must insert all 10 records"

        # Verify DB count in simulation DB
        sim_db = self.SimSessionLocal()
        raw_count = sim_db.query(RawTelemetry).count()
        sim_db.close()
        print(f"Verified raw_telemetry in simulation DB: {raw_count} records (Expected 11)")
        assert raw_count == 11, f"Expected 11 raw records, got {raw_count}"

        self.results["workflow_steps"]["ingestion"] = {
            "status": "PASSED",
            "single_status": res_single.status_code,
            "duplicate_status": res_dup.status_code,
            "batch_inserted": res_batch.json()["inserted_count"],
            "raw_records_inserted": raw_count,
        }

    def test_hourly_aggregation(self):
        print("\n--- [STEP 2] Hourly Telemetry Aggregation ---")
        sim_db = self.SimSessionLocal()
        cutoff_ref = datetime(2026, 11, 20, 12, 0, 0)
        agg_res = aggregate_hourly_telemetry(
            db=sim_db,
            transformer_id="T1",
            reference_time=cutoff_ref,
            recompute_downstream=True,
        )
        print(f"Aggregation Result: {agg_res}")
        assert agg_res["status"] == "success"
        assert agg_res["records_created"] == 1, "Must aggregate hour [10:00, 11:00)"

        hourly_rec = sim_db.query(HourlyTelemetry).filter_by(transformer_id="T1").first()
        assert hourly_rec is not None
        assert hourly_rec.reading_count == 11
        assert hourly_rec.ambient_temperature_C is None, "Ambient temp must remain null when not supplied"
        print(f"HourlyTelemetry Record Created: ID={hourly_rec.id}, hour_start={hourly_rec.hour_start}, "
              f"reading_count={hourly_rec.reading_count}, power_W={hourly_rec.power_W:.2f}W, "
              f"ambient_temperature_C={hourly_rec.ambient_temperature_C}")
        sim_db.close()

        self.results["workflow_steps"]["hourly_aggregation"] = {
            "status": "PASSED",
            "records_created": agg_res["records_created"],
            "reading_count": hourly_rec.reading_count,
            "ambient_is_null": hourly_rec.ambient_temperature_C is None,
        }

    def test_safety_gating_missing_ambient_and_nameplate(self):
        print("\n--- [STEP 3] Safety Gating: Missing Ambient & Nameplate ---")
        sim_db = self.SimSessionLocal()
        target_hour = datetime(2026, 11, 20, 10, 0, 0)

        # Ensure specs are reset to unverified
        reset_transformer_specs()
        spec = get_transformer_spec("T1")
        assert not spec["verified"], "Nameplate must be unverified by default"

        # 1. Feature generation
        feat = generate_live_features(db=sim_db, transformer_id="T1", target_hour=target_hour)
        print(f"Feature Generation Status: {feat.readiness_status}")
        print(f"Missing Features: {feat.missing_features}")
        assert not feat.is_ready_for_inference
        assert "ambient_temperature_C" in feat.missing_features or "ambient_temp_lag_1h" in feat.missing_features

        # 2. Model inference attempt
        inf = execute_live_inference(db=sim_db, transformer_id="T1", hour_timestamp=target_hour)
        print(f"Inference Status: {inf.status}")
        print(f"Status Reason: {inf.status_reason}")
        print(f"Predicted Power: {inf.predicted_power_W}")
        print(f"Predicted Load: {inf.predicted_load_pct}")
        print(f"Alert Level: {inf.alert_level}")

        # CRITICAL SAFETY ASSERTIONS:
        # 1. Must be UNAVAILABLE
        assert inf.status == "UNAVAILABLE", "Inference must be gated as UNAVAILABLE"
        # 2. Must NEVER fabricate values
        assert inf.predicted_power_W is None, "Power prediction must NOT be fabricated"
        assert inf.predicted_load_pct is None, "Load % must NOT be fabricated"
        assert inf.overload_probability is None, "Overload prob must NOT be fabricated"
        assert inf.is_overload_predicted is None, "Overload prediction flag must be None"
        assert inf.alert_level is None, "Alert level must be None when inference is unavailable"
        assert "Ambient temperature channel is missing/unconfigured" in inf.status_reason
        assert "Transformer nameplate capacity is unverified/unconfigured" in inf.status_reason

        sim_db.close()
        self.results["validations"]["missing_sensor_and_nameplate_gating"] = {
            "status": "PASSED",
            "inference_status": inf.status,
            "status_reason": inf.status_reason,
            "no_fabricated_power": inf.predicted_power_W is None,
            "no_fabricated_load": inf.predicted_load_pct is None,
        }

    def test_festival_features_and_timezone(self):
        print("\n--- [STEP 4] Festival Features & Timezone Validation ---")
        # Test Diwali 2026: 2026-11-08
        diwali_dt_utc = datetime(2026, 11, 8, 4, 30, 0, tzinfo=timezone.utc)
        diwali_dt_ist = localize_to_ist(diwali_dt_utc)
        print(f"UTC Time: {diwali_dt_utc} -> IST Time: {diwali_dt_ist}")
        assert diwali_dt_ist.day == 8 and diwali_dt_ist.month == 11

        fest_features = compute_festival_features(diwali_dt_ist)
        print(f"Festival Encoding on Diwali: is_festival={fest_features['is_festival']}, "
              f"fest_diwali={fest_features['fest_diwali']}")
        assert fest_features["is_festival"] == 1.0, "is_festival must be 1.0 on Diwali"
        assert fest_features["fest_diwali"] == 1.0, "fest_diwali must be 1.0"
        assert fest_features["fest_pongal_sankranti"] == 0.0

        # Test Cyclical features
        cyc_features = compute_calendar_cyclical_features(diwali_dt_ist)
        print(f"Cyclical Features: hour_sin={cyc_features['hour_sin']:.4f}, hour_cos={cyc_features['hour_cos']:.4f}")
        for k in ["hour_sin", "hour_cos", "month_sin", "month_cos", "day_sin", "day_cos"]:
            assert -1.0 <= cyc_features[k] <= 1.0, f"{k} must be in [-1, 1]"

        self.results["validations"]["festival_and_timezone"] = {
            "status": "PASSED",
            "diwali_detected": fest_features["fest_diwali"] == 1.0,
            "ist_timezone_applied": str(diwali_dt_ist.tzinfo) == "Asia/Kolkata",
        }

    def test_168h_cold_start_and_model_inference(self):
        print("\n--- [STEP 5 & 6] 168-Hour History & Full Model Inference ---")
        sim_db = self.SimSessionLocal()

        # Seed full 168 continuous hours of telemetry + 1 target hour (169 hours)
        start_time = datetime(2026, 11, 1, 0, 0, 0)
        print(f"Seeding 169 continuous hourly records starting from {start_time.isoformat()}...")

        base_powers = {"T1": 50000.0, "T2": 95000.0, "T3": 65000.0}
        for tid in ["T1", "T2", "T3"]:
            records = []
            for h in range(169):
                h_dt = start_time + timedelta(hours=h)
                hour_frac = h_dt.hour
                diurnal = 0.5 + 0.3 * math.sin((hour_frac - 6.0) / 24.0 * 2.0 * math.pi)
                active_p = base_powers[tid] * diurnal * (1.0 + random.uniform(-0.02, 0.02))
                v = 415.0 + random.uniform(-2, 2)
                pf = 0.92
                current = active_p / (v * pf)
                oil_temp = 45.0 + (active_p / base_powers[tid]) * 20.0
                ambient_temp = 28.0 + 5.0 * math.sin((hour_frac - 9.0) / 24.0 * 2.0 * math.pi)

                records.append(
                    HourlyTelemetry(
                        transformer_id=tid,
                        hour_start=h_dt,
                        hour_end=h_dt + timedelta(hours=1),
                        voltage_V=round(v, 2),
                        current_A=round(current, 2),
                        power_W=round(active_p, 2),
                        frequency_Hz=50.0,
                        power_factor=pf,
                        oil_temperature_C=round(oil_temp, 2),
                        ambient_temperature_C=round(ambient_temp, 2),
                        reading_count=1800,
                        aggregation_status="completed",
                        created_at=datetime.now(timezone.utc),
                    )
                )
            sim_db.add_all(records)
        sim_db.commit()

        # Now configure verified nameplate ratings
        set_transformer_spec("T1", rated_kVA=315.0, phases=1, rated_pf=0.9, verified=True)
        set_transformer_spec("T2", rated_kVA=315.0, phases=1, rated_pf=0.9, verified=True)
        set_transformer_spec("T3", rated_kVA=250.0, phases=1, rated_pf=0.9, verified=True)

        target_test_hour = start_time + timedelta(hours=168)
        print(f"Testing inference at target observation hour: {target_test_hour.isoformat()}")

        # Test T1, T2, T3 inference
        inference_outputs = {}
        for tid in ["T1", "T2", "T3"]:
            feat = generate_live_features(db=sim_db, transformer_id=tid, target_hour=target_test_hour)
            assert feat.is_ready_for_inference, f"{tid} must be ready for inference with 169h history"
            assert feat.readiness_status == "READY_FOR_INFERENCE"
            missing_list = json.loads(feat.missing_features) if feat.missing_features else []
            assert len(missing_list) == 0, f"Expected 0 missing features, got {missing_list}"

            inf = execute_live_inference(db=sim_db, transformer_id=tid, hour_timestamp=target_test_hour)
            print(f"\n[INFERENCE RESULT - {tid}]")
            print(f"  Status: {inf.status}")
            print(f"  Predicted Power: {inf.predicted_power_W:.1f} W")
            print(f"  Predicted Load: {inf.predicted_load_pct:.1f} %")
            print(f"  Overload Probability: {inf.overload_probability}")
            print(f"  Is Overload Predicted: {inf.is_overload_predicted}")
            print(f"  Anomaly Score: {inf.anomaly_score}")
            print(f"  Is Anomaly Detected: {inf.is_anomaly_detected}")
            print(f"  Alert Level: {inf.alert_level}")
            print(f"  Alert Type: {inf.alert_type}")
            print(f"  Action: {inf.recommended_action}")

            assert inf.status == "SUCCESS"
            assert inf.predicted_power_W > 0.0
            assert inf.predicted_load_pct > 0.0
            assert inf.alert_level in ["NORMAL", "WARNING", "CRITICAL"]

            inference_outputs[tid] = {
                "predicted_power_W": inf.predicted_power_W,
                "predicted_load_pct": inf.predicted_load_pct,
                "overload_prob": inf.overload_probability,
                "anomaly_score": inf.anomaly_score,
                "alert_level": inf.alert_level,
                "alert_type": inf.alert_type,
            }

        sim_db.close()
        self.results["workflow_steps"]["model_inference"] = {
            "status": "PASSED",
            "target_hour": target_test_hour.isoformat(),
            "outputs": inference_outputs,
        }

    def test_alert_generation_and_detection(self):
        print("\n--- [STEP 7] Overload Classification & Alert Verification ---")
        sim_db = self.SimSessionLocal()
        # Seed an extreme high-load observation for T2 at target hour (hour 168)
        # 315 kVA * 0.9 pf = 283.5 kW active capacity
        # Setting power_W = 295000 W corresponds to ~104.1% load
        target_hour = datetime(2026, 11, 8, 0, 0, 0)
        hourly_rec = (
            sim_db.query(HourlyTelemetry)
            .filter_by(transformer_id="T2", hour_start=target_hour)
            .first()
        )
        assert hourly_rec is not None, "Hour 168 for T2 must exist from prior step"
        hourly_rec.power_W = 295000.0
        hourly_rec.current_A = 780.0
        hourly_rec.voltage_V = 398.0
        hourly_rec.oil_temperature_C = 88.5
        hourly_rec.ambient_temperature_C = 32.0
        sim_db.commit()

        set_transformer_spec("T2", rated_kVA=315.0, phases=1, rated_pf=0.9, verified=True)
        generate_live_features(db=sim_db, transformer_id="T2", target_hour=target_hour)
        inf_alert = execute_live_inference(db=sim_db, transformer_id="T2", hour_timestamp=target_hour)

        print(f"[HIGH LOAD ALERT EVALUATION - T2]")
        print(f"  Predicted Load: {inf_alert.predicted_load_pct:.1f}%")
        print(f"  Overload Probability: {inf_alert.overload_probability}")
        print(f"  Alert Level: {inf_alert.alert_level}")
        print(f"  Alert Type: {inf_alert.alert_type}")
        print(f"  Reason: {inf_alert.alert_reason}")
        print(f"  Action: {inf_alert.recommended_action}")

        assert inf_alert.alert_level in ["WARNING", "CRITICAL"]
        assert inf_alert.alert_type in ["PREDICTED_OVERLOAD", "CURRENT_OVERLOAD", "ELEVATED_LOAD_WARNING"]
        assert inf_alert.recommended_action != "Continue standard automated monitoring."

        sim_db.close()
        self.results["workflow_steps"]["alert_engine"] = {
            "status": "PASSED",
            "high_load_alert_level": inf_alert.alert_level,
            "high_load_alert_type": inf_alert.alert_type,
            "recommended_action": inf_alert.recommended_action,
        }


if __name__ == "__main__":
    runner = E2EIntegrationTestRunner()
    results = runner.run_e2e_tests()

    report_path = os.path.join(BASE_DIR, "reports", "e2e_simulation_integration_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved structured integration report to {report_path}")
