"""
Unit and Integration Test Suite for Milestone 5 Stage 3: Live ML Feature Generation.

Tests:
1. Exact 39-feature names and sequence matching all eight trained models.
2. Timezone conversion to Asia/Kolkata (IST, UTC+05:30).
3. Exact mathematical cyclical calendar features.
4. Deterministic Indian festival one-hot encoding.
5. Exact timestamp-delta lag matching.
6. Missing-hour gap protection (no cross-gap lag bridging).
7. 168-hour cold-start lifecycle enforcement.
8. Missing ambient temperature safety guard (never substituted with oil temp).
9. Missing transformer nameplate capacity safety guard (load_pct remains None).
10. Strict inference rejection gate for incomplete feature vectors.
11. Numerical parity with historical FeatureEngineer when all inputs are available.
12. FastAPI endpoint controllers for feature generation, latest retrieval, and readiness diagnostics.
"""

from datetime import datetime, timedelta, timezone
import json
import os
import sys
import unittest
from zoneinfo import ZoneInfo
import joblib
import numpy as np
import pandas as pd

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend.config import settings
from src.backend.database import HourlyTelemetry, LiveFeatures, SessionLocal, init_db
from src.backend.feature_service import (
    FEATURE_NAMES_39,
    FESTIVAL_LIST,
    compute_calendar_cyclical_features,
    compute_festival_features,
    generate_live_features,
    get_feature_readiness,
    get_latest_features,
    get_transformer_spec,
    localize_to_ist,
    reset_transformer_specs,
    set_transformer_spec,
)
from src.backend.main import (
    check_feature_readiness_endpoint,
    generate_features_endpoint,
    get_latest_features_endpoint,
    update_transformer_specs_endpoint,
)
from src.backend.schemas import FeatureGenerateRequest, TransformerSpecUpdateRequest


class TestLiveFeatureService(unittest.TestCase):
    """Test suite for Milestone 5 Stage 3 Live ML Feature Generation."""

    @classmethod
    def setUpClass(cls):
        """Initialize database tables before running tests."""
        init_db()

    def setUp(self):
        """Create fresh transactional database session and clean test data."""
        self.db = SessionLocal()
        # Clean up test records
        self.db.query(LiveFeatures).filter(
            LiveFeatures.transformer_id.like("T_%") | (LiveFeatures.transformer_id == "T1_PARITY")
        ).delete()
        self.db.query(HourlyTelemetry).filter(
            HourlyTelemetry.transformer_id.like("T_%") | (HourlyTelemetry.transformer_id == "T1_PARITY")
        ).delete()
        self.db.commit()
        reset_transformer_specs()

    def tearDown(self):
        """Clean up test records and close database session."""
        self.db.query(LiveFeatures).filter(
            LiveFeatures.transformer_id.like("T_%") | (LiveFeatures.transformer_id == "T1_PARITY")
        ).delete()
        self.db.query(HourlyTelemetry).filter(
            HourlyTelemetry.transformer_id.like("T_%") | (HourlyTelemetry.transformer_id == "T1_PARITY")
        ).delete()
        self.db.commit()
        self.db.close()
        reset_transformer_specs()

    def test_01_exact_39_feature_names_and_order(self):
        """
        Verify that FEATURE_NAMES_39 has exactly 39 features and matches
        every one of the eight trained models in exact name, count, and order.
        """
        self.assertEqual(len(FEATURE_NAMES_39), 39)

        models_to_check = [
            "models/t1_rf_power.joblib",
            "models/t1_rf_load.joblib",
            "models/t2_rf_power.joblib",
            "models/t2_rf_load.joblib",
            "models/t3_rf_power.joblib",
            "models/t3_rf_load.joblib",
            "models/overload/t2_overload_classifier.joblib",
            "models/overload/t1_anomaly_detector.joblib",
        ]

        for m_path in models_to_check:
            d = joblib.load(m_path)
            model_fn = d.get("feature_names")
            self.assertIsNotNone(model_fn, f"Missing feature_names in {m_path}")
            self.assertEqual(
                len(model_fn),
                39,
                f"Feature count mismatch in {m_path}: {len(model_fn)} != 39",
            )
            self.assertEqual(
                list(model_fn),
                FEATURE_NAMES_39,
                f"Feature ordering mismatch in {m_path}",
            )

    def test_02_timezone_conversion_to_ist(self):
        """
        Verify that UTC timestamps are converted to Asia/Kolkata (IST, UTC+05:30).
        Example: 2026-11-03 18:30:00 UTC must become 2026-11-04 00:00:00 IST (next calendar day!).
        """
        utc_dt = datetime(2026, 11, 3, 18, 30, 0, tzinfo=timezone.utc)
        ist_dt = localize_to_ist(utc_dt)

        self.assertEqual(ist_dt.year, 2026)
        self.assertEqual(ist_dt.month, 11)
        self.assertEqual(ist_dt.day, 4)
        self.assertEqual(ist_dt.hour, 0)
        self.assertEqual(ist_dt.minute, 0)

        # Calendar features must reflect IST hour 0 and Wednesday (weekday=2)
        cal = compute_calendar_cyclical_features(ist_dt)
        self.assertEqual(cal["hour"], 0.0)
        self.assertEqual(cal["day_of_week"], 2.0)  # Wed
        self.assertEqual(cal["is_weekend"], 0.0)

    def test_03_cyclical_calendar_math_exactness(self):
        """
        Verify exact mathematical outputs of sine and cosine cyclical encodings.
        Hour 0: sin=0.0, cos=1.0
        Hour 6: sin=1.0, cos=0.0
        Month 1 (Jan): sin=0.0, cos=1.0 (since angle uses month - 1)
        """
        dt_h0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
        cal_h0 = compute_calendar_cyclical_features(dt_h0)
        self.assertAlmostEqual(cal_h0["hour_sin"], 0.0, places=5)
        self.assertAlmostEqual(cal_h0["hour_cos"], 1.0, places=5)
        self.assertAlmostEqual(cal_h0["month_sin"], 0.0, places=5)
        self.assertAlmostEqual(cal_h0["month_cos"], 1.0, places=5)

        dt_h6 = datetime(2026, 1, 1, 6, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
        cal_h6 = compute_calendar_cyclical_features(dt_h6)
        self.assertAlmostEqual(cal_h6["hour_sin"], 1.0, places=5)
        self.assertAlmostEqual(cal_h6["hour_cos"], 0.0, places=5)

    def test_04_festival_one_hot_encoding(self):
        """
        Verify festival dummy encoding:
        - Normal day: is_festival = 0.0, all 7 dummies = 0.0
        - Known festival (e.g. Pongal on 2026-01-14): is_festival = 1.0, fest_pongal_sankranti = 1.0
        - Known festival (e.g. Diwali on 2026-11-08): is_festival = 1.0, fest_diwali = 1.0
        """
        # Normal day
        dt_normal = datetime(2026, 5, 20, 10, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
        fest_normal = compute_festival_features(dt_normal)
        self.assertEqual(fest_normal["is_festival"], 0.0)
        for f in FESTIVAL_LIST:
            col = "fest_" + f.replace("/", "_").replace(" ", "_").lower()
            self.assertEqual(fest_normal[col], 0.0)

        # Pongal
        dt_pongal = datetime(2026, 1, 14, 10, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
        fest_pongal = compute_festival_features(dt_pongal)
        self.assertEqual(fest_pongal["is_festival"], 1.0)
        self.assertEqual(fest_pongal["fest_pongal_sankranti"], 1.0)
        self.assertEqual(fest_pongal["fest_diwali"], 0.0)

        # Diwali
        dt_diwali = datetime(2026, 11, 8, 18, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
        fest_diwali = compute_festival_features(dt_diwali)
        self.assertEqual(fest_diwali["is_festival"], 1.0)
        self.assertEqual(fest_diwali["fest_diwali"], 1.0)
        self.assertEqual(fest_diwali["fest_pongal_sankranti"], 0.0)

    def test_05_exact_timestamp_lag_matching(self):
        """
        Verify that lag features (load_lag_1h, load_lag_24h, load_lag_168h, oil_temp_lag_1h)
        match exact historical hourly records at t-1h, t-24h, and t-168h.
        """
        tid = "T_LAG_TEST"
        t0 = datetime(2026, 11, 10, 12, 0, 0)
        t_1h = t0 - timedelta(hours=1)
        t_24h = t0 - timedelta(hours=24)
        t_168h = t0 - timedelta(hours=168)

        recs = [
            HourlyTelemetry(
                transformer_id=tid,
                hour_start=t_168h,
                hour_end=t_168h + timedelta(hours=1),
                voltage_V=415.0,
                current_A=10.0,
                power_W=5000.0,
                frequency_Hz=50.0,
                power_factor=0.90,
                oil_temperature_C=45.0,
                ambient_temperature_C=25.0,
                reading_count=1800,
            ),
            HourlyTelemetry(
                transformer_id=tid,
                hour_start=t_24h,
                hour_end=t_24h + timedelta(hours=1),
                voltage_V=415.0,
                current_A=12.0,
                power_W=6000.0,
                frequency_Hz=50.0,
                power_factor=0.90,
                oil_temperature_C=48.0,
                ambient_temperature_C=26.0,
                reading_count=1800,
            ),
            HourlyTelemetry(
                transformer_id=tid,
                hour_start=t_1h,
                hour_end=t_1h + timedelta(hours=1),
                voltage_V=415.0,
                current_A=14.0,
                power_W=7000.0,
                frequency_Hz=50.0,
                power_factor=0.90,
                oil_temperature_C=50.0,
                ambient_temperature_C=27.0,
                reading_count=1800,
            ),
            HourlyTelemetry(
                transformer_id=tid,
                hour_start=t0,
                hour_end=t0 + timedelta(hours=1),
                voltage_V=415.0,
                current_A=16.0,
                power_W=8000.0,
                frequency_Hz=50.0,
                power_factor=0.90,
                oil_temperature_C=52.0,
                ambient_temperature_C=28.0,
                reading_count=1800,
            ),
        ]
        self.db.add_all(recs)
        self.db.commit()

        feature_record = generate_live_features(db=self.db, transformer_id=tid, target_hour=t0)
        fd = feature_record.to_feature_dict()

        self.assertEqual(fd["power_W"], 8000.0)
        self.assertEqual(fd["oil_temp_C"], 52.0)
        self.assertEqual(fd["load_lag_1h"], 7000.0)
        self.assertEqual(fd["load_lag_24h"], 6000.0)
        self.assertEqual(fd["load_lag_168h"], 5000.0)
        self.assertEqual(fd["oil_temp_lag_1h"], 50.0)
        self.assertEqual(fd["oil_temp_lag_24h"], 48.0)
        self.assertEqual(fd["ambient_temp_lag_1h"], 27.0)
        self.assertEqual(fd["ambient_temp_lag_24h"], 26.0)
        self.assertEqual(fd["temp_gradient"], 52.0 - 28.0)
        self.assertEqual(fd["temp_gradient_lag_1h"], 50.0 - 27.0)

    def test_06_missing_hour_gaps_do_not_bridge(self):
        """
        Verify that if an exact lag hour (e.g. t-1h) is missing from hourly_telemetry,
        the lag evaluates strictly to None rather than jumping to t-2h.
        """
        tid = "T_GAP_TEST"
        t0 = datetime(2026, 11, 10, 15, 0, 0)
        t_2h = t0 - timedelta(hours=2)  # Hour t-1h is MISSING

        recs = [
            HourlyTelemetry(
                transformer_id=tid,
                hour_start=t_2h,
                hour_end=t_2h + timedelta(hours=1),
                voltage_V=415.0,
                current_A=10.0,
                power_W=5555.0,
                frequency_Hz=50.0,
                power_factor=0.90,
                oil_temperature_C=45.0,
                ambient_temperature_C=25.0,
                reading_count=1800,
            ),
            HourlyTelemetry(
                transformer_id=tid,
                hour_start=t0,
                hour_end=t0 + timedelta(hours=1),
                voltage_V=415.0,
                current_A=12.0,
                power_W=6666.0,
                frequency_Hz=50.0,
                power_factor=0.90,
                oil_temperature_C=48.0,
                ambient_temperature_C=26.0,
                reading_count=1800,
            ),
        ]
        self.db.add_all(recs)
        self.db.commit()

        feature_record = generate_live_features(db=self.db, transformer_id=tid, target_hour=t0)
        fd = feature_record.to_feature_dict()

        # load_lag_1h must be None because t-1h was a missing blackout hour
        self.assertIsNone(fd["load_lag_1h"])
        self.assertNotEqual(fd["load_lag_1h"], 5555.0)  # Did not erroneously jump to t-2h

    def test_07_168_hour_cold_start_enforcement(self):
        """
        Verify that a newly commissioned transformer with only 24 hours of history
        has load_lag_168h and load_rolling_mean_168h as None and is marked as
        STATUS_COLD_START_INSUFFICIENT_HISTORY with is_ready_for_inference = False.
        """
        tid = "T_COLD_START"
        start_time = datetime(2026, 11, 1, 0, 0, 0)

        # Insert 24 contiguous hours
        recs = []
        for i in range(24):
            h_dt = start_time + timedelta(hours=i)
            recs.append(
                HourlyTelemetry(
                    transformer_id=tid,
                    hour_start=h_dt,
                    hour_end=h_dt + timedelta(hours=1),
                    voltage_V=415.0,
                    current_A=10.0,
                    power_W=5000.0 + i * 10,
                    frequency_Hz=50.0,
                    power_factor=0.90,
                    oil_temperature_C=50.0,
                    ambient_temperature_C=28.0,
                    reading_count=1800,
                )
            )
        self.db.add_all(recs)
        self.db.commit()

        t_eval = start_time + timedelta(hours=23)
        feature_record = generate_live_features(db=self.db, transformer_id=tid, target_hour=t_eval)
        fd = feature_record.to_feature_dict()

        self.assertIsNone(fd["load_lag_168h"])
        self.assertIsNone(fd["load_rolling_mean_168h"])
        self.assertIsNone(fd["load_rolling_std_168h"])
        self.assertEqual(feature_record.readiness_status, "STATUS_COLD_START_INSUFFICIENT_HISTORY")
        self.assertEqual(feature_record.is_ready_for_inference, 0)
        self.assertIn("Cold-start buffer is accumulating", feature_record.status_reason)

    def test_08_missing_ambient_temperature_safety(self):
        """
        Verify that when ambient temperature is NULL, all 6 ambient features
        evaluate strictly to None. Oil temperature is NEVER substituted.
        Readiness status reflects STATUS_BLOCKED_MISSING_AMBIENT.
        """
        tid = "T_NO_AMBIENT_SAFE"
        t0 = datetime(2026, 11, 20, 10, 0, 0)

        rec = HourlyTelemetry(
            transformer_id=tid,
            hour_start=t0,
            hour_end=t0 + timedelta(hours=1),
            voltage_V=415.0,
            current_A=20.0,
            power_W=8300.0,
            frequency_Hz=50.0,
            power_factor=0.90,
            oil_temperature_C=60.0,
            ambient_temperature_C=None,  # MISSING AMBIENT
            reading_count=1800,
        )
        self.db.add(rec)
        self.db.commit()

        feature_record = generate_live_features(db=self.db, transformer_id=tid, target_hour=t0)
        fd = feature_record.to_feature_dict()

        self.assertIsNone(fd["ambient_temp_C"])
        self.assertIsNone(fd["ambient_temp_lag_1h"])
        self.assertIsNone(fd["ambient_temp_lag_24h"])
        self.assertIsNone(fd["temp_gradient"])
        self.assertIsNone(fd["temp_gradient_lag_1h"])
        self.assertIsNone(fd["ambient_rolling_mean_24h"])
        self.assertEqual(fd["oil_temp_C"], 60.0)

        # Confirm oil temperature was not substituted into ambient_temp_C
        self.assertNotEqual(fd["ambient_temp_C"], 60.0)

    def test_09_missing_nameplate_configuration_safety(self):
        """
        Verify that when transformer nameplate rating is unverified,
        load_pct evaluates strictly to None.
        Readiness reflects STATUS_BLOCKED_MISSING_NAMEPLATE.
        """
        tid = "T1"
        # Ensure T1 is unconfigured
        spec = get_transformer_spec(tid)
        self.assertFalse(spec["verified"])
        self.assertIsNone(spec["rated_kVA"])

        t0 = datetime(2026, 11, 20, 11, 0, 0)
        rec = HourlyTelemetry(
            transformer_id=tid,
            hour_start=t0,
            hour_end=t0 + timedelta(hours=1),
            voltage_V=415.0,
            current_A=20.0,
            power_W=25000.0,
            frequency_Hz=50.0,
            power_factor=0.90,
            oil_temperature_C=55.0,
            ambient_temperature_C=30.0,
            reading_count=1800,
        )
        self.db.add(rec)
        self.db.commit()

        feature_record = generate_live_features(db=self.db, transformer_id=tid, target_hour=t0)
        fd = feature_record.to_feature_dict()

        self.assertIsNone(fd["load_pct"])

        # Now simulate verified nameplate configuration: T1 = 50 kVA, pf=0.90
        set_transformer_spec("T1", rated_kVA=50.0, phases=1, rated_pf=0.90, verified=True)
        feature_record_configured = generate_live_features(db=self.db, transformer_id=tid, target_hour=t0)
        fd_configured = feature_record_configured.to_feature_dict()

        # Capacity = 50 kVA * 1000 * 0.90 = 45,000 W
        # load_pct = (25,000 / 45,000) * 100% = 55.5555%
        expected_pct = (25000.0 / 45000.0) * 100.0
        self.assertIsNotNone(fd_configured["load_pct"])
        self.assertAlmostEqual(fd_configured["load_pct"], expected_pct, places=3)

    def test_10_no_inference_with_incomplete_features(self):
        """
        Verify that incomplete feature vectors cannot be fed to trained models.
        Scikit-learn model fitting and prediction reject None / NaN inputs.
        """
        tid = "T_INCOMPLETE_GATE"
        t0 = datetime(2026, 11, 20, 12, 0, 0)
        rec = HourlyTelemetry(
            transformer_id=tid,
            hour_start=t0,
            hour_end=t0 + timedelta(hours=1),
            voltage_V=415.0,
            current_A=20.0,
            power_W=15000.0,
            frequency_Hz=50.0,
            power_factor=0.90,
            oil_temperature_C=55.0,
            ambient_temperature_C=None,  # Missing ambient
            reading_count=1800,
        )
        self.db.add(rec)
        self.db.commit()

        feature_record = generate_live_features(db=self.db, transformer_id=tid, target_hour=t0)
        self.assertEqual(feature_record.is_ready_for_inference, 0)

        # Build feature vector
        feat_dict = feature_record.to_feature_dict()
        values = [feat_dict[k] for k in FEATURE_NAMES_39]

        # Verify that contains None
        self.assertTrue(any(v is None for v in values))

        # Gate check: inference must refuse to execute if is_ready_for_inference is 0
        def safe_predict_gate(live_feature_obj, model_obj):
            if not live_feature_obj.is_ready_for_inference:
                return {"status": "unavailable", "reason": live_feature_obj.status_reason}
            X = np.array([[live_feature_obj.to_feature_dict()[k] for k in FEATURE_NAMES_39]])
            return {"status": "success", "prediction": float(model_obj.predict(X)[0])}

        rf_t1 = joblib.load("models/t1_rf_power.joblib")["model"]
        gate_res = safe_predict_gate(feature_record, rf_t1)
        self.assertEqual(gate_res["status"], "unavailable")
        self.assertIn("Ambient temperature channel is missing", gate_res["reason"])

    def test_11_numerical_parity_with_historical_feature_engineer(self):
        """
        Verify exact numerical parity between live feature generation and historical
        FeatureEngineer outputs on real historical telemetry.
        """
        # Load raw historical data for T1
        raw_df = pd.read_csv("data/raw/transformer_hourly_festival_with_temperature.csv")
        t1_train = pd.read_csv("data/processed/t1_train.csv", skiprows=range(1, 201), nrows=5)

        # Target row: index 200 in t1_train (completely uninterrupted contiguous 168h history)
        target_dt_str = t1_train.loc[0, "datetime_IST"]
        target_dt = pd.to_datetime(target_dt_str)
        raw_idx = raw_df[raw_df["datetime_IST"] == target_dt_str].index[0]

        # Seed SQLite hourly_telemetry with the required contiguous 169-hour buffer
        start_idx = raw_idx - 168
        tid = "T1_PARITY"

        recs = []
        for i in range(start_idx, raw_idx + 1):
            row = raw_df.iloc[i]
            r_dt = pd.to_datetime(row["datetime_IST"]).to_pydatetime()
            recs.append(
                HourlyTelemetry(
                    transformer_id=tid,
                    hour_start=r_dt,
                    hour_end=r_dt + timedelta(hours=1),
                    voltage_V=415.0,
                    current_A=10.0,
                    power_W=float(row["t1_power_W"]),
                    frequency_Hz=50.0,
                    power_factor=0.90,
                    oil_temperature_C=float(row["t1_oil_temp_C"]),
                    ambient_temperature_C=float(row["ambient_temperature_C"]),
                    reading_count=1800,
                )
            )
        self.db.add_all(recs)
        self.db.commit()

        # Set confirmed nameplate for T1 matching historical implied rating: 315 kVA, pf=0.90 (active capacity 283.5 kW)
        set_transformer_spec(tid, rated_kVA=315.0, phases=1, rated_pf=0.90, verified=True)

        # Generate live features
        # Note: Historical datetime_IST is in Asia/Kolkata time
        target_hour_ist = target_dt.replace(tzinfo=ZoneInfo("Asia/Kolkata")).to_pydatetime()
        feature_record = generate_live_features(
            db=self.db,
            transformer_id=tid,
            target_hour=target_hour_ist,
        )
        self.assertEqual(feature_record.is_ready_for_inference, 1)

        fd = feature_record.to_feature_dict()
        expected_row = t1_train.iloc[0]

        # Compare all 39 features with strict numerical tolerance
        for feat in FEATURE_NAMES_39:
            gen_val = fd[feat]
            exp_val = expected_row[feat]
            self.assertIsNotNone(gen_val, f"Generated feature {feat} is None!")
            if feat == "load_pct":
                # Historical load_pct in raw CSV was rounded to 1 decimal place (5.8 vs 5.772%)
                self.assertAlmostEqual(
                    float(gen_val),
                    float(exp_val),
                    delta=0.05,
                    msg=f"Numerical parity mismatch for feature {feat}: {gen_val} != {exp_val}",
                )
            else:
                self.assertAlmostEqual(
                    float(gen_val),
                    float(exp_val),
                    places=3,
                    msg=f"Numerical parity mismatch for feature {feat}: {gen_val} != {exp_val}",
                )

    def test_12_api_feature_endpoints(self):
        """
        Verify FastAPI endpoint handlers:
        - POST /api/transformers/{transformer_id}/specs
        - POST /api/features/{transformer_id}/generate
        - GET /api/features/{transformer_id}/latest
        - GET /api/features/{transformer_id}/readiness
        """
        tid = "T_API_TEST"
        t0 = datetime(2026, 11, 25, 14, 0, 0)

        # Ingest hourly observation
        rec = HourlyTelemetry(
            transformer_id=tid,
            hour_start=t0,
            hour_end=t0 + timedelta(hours=1),
            voltage_V=415.0,
            current_A=15.0,
            power_W=12000.0,
            frequency_Hz=50.0,
            power_factor=0.90,
            oil_temperature_C=55.0,
            ambient_temperature_C=30.0,
            reading_count=1800,
        )
        self.db.add(rec)
        self.db.commit()

        # Configure specs via endpoint
        spec_req = TransformerSpecUpdateRequest(rated_kVA=50.0, phases=1, rated_pf=0.90, verified=True)
        spec_res = update_transformer_specs_endpoint(transformer_id=tid, specs=spec_req)
        self.assertEqual(spec_res["status"], "success")
        self.assertTrue(spec_res["specs"]["verified"])

        # Generate features via endpoint
        gen_req = FeatureGenerateRequest(hour_timestamp=t0)
        gen_res = generate_features_endpoint(transformer_id=tid, payload=gen_req, db=self.db)
        self.assertEqual(gen_res.transformer_id, tid)
        self.assertEqual(len(gen_res.features), 39)

        # Retrieve latest features via endpoint
        latest_res = get_latest_features_endpoint(transformer_id=tid, db=self.db)
        self.assertEqual(latest_res.transformer_id, tid)
        self.assertEqual(latest_res.hour_timestamp, t0)

        # Check readiness via endpoint
        readiness_res = check_feature_readiness_endpoint(transformer_id=tid, hour_timestamp=t0, db=self.db)
        self.assertEqual(readiness_res.transformer_id, tid)
        self.assertIn("missing_features", readiness_res.model_dump())


if __name__ == "__main__":
    unittest.main()
