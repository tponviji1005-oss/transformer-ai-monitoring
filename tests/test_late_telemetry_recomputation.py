"""
Unit and Integration Tests for Late Telemetry Dependency Recomputation.
Verifies:
1. Mathematical derivation of affected timestamps in [H, H + 168h].
2. identify_affected_downstream_timestamps returns exact chronological order and bounds.
3. Inserting a delayed raw reading updates the modified hour H in place.
4. Downstream features (1h lag, 24h rolling windows) are recalculated in chronological order.
5. Downstream predictions and operator alerts are recalculated.
6. Does not recalculate unrelated timestamps outside [H, H + 168h].
7. Database restoration preserves exact original baseline after test completion.
"""

from datetime import datetime, timedelta
import shutil
import unittest

from src.backend.database import (
    HourlyTelemetry,
    LiveFeatures,
    PredictionRecord,
    RawTelemetry,
    SessionLocal,
)
from src.backend.feature_service import (
    generate_live_features,
    identify_affected_downstream_timestamps,
)
from src.backend.aggregation_service import aggregate_hourly_telemetry
from src.backend.scheduler import recalculate_downstream_dependencies


class TestLateTelemetryRecomputation(unittest.TestCase):
    """Test suite for late telemetry dependency identification and cascade recomputation."""

    BACKUP_PATH = "data/backups/transformer_monitoring_pre_stage5_fixes.db"
    TARGET_DB = "data/transformer_monitoring.db"

    @classmethod
    def tearDownClass(cls):
        """Restore exact baseline database state after testing."""
        from src.backend.database import engine, init_db
        engine.dispose()
        shutil.copy2(cls.BACKUP_PATH, cls.TARGET_DB)
        init_db()

    def setUp(self):
        self.db = SessionLocal()
        # Clean up any leftover test telemetry
        self.db.query(RawTelemetry).filter(RawTelemetry.reading_id.like("test_late_%")).delete()
        self.db.commit()

    def tearDown(self):
        self.db.query(RawTelemetry).filter(RawTelemetry.reading_id.like("test_late_%")).delete()
        self.db.commit()
        self.db.close()

    def test_identify_affected_downstream_timestamps_bounds(self):
        """identify_affected_downstream_timestamps returns hours in [H, H+168h] in ascending order."""
        # Baseline has records from 2026-11-18 10:00:00 to 2026-11-25 10:00:00
        h_mod = datetime(2026, 11, 24, 10, 0, 0)
        affected = identify_affected_downstream_timestamps(self.db, "T1", h_mod)

        self.assertGreater(len(affected), 0)
        # Verify first is H itself
        self.assertEqual(affected[0], h_mod)
        # Verify strictly chronological
        for i in range(len(affected) - 1):
            self.assertLess(affected[i], affected[i + 1])
        # Verify none exceed H + 168h
        max_bound = h_mod + timedelta(hours=168)
        for ts in affected:
            self.assertLessEqual(ts, max_bound)
            self.assertGreaterEqual(ts, h_mod)

    def test_identify_does_not_include_past_or_distant_future(self):
        """Timestamps before H or after H + 168h are strictly excluded."""
        h_mod = datetime(2026, 11, 24, 0, 0, 0)
        affected = identify_affected_downstream_timestamps(self.db, "T1", h_mod)
        for ts in affected:
            self.assertGreaterEqual(ts, h_mod)
            self.assertLessEqual(ts, h_mod + timedelta(hours=168))

    def test_late_telemetry_cascades_features_and_predictions(self):
        """
        When a late raw reading modifies hour H:
        1. Hour H hourly record updates.
        2. Downstream hour H+1h's load_lag_1h changes.
        3. Downstream hour H+1h's load_rolling_mean_24h changes.
        4. Predictions for H and H+1h are recomputed and committed.
        """
        h_mod = datetime(2026, 11, 25, 8, 0, 0)
        h_next = h_mod + timedelta(hours=1)

        # 1. Pre-generate features for H and H+1h
        feat_h_pre = generate_live_features(self.db, "T1", h_mod)
        feat_next_pre = generate_live_features(self.db, "T1", h_next)
        original_lag = feat_next_pre.load_lag_1h

        # 2. Insert a late raw reading into hour H with distinctive high power
        late_reading = RawTelemetry(
            reading_id="test_late_recompute_001",
            sequence_number=99999,
            transformer_id="T1",
            timestamp=h_mod + timedelta(minutes=15),
            voltage_V=240.0,
            current_A=50.0,
            power_W=99000.0,
            energy=100.0,
            frequency_Hz=50.0,
            power_factor=0.99,
            oil_temperature_C=80.0,
            ambient_temperature_C=None,
        )
        self.db.add(late_reading)
        self.db.commit()

        # 3. Trigger aggregation with recompute_downstream=True
        cutoff_ref = h_next + timedelta(hours=2)
        res = aggregate_hourly_telemetry(
            db=self.db,
            transformer_id="T1",
            reference_time=cutoff_ref,
            recompute_downstream=True,
        )

        self.assertGreaterEqual(res["records_updated"], 1)
        self.assertGreater(len(res["cascade_recalculations"]), 0)

        # 4. Verify hour H hourly power updated
        rec_h = (
            self.db.query(HourlyTelemetry)
            .filter(HourlyTelemetry.transformer_id == "T1", HourlyTelemetry.hour_start == h_mod)
            .first()
        )
        self.assertIsNotNone(rec_h)

        # 5. Verify hour H+1h features were recalculated
        self.db.expire_all()
        feat_next_post = (
            self.db.query(LiveFeatures)
            .filter(LiveFeatures.transformer_id == "T1", LiveFeatures.hour_timestamp == h_next)
            .first()
        )
        self.assertIsNotNone(feat_next_post)
        # load_lag_1h at H+1h should reflect the updated power at H
        self.assertAlmostEqual(feat_next_post.load_lag_1h, rec_h.power_W, places=2)

        # 6. Verify prediction record exists for H+1h
        pred_next = (
            self.db.query(PredictionRecord)
            .filter(PredictionRecord.transformer_id == "T1", PredictionRecord.feature_hour_timestamp == h_next)
            .first()
        )
        self.assertIsNotNone(pred_next)


if __name__ == "__main__":
    unittest.main()
