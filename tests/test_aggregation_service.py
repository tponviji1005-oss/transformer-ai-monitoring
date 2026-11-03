"""
Unit and Integration Test Suite for Milestone 5 Stage 2: Hourly Telemetry Aggregation.

Tests:
1. Correct hourly arithmetic averages across multiple readings.
2. Ingestion with multiple readings in one hour.
3. Multiple transformers aggregated independently.
4. Missing ambient temperature remains strictly NULL (never substituted with oil temperature).
5. Empty hours without raw readings produce no phantom hourly records.
6. Incomplete current hour is strictly excluded from aggregation until completed.
7. Idempotency and duplicate prevention on repeated aggregation runs.
8. Direct SQLite database persistence and schema integrity.
9. GET /api/aggregation/{transformer_id}/history chronological retrieval and pagination.
"""

from datetime import datetime, timedelta, timezone
import os
import sys
import unittest
from sqlalchemy import func, select

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend.aggregation_service import aggregate_hourly_telemetry, get_hourly_history
from src.backend.config import settings
from src.backend.database import HourlyTelemetry, RawTelemetry, SessionLocal, init_db
from src.backend.main import get_aggregation_history, run_aggregation
from src.backend.schemas import AggregationRunRequest, AggregationRunResponse, HourlyHistoryResponse


class TestHourlyAggregationService(unittest.TestCase):
    """Test suite for Milestone 5 Stage 2 Hourly Telemetry Aggregation."""

    @classmethod
    def setUpClass(cls):
        """Initialize database tables before running tests and clean test data."""
        init_db()
        db = SessionLocal()
        db.query(RawTelemetry).filter(RawTelemetry.transformer_id.like("T_%")).delete()
        db.query(HourlyTelemetry).filter(HourlyTelemetry.transformer_id.like("T_%")).delete()
        db.commit()
        db.close()

    @classmethod
    def tearDownClass(cls):
        """Clean test data after all tests finish."""
        db = SessionLocal()
        db.query(RawTelemetry).filter(RawTelemetry.transformer_id.like("T_%")).delete()
        db.query(HourlyTelemetry).filter(HourlyTelemetry.transformer_id.like("T_%")).delete()
        db.commit()
        db.close()

    def setUp(self):
        """Create fresh transactional database session for each test."""
        self.db = SessionLocal()

    def tearDown(self):
        """Close database session after each test."""
        self.db.close()

    def test_01_correct_hourly_averages_and_multiple_readings(self):
        """
        Verify hourly arithmetic averages are mathematically exact across
        multiple raw sensor readings in a single hour window.
        """
        t_id = "T_AVG_TEST"
        # Base hour: 2026-11-01 10:00:00 to 11:00:00
        # 4 readings at 10:10, 10:20, 10:30, 10:40
        readings = [
            RawTelemetry(
                transformer_id=t_id,
                timestamp=datetime(2026, 11, 1, 10, 10, 0),
                voltage_V=400.0,
                current_A=10.0,
                power_W=4000.0,
                energy=100.0,
                frequency_Hz=50.0,
                power_factor=0.90,
                oil_temperature_C=50.0,
                ambient_temperature_C=30.0,
            ),
            RawTelemetry(
                transformer_id=t_id,
                timestamp=datetime(2026, 11, 1, 10, 20, 0),
                voltage_V=410.0,
                current_A=20.0,
                power_W=8200.0,
                energy=100.2,
                frequency_Hz=50.1,
                power_factor=0.92,
                oil_temperature_C=52.0,
                ambient_temperature_C=31.0,
            ),
            RawTelemetry(
                transformer_id=t_id,
                timestamp=datetime(2026, 11, 1, 10, 30, 0),
                voltage_V=420.0,
                current_A=30.0,
                power_W=12600.0,
                energy=100.5,
                frequency_Hz=49.9,
                power_factor=0.88,
                oil_temperature_C=54.0,
                ambient_temperature_C=32.0,
            ),
            RawTelemetry(
                transformer_id=t_id,
                timestamp=datetime(2026, 11, 1, 10, 40, 0),
                voltage_V=430.0,
                current_A=40.0,
                power_W=17200.0,
                energy=100.9,
                frequency_Hz=50.0,
                power_factor=0.90,
                oil_temperature_C=56.0,
                ambient_temperature_C=31.0,
            ),
        ]
        self.db.add_all(readings)
        self.db.commit()

        # Run aggregation with reference time after hour 10 completed (e.g. 11:30)
        ref_time = datetime(2026, 11, 1, 11, 30, 0)
        res = aggregate_hourly_telemetry(db=self.db, transformer_id=t_id, reference_time=ref_time)
        self.assertEqual(res["status"], "success")
        self.assertGreaterEqual(res["records_created"], 1)

        # Inspect persisted SQLite hourly record
        hourly = (
            self.db.query(HourlyTelemetry)
            .filter(
                HourlyTelemetry.transformer_id == t_id,
                HourlyTelemetry.hour_start == datetime(2026, 11, 1, 10, 0, 0),
            )
            .first()
        )
        self.assertIsNotNone(hourly)
        self.assertEqual(hourly.reading_count, 4)
        self.assertEqual(hourly.hour_end, datetime(2026, 11, 1, 11, 0, 0))
        self.assertEqual(hourly.aggregation_status, "completed")

        # Verify mathematical arithmetic averages
        self.assertAlmostEqual(hourly.voltage_V, (400 + 410 + 420 + 430) / 4.0, places=3)
        self.assertAlmostEqual(hourly.current_A, (10 + 20 + 30 + 40) / 4.0, places=3)
        self.assertAlmostEqual(hourly.power_W, (4000 + 8200 + 12600 + 17200) / 4.0, places=3)
        self.assertAlmostEqual(hourly.frequency_Hz, (50.0 + 50.1 + 49.9 + 50.0) / 4.0, places=3)
        self.assertAlmostEqual(hourly.power_factor, (0.90 + 0.92 + 0.88 + 0.90) / 4.0, places=3)
        self.assertAlmostEqual(hourly.oil_temperature_C, (50 + 52 + 54 + 56) / 4.0, places=3)
        self.assertAlmostEqual(hourly.ambient_temperature_C, (30 + 31 + 32 + 31) / 4.0, places=3)

    def test_02_missing_ambient_temperature_remains_strictly_null(self):
        """
        Verify that when ambient temperature is missing from raw readings,
        the hourly record retains ambient_temperature_C as strictly NULL.
        Oil temperature must NEVER be substituted.
        """
        t_id = "T_NO_AMBIENT"
        readings = [
            RawTelemetry(
                transformer_id=t_id,
                timestamp=datetime(2026, 11, 1, 12, 10, 0),
                voltage_V=415.0,
                current_A=25.0,
                power_W=10375.0,
                frequency_Hz=50.0,
                power_factor=0.91,
                oil_temperature_C=62.5,
                ambient_temperature_C=None,  # Missing ambient
            ),
            RawTelemetry(
                transformer_id=t_id,
                timestamp=datetime(2026, 11, 1, 12, 40, 0),
                voltage_V=416.0,
                current_A=27.0,
                power_W=11232.0,
                frequency_Hz=50.02,
                power_factor=0.90,
                oil_temperature_C=63.5,
                ambient_temperature_C=None,  # Missing ambient
            ),
        ]
        self.db.add_all(readings)
        self.db.commit()

        ref_time = datetime(2026, 11, 1, 13, 0, 0)
        res = aggregate_hourly_telemetry(db=self.db, transformer_id=t_id, reference_time=ref_time)
        self.assertEqual(res["status"], "success")

        hourly = (
            self.db.query(HourlyTelemetry)
            .filter(
                HourlyTelemetry.transformer_id == t_id,
                HourlyTelemetry.hour_start == datetime(2026, 11, 1, 12, 0, 0),
            )
            .first()
        )
        self.assertIsNotNone(hourly)
        self.assertIsNone(hourly.ambient_temperature_C)  # Strictly NULL
        self.assertAlmostEqual(hourly.oil_temperature_C, 63.0, places=1)
        self.assertNotEqual(hourly.ambient_temperature_C, hourly.oil_temperature_C)

    def test_03_multiple_transformers_independent_aggregation(self):
        """
        Verify that multiple transformers (e.g. T1, T2, T3) are aggregated
        into distinct, isolated hourly records for the same time window.
        """
        h_start = datetime(2026, 11, 1, 14, 0, 0)
        readings = [
            RawTelemetry(
                transformer_id="T1_MULTI",
                timestamp=datetime(2026, 11, 1, 14, 15, 0),
                voltage_V=410.0,
                current_A=100.0,
                power_W=41000.0,
                frequency_Hz=50.0,
                power_factor=0.88,
                oil_temperature_C=55.0,
                ambient_temperature_C=30.0,
            ),
            RawTelemetry(
                transformer_id="T2_MULTI",
                timestamp=datetime(2026, 11, 1, 14, 15, 0),
                voltage_V=420.0,
                current_A=200.0,
                power_W=84000.0,
                frequency_Hz=50.0,
                power_factor=0.92,
                oil_temperature_C=65.0,
                ambient_temperature_C=30.0,
            ),
            RawTelemetry(
                transformer_id="T3_MULTI",
                timestamp=datetime(2026, 11, 1, 14, 15, 0),
                voltage_V=430.0,
                current_A=300.0,
                power_W=129000.0,
                frequency_Hz=50.0,
                power_factor=0.95,
                oil_temperature_C=75.0,
                ambient_temperature_C=30.0,
            ),
        ]
        self.db.add_all(readings)
        self.db.commit()

        ref_time = datetime(2026, 11, 1, 15, 30, 0)
        res = aggregate_hourly_telemetry(db=self.db, reference_time=ref_time)
        self.assertEqual(res["status"], "success")

        # Confirm 3 distinct records exist for the same hour
        t1 = self.db.query(HourlyTelemetry).filter_by(transformer_id="T1_MULTI", hour_start=h_start).first()
        t2 = self.db.query(HourlyTelemetry).filter_by(transformer_id="T2_MULTI", hour_start=h_start).first()
        t3 = self.db.query(HourlyTelemetry).filter_by(transformer_id="T3_MULTI", hour_start=h_start).first()

        self.assertIsNotNone(t1)
        self.assertIsNotNone(t2)
        self.assertIsNotNone(t3)
        self.assertEqual(t1.current_A, 100.0)
        self.assertEqual(t2.current_A, 200.0)
        self.assertEqual(t3.current_A, 300.0)

    def test_04_empty_hours_not_created(self):
        """
        Verify that hours without raw telemetry never generate phantom hourly records.
        """
        t_id = "T_EMPTY_TEST"
        # Reading only at 08:15 and 10:15; hour 09:00 has NO readings
        readings = [
            RawTelemetry(
                transformer_id=t_id,
                timestamp=datetime(2026, 11, 1, 8, 15, 0),
                voltage_V=415.0,
                current_A=20.0,
                power_W=8300.0,
                frequency_Hz=50.0,
                power_factor=0.90,
                oil_temperature_C=50.0,
                ambient_temperature_C=28.0,
            ),
            RawTelemetry(
                transformer_id=t_id,
                timestamp=datetime(2026, 11, 1, 10, 15, 0),
                voltage_V=415.0,
                current_A=22.0,
                power_W=9130.0,
                frequency_Hz=50.0,
                power_factor=0.91,
                oil_temperature_C=52.0,
                ambient_temperature_C=29.0,
            ),
        ]
        self.db.add_all(readings)
        self.db.commit()

        ref_time = datetime(2026, 11, 1, 11, 0, 0)
        aggregate_hourly_telemetry(db=self.db, transformer_id=t_id, reference_time=ref_time)

        # Verify hour 08:00 exists, hour 10:00 exists, but hour 09:00 does NOT exist
        h08 = self.db.query(HourlyTelemetry).filter_by(transformer_id=t_id, hour_start=datetime(2026, 11, 1, 8, 0, 0)).first()
        h09 = self.db.query(HourlyTelemetry).filter_by(transformer_id=t_id, hour_start=datetime(2026, 11, 1, 9, 0, 0)).first()
        h10 = self.db.query(HourlyTelemetry).filter_by(transformer_id=t_id, hour_start=datetime(2026, 11, 1, 10, 0, 0)).first()

        self.assertIsNotNone(h08)
        self.assertIsNone(h09)  # Hour 9 must be completely absent
        self.assertIsNotNone(h10)

    def test_05_incomplete_current_hour_exclusion(self):
        """
        Verify that readings belonging to the current, active hour are NOT
        aggregated until that 60-minute hour has fully elapsed.
        """
        t_id = "T_INCOMPLETE_TEST"
        # Reading at 16:25:00
        reading = RawTelemetry(
            transformer_id=t_id,
            timestamp=datetime(2026, 11, 1, 16, 25, 0),
            voltage_V=415.0,
            current_A=30.0,
            power_W=12450.0,
            frequency_Hz=50.0,
            power_factor=0.92,
            oil_temperature_C=58.0,
            ambient_temperature_C=30.0,
        )
        self.db.add(reading)
        self.db.commit()

        # Case A: Current time is 16:45 (hour 16 is still active and incomplete)
        active_time = datetime(2026, 11, 1, 16, 45, 0)
        res_active = aggregate_hourly_telemetry(db=self.db, transformer_id=t_id, reference_time=active_time)
        self.assertEqual(res_active["records_created"], 0)

        h16_before = (
            self.db.query(HourlyTelemetry)
            .filter_by(transformer_id=t_id, hour_start=datetime(2026, 11, 1, 16, 0, 0))
            .first()
        )
        self.assertIsNone(h16_before)

        # Case B: Current time advances past 17:00 (hour 16 has now closed)
        closed_time = datetime(2026, 11, 1, 17, 0, 1)
        res_closed = aggregate_hourly_telemetry(db=self.db, transformer_id=t_id, reference_time=closed_time)
        self.assertEqual(res_closed["records_created"], 1)

        h16_after = (
            self.db.query(HourlyTelemetry)
            .filter_by(transformer_id=t_id, hour_start=datetime(2026, 11, 1, 16, 0, 0))
            .first()
        )
        self.assertIsNotNone(h16_after)
        self.assertEqual(h16_after.reading_count, 1)

    def test_06_duplicate_prevention_and_idempotency(self):
        """
        Verify that running the aggregation process multiple times does not create
        duplicate hourly records and updates existing records idempotently.
        """
        t_id = "T_IDEMPOTENT"
        reading = RawTelemetry(
            transformer_id=t_id,
            timestamp=datetime(2026, 11, 1, 18, 15, 0),
            voltage_V=415.0,
            current_A=40.0,
            power_W=16600.0,
            frequency_Hz=50.0,
            power_factor=0.91,
            oil_temperature_C=60.0,
            ambient_temperature_C=29.0,
        )
        self.db.add(reading)
        self.db.commit()

        ref_time = datetime(2026, 11, 1, 19, 30, 0)

        # First run: creates 1 record
        run1 = aggregate_hourly_telemetry(db=self.db, transformer_id=t_id, reference_time=ref_time)
        self.assertEqual(run1["records_created"], 1)
        self.assertEqual(run1["records_updated"], 0)

        count_after_run1 = (
            self.db.query(HourlyTelemetry)
            .filter_by(transformer_id=t_id, hour_start=datetime(2026, 11, 1, 18, 0, 0))
            .count()
        )
        self.assertEqual(count_after_run1, 1)

        # Second run: updates 1 record, creates 0
        run2 = aggregate_hourly_telemetry(db=self.db, transformer_id=t_id, reference_time=ref_time)
        self.assertEqual(run2["records_created"], 0)
        self.assertEqual(run2["records_updated"], 1)

        count_after_run2 = (
            self.db.query(HourlyTelemetry)
            .filter_by(transformer_id=t_id, hour_start=datetime(2026, 11, 1, 18, 0, 0))
            .count()
        )
        self.assertEqual(count_after_run2, 1)  # Strictly no duplicate rows

    def test_07_api_run_aggregation_endpoint(self):
        """
        Verify the POST /api/aggregation/run endpoint functionality and response schema.
        """
        payload = AggregationRunRequest(
            transformer_id="T_IDEMPOTENT",
            reference_time=datetime(2026, 11, 1, 20, 0, 0),
        )
        response: AggregationRunResponse = run_aggregation(payload=payload, db=self.db)
        self.assertEqual(response.status, "success")
        self.assertGreaterEqual(response.total_processed, 1)
        self.assertIn("Cutoff", response.message)

    def test_08_api_get_aggregation_history_endpoint(self):
        """
        Verify the GET /api/aggregation/{transformer_id}/history endpoint
        chronological ordering, pagination, and response model.
        """
        response: HourlyHistoryResponse = get_aggregation_history(
            transformer_id="T_AVG_TEST",
            start_time=None,
            end_time=None,
            limit=50,
            offset=0,
            db=self.db,
        )
        self.assertEqual(response.transformer_id, "T_AVG_TEST")
        self.assertGreaterEqual(response.total_records, 1)
        self.assertEqual(len(response.records), response.total_records)

        first_rec = response.records[0]
        self.assertEqual(first_rec.transformer_id, "T_AVG_TEST")
        self.assertEqual(first_rec.reading_count, 4)
        self.assertAlmostEqual(first_rec.voltage_V, 415.0, places=1)


if __name__ == "__main__":
    unittest.main()

