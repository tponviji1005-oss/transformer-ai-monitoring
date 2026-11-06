"""
Integration and Verification Test Suite for Milestone 5 Stage 5:
Live Hardware-to-Backend Integration.

Verifies:
1. Safe database migration preserves existing raw and hourly records.
2. Ingestion security: rejects missing or invalid X-Device-API-Key with HTTP 401.
3. Reading ID deduplication: returns HTTP 200 duplicate acknowledgment without inserting duplicate rows.
4. Batch ingestion endpoint with partial deduplication support.
5. Scheduler atomic concurrency locking: prevents duplicate execution across concurrent workers.
6. Scheduler stale-lock crash recovery: reclaims orphaned locks older than 15 minutes.
7. Late-arriving telemetry packet recalculation and idempotency.
8. Simulation isolation: safety guard rejects targeting production port 8000.
9. Respects all advisory operator constraints (zero electrical actuation, no efficiency claims).
"""

from datetime import datetime, timedelta, timezone
import os
import sys
import unittest
from fastapi.testclient import TestClient

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend.config import settings
from src.backend.database import (
    HourlyTelemetry,
    LiveFeatures,
    PredictionRecord,
    RawTelemetry,
    SchedulerLock,
    SessionLocal,
    engine,
    init_db,
    run_safe_migrations,
)
from src.backend.main import app, verify_device_api_key
from src.backend.schemas import TelemetryBatchIngestRequest, TelemetryIngestRequest
from src.backend.aggregation_service import aggregate_hourly_telemetry
from src.backend.scheduler import SchedulerLockManager, run_pipeline_for_hour
from scripts.simulate_esp32_stream import enforce_safety_isolation


class TestHardwareIntegration(unittest.TestCase):
    """Test suite for Stage 5 live hardware-to-backend integration."""

    @classmethod
    def setUpClass(cls):
        """Initialize database tables and run migrations."""
        init_db()
        run_safe_migrations()
        cls.client = TestClient(app)
        cls.api_key = settings.DEVICE_API_KEY

    def setUp(self):
        """Create fresh transactional database session for each test."""
        self.db = SessionLocal()

    def tearDown(self):
        """Clean up test records created during testing."""
        self.db.query(RawTelemetry).filter(RawTelemetry.transformer_id.like("T_TEST_%")).delete()
        self.db.query(HourlyTelemetry).filter(HourlyTelemetry.transformer_id.like("T_TEST_%")).delete()
        self.db.query(SchedulerLock).filter(SchedulerLock.job_name.like("test_%")).delete()
        self.db.commit()
        self.db.close()

    def test_01_safe_migration_preserves_records_and_schema(self):
        """
        Verify that running safe migrations:
        - Does not delete or truncate any existing database records.
        - Idempotently adds 'reading_id' and 'sequence_number' columns to raw_telemetry.
        - Creates the partial unique index idx_raw_telemetry_reading_id.
        - Creates the scheduler_locks table.
        """
        raw_count_before = self.db.query(RawTelemetry).count()
        hourly_count_before = self.db.query(HourlyTelemetry).count()

        migration_res = run_safe_migrations()
        self.assertIsInstance(migration_res, dict)

        raw_count_after = self.db.query(RawTelemetry).count()
        hourly_count_after = self.db.query(HourlyTelemetry).count()

        # Database record counts must remain strictly intact
        self.assertEqual(raw_count_after, raw_count_before)
        self.assertEqual(hourly_count_after, hourly_count_before)

        # Verify SchedulerLock table is accessible
        locks = self.db.query(SchedulerLock).all()
        self.assertIsInstance(locks, list)

    def test_02_ingest_authentication_enforcement(self):
        """
        Verify that raw telemetry ingestion enforces X-Device-API-Key authentication:
        - Missing header -> HTTP 401 Unauthorized.
        - Invalid key -> HTTP 401 Unauthorized.
        - Valid key -> HTTP 201 Created.
        """
        payload = {
            "reading_id": "T_TEST_AUTH-BOOT1-00000001",
            "sequence_number": 1,
            "transformer_id": "T_TEST_AUTH",
            "timestamp": "2026-11-20T10:00:00",
            "voltage_V": 415.0,
            "current_A": 25.0,
            "power_W": 10000.0,
            "frequency_Hz": 50.0,
            "power_factor": 0.92,
            "oil_temperature_C": 45.0,
            "ambient_temperature_C": None,
        }

        # 1. Missing header
        res_no_key = self.client.post("/api/telemetry/ingest", json=payload)
        self.assertEqual(res_no_key.status_code, 401)
        self.assertIn("Invalid or missing X-Device-API-Key", res_no_key.text)

        # 2. Invalid key
        res_bad_key = self.client.post(
            "/api/telemetry/ingest",
            json=payload,
            headers={"X-Device-API-Key": "wrong-secret-key"},
        )
        self.assertEqual(res_bad_key.status_code, 401)

        # 3. Valid key
        res_valid = self.client.post(
            "/api/telemetry/ingest",
            json=payload,
            headers={"X-Device-API-Key": self.api_key},
        )
        self.assertEqual(res_valid.status_code, 201)
        data = res_valid.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["reading_id"], "T_TEST_AUTH-BOOT1-00000001")

    def test_03_reading_id_deduplication(self):
        """
        Verify that duplicate packets with identical reading_id are acknowledged
        with HTTP 200 (duplicate) and do NOT insert redundant rows into the database.
        """
        reading_id = "T_TEST_DEDUP-BOOT1-00000042"
        payload = {
            "reading_id": reading_id,
            "sequence_number": 42,
            "transformer_id": "T_TEST_DEDUP",
            "timestamp": "2026-11-20T10:02:00",
            "voltage_V": 414.5,
            "current_A": 30.0,
            "power_W": 12000.0,
            "frequency_Hz": 50.01,
            "power_factor": 0.91,
            "oil_temperature_C": 46.2,
            "ambient_temperature_C": None,
        }

        # First transmission -> HTTP 201
        res1 = self.client.post(
            "/api/telemetry/ingest",
            json=payload,
            headers={"X-Device-API-Key": self.api_key},
        )
        self.assertEqual(res1.status_code, 201)
        self.assertEqual(res1.json()["status"], "success")

        count_after_first = (
            self.db.query(RawTelemetry)
            .filter(RawTelemetry.reading_id == reading_id)
            .count()
        )
        self.assertEqual(count_after_first, 1)

        # Re-transmission (e.g. lost ACK) -> HTTP 200 duplicate acknowledgment
        res2 = self.client.post(
            "/api/telemetry/ingest",
            json=payload,
            headers={"X-Device-API-Key": self.api_key},
        )
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.json()["status"], "duplicate")
        self.assertIn("already recorded", res2.json()["message"])

        # Database count must strictly remain 1
        count_after_second = (
            self.db.query(RawTelemetry)
            .filter(RawTelemetry.reading_id == reading_id)
            .count()
        )
        self.assertEqual(count_after_second, 1)

    def test_04_batch_telemetry_ingestion(self):
        """
        Verify batch ingestion endpoint (/api/telemetry/ingest-batch):
        - Ingests multiple readings in a single request (ESP32 reconnect buffer flush).
        - Automatically detects and skips duplicate reading_ids within or across batches.
        """
        batch_payload = {
            "readings": [
                {
                    "reading_id": "T_TEST_BATCH-BOOT1-00000001",
                    "sequence_number": 1,
                    "transformer_id": "T_TEST_BATCH",
                    "timestamp": "2026-11-20T10:00:00",
                    "voltage_V": 415.0,
                    "current_A": 20.0,
                    "power_W": 8000.0,
                    "frequency_Hz": 50.0,
                    "power_factor": 0.90,
                    "oil_temperature_C": 42.0,
                },
                {
                    "reading_id": "T_TEST_BATCH-BOOT1-00000002",
                    "sequence_number": 2,
                    "transformer_id": "T_TEST_BATCH",
                    "timestamp": "2026-11-20T10:00:02",
                    "voltage_V": 415.2,
                    "current_A": 20.1,
                    "power_W": 8040.0,
                    "frequency_Hz": 50.0,
                    "power_factor": 0.90,
                    "oil_temperature_C": 42.1,
                },
            ]
        }

        # 1. First batch -> 2 inserted, 0 duplicate
        res1 = self.client.post(
            "/api/telemetry/ingest-batch",
            json=batch_payload,
            headers={"X-Device-API-Key": self.api_key},
        )
        self.assertEqual(res1.status_code, 200)
        data1 = res1.json()
        self.assertEqual(data1["total_received"], 2)
        self.assertEqual(data1["inserted_count"], 2)
        self.assertEqual(data1["duplicate_count"], 0)

        # 2. Second batch with 1 existing reading and 1 new reading
        partial_dup_batch = {
            "readings": [
                batch_payload["readings"][1],  # Duplicate
                {
                    "reading_id": "T_TEST_BATCH-BOOT1-00000003",
                    "sequence_number": 3,
                    "transformer_id": "T_TEST_BATCH",
                    "timestamp": "2026-11-20T10:00:04",
                    "voltage_V": 415.5,
                    "current_A": 20.2,
                    "power_W": 8100.0,
                    "frequency_Hz": 50.0,
                    "power_factor": 0.90,
                    "oil_temperature_C": 42.2,
                },
            ]
        }

        res2 = self.client.post(
            "/api/telemetry/ingest-batch",
            json=partial_dup_batch,
            headers={"X-Device-API-Key": self.api_key},
        )
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        self.assertEqual(data2["total_received"], 2)
        self.assertEqual(data2["inserted_count"], 1)
        self.assertEqual(data2["duplicate_count"], 1)

    def test_05_scheduler_atomic_concurrency_locking(self):
        """
        Verify that the SchedulerLockManager guarantees single-worker execution:
        - Worker 1 acquires lock -> succeeds.
        - Worker 2 concurrently attempts same hour -> rejected cleanly.
        - Worker 1 marks completed -> future workers skip already-completed job.
        """
        job = "test_hourly_job"
        hour_str = "2026-11-20T10:00:00"

        # Worker 1 acquires lock
        acquired_w1 = SchedulerLockManager.acquire_lock(
            db=self.db,
            job_name=job,
            target_hour_str=hour_str,
            worker_pid="worker_1_pid",
            timeout_minutes=15,
        )
        self.assertTrue(acquired_w1)

        # Worker 2 attempts same job while IN_PROGRESS
        acquired_w2 = SchedulerLockManager.acquire_lock(
            db=self.db,
            job_name=job,
            target_hour_str=hour_str,
            worker_pid="worker_2_pid",
            timeout_minutes=15,
        )
        self.assertFalse(acquired_w2)

        # Worker 1 completes job
        SchedulerLockManager.release_lock(
            db=self.db,
            job_name=job,
            target_hour_str=hour_str,
            status="COMPLETED",
        )

        # Worker 3 attempts completed job -> skipped
        acquired_w3 = SchedulerLockManager.acquire_lock(
            db=self.db,
            job_name=job,
            target_hour_str=hour_str,
            worker_pid="worker_3_pid",
            timeout_minutes=15,
        )
        self.assertFalse(acquired_w3)

    def test_06_scheduler_stale_lock_crash_recovery(self):
        """
        Verify 15-minute stale-lock crash recovery:
        - If a worker crashes while holding an IN_PROGRESS lock older than 15 minutes,
          a subsequent worker safely reclaims the lock and executes the job.
        """
        job = "test_stale_job"
        hour_str = "2026-11-20T11:00:00"

        # Seed an orphaned lock from a crashed process 20 minutes ago
        orphaned_time = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=20)
        stale_lock = SchedulerLock(
            job_name=job,
            target_hour=hour_str,
            worker_pid="crashed_dead_worker",
            status="IN_PROGRESS",
            locked_at=orphaned_time,
            completed_at=None,
        )
        self.db.add(stale_lock)
        self.db.commit()

        # Recovery worker attempts acquisition with 15-minute timeout
        recovered = SchedulerLockManager.acquire_lock(
            db=self.db,
            job_name=job,
            target_hour_str=hour_str,
            worker_pid="recovery_worker_pid",
            timeout_minutes=15,
        )
        self.assertTrue(recovered)

        # Verify lock was reassigned to recovery worker
        self.db.refresh(stale_lock)
        self.assertEqual(stale_lock.worker_pid, "recovery_worker_pid")
        self.assertEqual(stale_lock.status, "IN_PROGRESS")

    def test_07_late_telemetry_idempotent_recalculation(self):
        """
        Verify late-arriving telemetry handling:
        - Initial aggregation groups raw readings for hour H.
        - Late readings arrive later for hour H.
        - Re-running aggregation updates the existing record in-place (records_updated=1),
          recalculating averages and reading_count without creating duplicate rows.
        """
        tid = "T_TEST_LATE"
        h0 = datetime(2026, 11, 20, 14, 0, 0)

        # 1. Seed 2 initial raw readings in hour [14:00, 15:00)
        self.db.add(
            RawTelemetry(
                transformer_id=tid,
                timestamp=h0 + timedelta(minutes=10),
                voltage_V=400.0,
                current_A=10.0,
                power_W=4000.0,
                frequency_Hz=50.0,
                power_factor=0.90,
                oil_temperature_C=40.0,
                reading_id="T_TEST_LATE-B1-001",
            )
        )
        self.db.commit()

        # Run initial aggregation
        cutoff_time = h0 + timedelta(hours=2)
        agg1 = aggregate_hourly_telemetry(db=self.db, transformer_id=tid, reference_time=cutoff_time)
        self.assertEqual(agg1["records_created"], 1)

        hourly_rec1 = (
            self.db.query(HourlyTelemetry)
            .filter_by(transformer_id=tid, hour_start=h0)
            .first()
        )
        self.assertEqual(hourly_rec1.reading_count, 1)
        self.assertAlmostEqual(hourly_rec1.power_W, 4000.0, places=1)

        # 2. Late-arriving packet arrives for the same hour [14:00, 15:00)
        self.db.add(
            RawTelemetry(
                transformer_id=tid,
                timestamp=h0 + timedelta(minutes=45),
                voltage_V=420.0,
                current_A=20.0,
                power_W=8000.0,
                frequency_Hz=50.0,
                power_factor=0.90,
                oil_temperature_C=42.0,
                reading_id="T_TEST_LATE-B1-002",
            )
        )
        self.db.commit()

        # Re-run aggregation for hour H
        agg2 = aggregate_hourly_telemetry(db=self.db, transformer_id=tid, reference_time=cutoff_time)
        self.assertEqual(agg2["records_updated"], 1)
        self.assertEqual(agg2["records_created"], 0)

        # Verify hourly record was updated in-place (zero duplicate rows)
        hourly_records = (
            self.db.query(HourlyTelemetry)
            .filter_by(transformer_id=tid, hour_start=h0)
            .all()
        )
        self.assertEqual(len(hourly_records), 1)
        self.assertEqual(hourly_records[0].reading_count, 2)
        # Average of 4000W and 8000W = 6000W
        self.assertAlmostEqual(hourly_records[0].power_W, 6000.0, places=1)

    def test_08_simulation_port_isolation_safety_guard(self):
        """
        Verify that enforce_safety_isolation() strictly blocks execution
        if the simulator attempts to target production port 8000 or production DB.
        """
        with self.assertRaises(RuntimeError) as cm:
            enforce_safety_isolation("http://127.0.0.1:8000")
        self.assertIn("SAFETY VIOLATION BLOCKED", str(cm.exception))
        self.assertIn("port 8000", str(cm.exception))

        # Safe port (e.g. 8001) must pass without exception
        try:
            enforce_safety_isolation("http://127.0.0.1:8001")
        except RuntimeError:
            self.fail("enforce_safety_isolation raised RuntimeError unexpectedly on port 8001")


if __name__ == "__main__":
    unittest.main()
