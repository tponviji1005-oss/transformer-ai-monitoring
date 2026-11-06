"""
Unit and Integration Tests for Scheduler Concurrency Safety, Fencing Tokens, and Lease Renewal.
Verifies:
1. Simultaneous worker exclusion (only 1 worker acquires lock).
2. Stale-lock crash recovery increments fencing token.
3. Stale worker is strictly fenced off from writing results, renewing lease, or releasing lock.
4. Active lease renewal / heartbeat prevents premature lock stealing.
5. run_pipeline_for_hour cleanly aborts with 'lock_lost' when superseded.
"""

from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch

from src.backend.database import SessionLocal, SchedulerLock
from src.backend.scheduler import (
    LockLease,
    SchedulerLockManager,
    SchedulerLockLostError,
    run_pipeline_for_hour,
)


class TestSchedulerConcurrency(unittest.TestCase):
    """Test suite verifying distributed lock concurrency and fencing tokens."""

    JOB_NAME = "test_concurrency_job"
    TARGET_HOUR = "2026-11-05T08:00:00"

    @classmethod
    def setUpClass(cls):
        from src.backend.database import init_db
        init_db()

    @classmethod
    def tearDownClass(cls):
        import shutil, os
        from src.backend.database import engine, init_db
        engine.dispose()
        backup = os.path.join("data", "backups", "transformer_monitoring_pre_stage5_fixes.db")
        target = os.path.join("data", "transformer_monitoring.db")
        if os.path.exists(backup):
            shutil.copy2(backup, target)
            init_db()

    def setUp(self):
        self.db = SessionLocal()
        # Clean up any test locks before each test
        self.db.query(SchedulerLock).filter(
            SchedulerLock.job_name == self.JOB_NAME
        ).delete()
        self.db.commit()

    def tearDown(self):
        # Clean up after each test
        self.db.query(SchedulerLock).filter(
            SchedulerLock.job_name == self.JOB_NAME
        ).delete()
        self.db.commit()
        self.db.close()

    def test_simultaneous_workers_conflict(self):
        """Worker 1 acquires lock; Worker 2 is rejected."""
        lease1 = SchedulerLockManager.acquire_lock(
            db=self.db,
            job_name=self.JOB_NAME,
            target_hour_str=self.TARGET_HOUR,
            worker_pid="worker_alpha",
            timeout_minutes=15,
        )
        self.assertTrue(lease1.acquired)
        self.assertEqual(lease1.fencing_token, 1)
        self.assertEqual(lease1.worker_pid, "worker_alpha")

        # Worker 2 attempts same lock
        lease2 = SchedulerLockManager.acquire_lock(
            db=self.db,
            job_name=self.JOB_NAME,
            target_hour_str=self.TARGET_HOUR,
            worker_pid="worker_beta",
            timeout_minutes=15,
        )
        self.assertFalse(lease2.acquired)

    def test_stale_lock_recovery_increments_fencing_token(self):
        """When Worker 1 crashes/times out, Worker 2 reclaims with incremented token."""
        lease1 = SchedulerLockManager.acquire_lock(
            db=self.db,
            job_name=self.JOB_NAME,
            target_hour_str=self.TARGET_HOUR,
            worker_pid="worker_alpha",
            timeout_minutes=15,
        )
        self.assertEqual(lease1.fencing_token, 1)

        # Simulate 20 minutes elapsed since Worker 1 acquired lock
        lock_row = (
            self.db.query(SchedulerLock)
            .filter(
                SchedulerLock.job_name == self.JOB_NAME,
                SchedulerLock.target_hour == self.TARGET_HOUR,
            )
            .first()
        )
        lock_row.locked_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=20)
        self.db.commit()

        # Worker 2 now reclaims stale lock
        lease2 = SchedulerLockManager.acquire_lock(
            db=self.db,
            job_name=self.JOB_NAME,
            target_hour_str=self.TARGET_HOUR,
            worker_pid="worker_beta",
            timeout_minutes=15,
        )
        self.assertTrue(lease2.acquired)
        self.assertEqual(lease2.fencing_token, 2)
        self.assertEqual(lease2.worker_pid, "worker_beta")

    def test_stale_worker_fenced_off(self):
        """A stale worker cannot renew lease, verify ownership, or release the stolen lock."""
        # Worker 1 acquires
        lease1 = SchedulerLockManager.acquire_lock(
            db=self.db,
            job_name=self.JOB_NAME,
            target_hour_str=self.TARGET_HOUR,
            worker_pid="worker_alpha",
            timeout_minutes=15,
        )

        # Stale recovery: Worker 2 takes over
        lock_row = (
            self.db.query(SchedulerLock)
            .filter(
                SchedulerLock.job_name == self.JOB_NAME,
                SchedulerLock.target_hour == self.TARGET_HOUR,
            )
            .first()
        )
        lock_row.locked_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=20)
        self.db.commit()

        lease2 = SchedulerLockManager.acquire_lock(
            db=self.db,
            job_name=self.JOB_NAME,
            target_hour_str=self.TARGET_HOUR,
            worker_pid="worker_beta",
            timeout_minutes=15,
        )
        self.assertTrue(lease2.acquired)
        self.assertEqual(lease2.fencing_token, 2)

        # Now Worker 1 wakes up and tries operations using lease1 (token 1)
        # 1. verify_ownership must fail
        self.assertFalse(SchedulerLockManager.verify_ownership(self.db, lease1))

        # 2. renew_lease must fail
        self.assertFalse(SchedulerLockManager.renew_lease(self.db, lease1))

        # 3. release_lock must fail and not disturb Worker 2's status
        released = SchedulerLockManager.release_lock(self.db, lease1, status="COMPLETED")
        self.assertFalse(released)

        # Confirm lock still belongs to Worker 2 and is still IN_PROGRESS
        self.db.refresh(lock_row)
        self.assertEqual(lock_row.worker_pid, "worker_beta")
        self.assertEqual(lock_row.status, "IN_PROGRESS")
        self.assertEqual(lock_row.fencing_token, 2)

        # Worker 2 can legitimately verify and complete
        self.assertTrue(SchedulerLockManager.verify_ownership(self.db, lease2))
        self.assertTrue(SchedulerLockManager.release_lock(self.db, lease2, status="COMPLETED"))
        self.db.refresh(lock_row)
        self.assertEqual(lock_row.status, "COMPLETED")

    def test_heartbeat_lease_renewal(self):
        """Active heartbeats extend locked_at and prevent lock stealing."""
        lease = SchedulerLockManager.acquire_lock(
            db=self.db,
            job_name=self.JOB_NAME,
            target_hour_str=self.TARGET_HOUR,
            worker_pid="worker_heartbeat",
            timeout_minutes=15,
        )
        lock_row = (
            self.db.query(SchedulerLock)
            .filter(
                SchedulerLock.job_name == self.JOB_NAME,
                SchedulerLock.target_hour == self.TARGET_HOUR,
            )
            .first()
        )
        old_time = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=10)
        lock_row.locked_at = old_time
        self.db.commit()

        # Send heartbeat
        renewed = SchedulerLockManager.renew_lease(self.db, lease)
        self.assertTrue(renewed)

        self.db.refresh(lock_row)
        self.assertGreater(lock_row.locked_at, old_time)

    def test_pipeline_aborts_cleanly_on_lock_loss(self):
        """run_pipeline_for_hour detects lock loss and returns 'lock_lost' without crashing."""
        th = datetime.fromisoformat(self.TARGET_HOUR)

        with patch("src.backend.scheduler.aggregate_hourly_telemetry", return_value={"status": "mocked"}), \
             patch.object(SchedulerLockManager, "verify_ownership", side_effect=[True, False]):
            res = run_pipeline_for_hour(
                db=self.db,
                target_hour=th,
                job_name=self.JOB_NAME,
                worker_pid="worker_aborted",
            )
            self.assertEqual(res["status"], "lock_lost")
            self.assertIn("Scheduler lock lost", res["error"])


if __name__ == "__main__":
    unittest.main()
