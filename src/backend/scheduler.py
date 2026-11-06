"""
Background Hourly Scheduler & Pipeline Execution Engine.
Manages atomic concurrency locking, stale-lock crash recovery, and end-to-end
execution of aggregation -> feature generation -> inference pipeline.
"""

from datetime import datetime, timedelta, timezone
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from src.backend.config import settings
from src.backend.database import (
    HourlyTelemetry,
    LiveFeatures,
    PredictionRecord,
    SchedulerLock,
    SessionLocal,
)
from src.backend.aggregation_service import aggregate_hourly_telemetry
from src.backend.feature_service import (
    generate_live_features,
    get_feature_readiness,
    identify_affected_downstream_timestamps,
)
from src.backend.inference_service import execute_live_inference

logger = logging.getLogger("transformer_scheduler")


class SchedulerLockLostError(RuntimeError):
    """Raised when a scheduler worker detects it has lost lock ownership or its lease was fenced."""
    pass


class LockLease:
    """
    Represents an atomic distributed lock lease with an explicit fencing token.
    Fencing token monotonically increases whenever a lock is claimed or reclaimed,
    ensuring stale workers are strictly fenced off from committing state.
    """

    def __init__(
        self,
        acquired: bool,
        job_name: str,
        target_hour: str,
        worker_pid: str,
        fencing_token: int = 0,
    ):
        self.acquired = acquired
        self.job_name = job_name
        self.target_hour = target_hour
        self.worker_pid = worker_pid
        self.fencing_token = fencing_token

    def __bool__(self) -> bool:
        return self.acquired

    def __repr__(self) -> str:
        return (
            f"<LockLease acquired={self.acquired} job={self.job_name} "
            f"hour={self.target_hour} worker={self.worker_pid} token={self.fencing_token}>"
        )


class SchedulerLockManager:
    """
    Manages atomic job acquisition and crash recovery using the scheduler_locks SQLite table.
    Enforces concurrency protection and stale-worker fencing via monotonically increasing tokens.
    """

    @staticmethod
    def acquire_lock(
        db: Session,
        job_name: str,
        target_hour_str: str,
        worker_pid: Optional[str] = None,
        timeout_minutes: int = 15,
    ) -> LockLease:
        """
        Attempts to acquire an atomic execution lock for (job_name, target_hour_str).
        
        Returns:
            LockLease with acquired=True and fencing_token if acquired.
            LockLease with acquired=False if held by active worker or already completed.
        """
        current_pid = worker_pid or f"pid_{os.getpid()}_{threading.get_ident()}"
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        lock = (
            db.query(SchedulerLock)
            .filter(
                SchedulerLock.job_name == job_name,
                SchedulerLock.target_hour == target_hour_str,
            )
            .with_for_update() if hasattr(db.query(SchedulerLock), "with_for_update")
            else db.query(SchedulerLock).filter(
                SchedulerLock.job_name == job_name,
                SchedulerLock.target_hour == target_hour_str,
            )
        ).first()

        if lock is None:
            # First worker to claim this hour -> fencing_token = 1
            new_lock = SchedulerLock(
                job_name=job_name,
                target_hour=target_hour_str,
                worker_pid=current_pid,
                status="IN_PROGRESS",
                fencing_token=1,
                locked_at=now,
                completed_at=None,
            )
            db.add(new_lock)
            try:
                db.commit()
                logger.info(f"Lock acquired for {job_name} at {target_hour_str} by {current_pid} (token: 1)")
                return LockLease(True, job_name, target_hour_str, current_pid, 1)
            except Exception as exc:
                db.rollback()
                logger.warning(f"Failed to insert lock for {job_name} at {target_hour_str}: {exc}")
                return LockLease(False, job_name, target_hour_str, current_pid, 0)

        if lock.status == "COMPLETED":
            logger.debug(f"Job {job_name} at {target_hour_str} already completed. Skipping.")
            return LockLease(False, job_name, target_hour_str, current_pid, lock.fencing_token)

        if lock.status == "IN_PROGRESS":
            lock_age = now - lock.locked_at
            if lock_age < timedelta(minutes=timeout_minutes):
                logger.debug(
                    f"Job {job_name} at {target_hour_str} is currently IN_PROGRESS by {lock.worker_pid} (age: {lock_age}). Skipping."
                )
                return LockLease(False, job_name, target_hour_str, current_pid, lock.fencing_token)
            else:
                # Stale lock detected (worker crashed or exceeded lease). Reclaim and increment fencing token!
                new_token = (lock.fencing_token or 1) + 1
                logger.warning(
                    f"Stale lock detected for {job_name} at {target_hour_str} (held by {lock.worker_pid} for {lock_age}). "
                    f"Reclaiming lock with new fencing token {new_token}."
                )
                lock.worker_pid = current_pid
                lock.status = "IN_PROGRESS"
                lock.fencing_token = new_token
                lock.locked_at = now
                lock.completed_at = None
                db.commit()
                return LockLease(True, job_name, target_hour_str, current_pid, new_token)

        if lock.status == "FAILED":
            # Reclaim previously failed lock with incremented fencing token
            new_token = (lock.fencing_token or 1) + 1
            logger.info(f"Reclaiming failed lock for {job_name} at {target_hour_str} with token {new_token}")
            lock.worker_pid = current_pid
            lock.status = "IN_PROGRESS"
            lock.fencing_token = new_token
            lock.locked_at = now
            lock.completed_at = None
            db.commit()
            return LockLease(True, job_name, target_hour_str, current_pid, new_token)

        return LockLease(False, job_name, target_hour_str, current_pid, 0)

    @staticmethod
    def verify_ownership(
        db: Session,
        job_or_lease: Any,
        target_hour_str: Optional[str] = None,
        worker_pid: Optional[str] = None,
        fencing_token: Optional[int] = None,
    ) -> bool:
        """
        Verifies that the calling worker still holds valid, unfenced ownership of the lock.
        Returns True if current DB lock matches worker_pid, fencing_token, and status == 'IN_PROGRESS'.
        """
        if isinstance(job_or_lease, LockLease):
            j_name = job_or_lease.job_name
            t_hour = job_or_lease.target_hour
            w_pid = job_or_lease.worker_pid
            f_token = job_or_lease.fencing_token
        else:
            j_name = str(job_or_lease)
            t_hour = str(target_hour_str)
            w_pid = str(worker_pid)
            f_token = fencing_token

        lock = (
            db.query(SchedulerLock)
            .filter(
                SchedulerLock.job_name == j_name,
                SchedulerLock.target_hour == t_hour,
            )
            .first()
        )
        if not lock:
            return False
        if lock.worker_pid != w_pid:
            return False
        if f_token is not None and lock.fencing_token != f_token:
            return False
        if lock.status != "IN_PROGRESS":
            return False
        return True

    @staticmethod
    def renew_lease(
        db: Session,
        job_or_lease: Any,
        target_hour_str: Optional[str] = None,
        worker_pid: Optional[str] = None,
        fencing_token: Optional[int] = None,
    ) -> bool:
        """
        Heartbeat / lease renewal mechanism for long-running pipelines.
        Extends locked_at timestamp if caller is the active owner.
        """
        if isinstance(job_or_lease, LockLease):
            j_name = job_or_lease.job_name
            t_hour = job_or_lease.target_hour
            w_pid = job_or_lease.worker_pid
            f_token = job_or_lease.fencing_token
        else:
            j_name = str(job_or_lease)
            t_hour = str(target_hour_str)
            w_pid = str(worker_pid)
            f_token = fencing_token

        lock = (
            db.query(SchedulerLock)
            .filter(
                SchedulerLock.job_name == j_name,
                SchedulerLock.target_hour == t_hour,
            )
            .first()
        )
        if not lock or lock.worker_pid != w_pid or (f_token is not None and lock.fencing_token != f_token):
            logger.warning(f"Lease renewal rejected for {j_name} at {t_hour}: ownership lost.")
            return False
        if lock.status != "IN_PROGRESS":
            return False

        lock.locked_at = datetime.now(timezone.utc).replace(tzinfo=None)
        db.commit()
        return True

    @staticmethod
    def release_lock(
        db: Session,
        job_or_lease: Any = None,
        target_hour_str: Optional[str] = None,
        status: str = "COMPLETED",
        fencing_token: Optional[int] = None,
        worker_pid: Optional[str] = None,
        job_name: Optional[str] = None,
    ) -> bool:
        """
        Releases lock, transitioning status to COMPLETED or FAILED.
        Strictly checks fencing token and worker_pid to prevent stale workers
        from modifying locks owned by newer workers.
        """
        if job_name is not None:
            j_name = job_name
            t_hour = str(target_hour_str)
            w_pid = worker_pid
            f_token = fencing_token
        elif isinstance(job_or_lease, LockLease):
            j_name = job_or_lease.job_name
            t_hour = job_or_lease.target_hour
            w_pid = job_or_lease.worker_pid
            f_token = job_or_lease.fencing_token
        else:
            j_name = str(job_or_lease)
            t_hour = str(target_hour_str)
            w_pid = worker_pid
            f_token = fencing_token

        lock = (
            db.query(SchedulerLock)
            .filter(
                SchedulerLock.job_name == j_name,
                SchedulerLock.target_hour == t_hour,
            )
            .first()
        )
        if not lock:
            return False

        # If fencing token or worker_pid are provided, verify ownership before modifying status
        if f_token is not None and lock.fencing_token != f_token:
            logger.warning(
                f"Fencing token violation on release: worker {w_pid} (token {f_token}) "
                f"attempted to release lock now held with token {lock.fencing_token}. Rejected."
            )
            return False

        if w_pid is not None and lock.worker_pid != w_pid:
            logger.warning(
                f"Worker mismatch on release: worker {w_pid} attempted to release lock held by {lock.worker_pid}. Rejected."
            )
            return False

        lock.status = status
        if status == "COMPLETED":
            lock.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
        db.commit()
        return True


def run_pipeline_for_hour(
    db: Session,
    target_hour: datetime,
    job_name: str = "hourly_pipeline",
    worker_pid: Optional[str] = None,
    timeout_minutes: int = 15,
) -> Dict[str, Any]:
    """
    Executes the full automated pipeline for a completed 1-hour window:
    1. Acquires atomic lock with crash recovery.
    2. Runs hourly telemetry aggregation.
    3. Runs feature generation for all active transformers.
    4. Runs ML inference & advisory alert generation for all transformers.
    5. Releases lock upon completion.
    
    Idempotent: updates existing records in place if run multiple times.
    """
    # Normalize target hour to start of hour, naive UTC
    target_hour_clean = target_hour.replace(minute=0, second=0, microsecond=0, tzinfo=None)
    target_hour_str = target_hour_clean.isoformat()

    lease = SchedulerLockManager.acquire_lock(
        db=db,
        job_name=job_name,
        target_hour_str=target_hour_str,
        worker_pid=worker_pid,
        timeout_minutes=timeout_minutes,
    )

    if not lease:
        return {
            "status": "skipped",
            "message": f"Lock not acquired for hour {target_hour_str} (in progress or completed)",
            "target_hour": target_hour_str,
        }

    try:
        logger.info(f"Starting hourly pipeline execution for window {target_hour_str} (lease token: {lease.fencing_token})")

        # Verify initial lock ownership before executing aggregation
        if not SchedulerLockManager.verify_ownership(db, lease):
            raise SchedulerLockLostError(
                f"Scheduler lock lost for {target_hour_str} prior to aggregation (token: {lease.fencing_token})."
            )

        # Step 1: Hourly Aggregation for completed hours up to target_hour
        cutoff_ref = target_hour_clean + timedelta(hours=2)
        agg_result = aggregate_hourly_telemetry(db=db, reference_time=cutoff_ref)

        # Heartbeat: renew lease after aggregation step
        SchedulerLockManager.renew_lease(db, lease)

        # Step 2 & 3: Feature Generation & Inference for each transformer
        transformers = ["T1", "T2", "T3"]
        feature_results = {}
        inference_results = {}

        for tid in transformers:
            # Check ownership before each transformer inference pass
            if not SchedulerLockManager.verify_ownership(db, lease):
                raise SchedulerLockLostError(
                    f"Scheduler lock lost for {target_hour_str} during processing of transformer {tid} (token: {lease.fencing_token})."
                )

            # Check if there is hourly data for this transformer
            hourly_record = (
                db.query(HourlyTelemetry)
                .filter(
                    HourlyTelemetry.transformer_id == tid,
                    HourlyTelemetry.hour_start == target_hour_clean,
                )
                .first()
            )

            if hourly_record is not None:
                feat = generate_live_features(db=db, transformer_id=tid, target_hour=target_hour_clean)
                feature_results[tid] = {
                    "readiness_status": feat.readiness_status,
                    "is_ready_for_inference": feat.is_ready_for_inference,
                }

                inf = execute_live_inference(db=db, transformer_id=tid, hour_timestamp=target_hour_clean)
                inference_results[tid] = {
                    "status": inf.status,
                    "alert_level": inf.alert_level,
                    "alert_type": inf.alert_type,
                }
            else:
                feature_results[tid] = {"status": "no_hourly_data"}
                inference_results[tid] = {"status": "no_hourly_data"}

        # Final verification of lock ownership before committing release as COMPLETED
        if not SchedulerLockManager.verify_ownership(db, lease):
            raise SchedulerLockLostError(
                f"Scheduler lock lost for {target_hour_str} before final completion (token: {lease.fencing_token})."
            )

        SchedulerLockManager.release_lock(
            db=db,
            job_or_lease=lease,
            status="COMPLETED",
        )

        logger.info(f"Successfully completed hourly pipeline for window {target_hour_str} (token: {lease.fencing_token})")
        return {
            "status": "success",
            "target_hour": target_hour_str,
            "aggregation": agg_result,
            "features": feature_results,
            "inference": inference_results,
        }

    except SchedulerLockLostError as lerr:
        db.rollback()
        logger.warning(
            f"Hourly pipeline aborted for {target_hour_str}: {lerr}. Changes rolled back to protect active worker."
        )
        return {
            "status": "lock_lost",
            "target_hour": target_hour_str,
            "error": str(lerr),
        }

    except Exception as exc:
        db.rollback()
        logger.error(f"Hourly pipeline failed for window {target_hour_str}: {exc}", exc_info=True)
        SchedulerLockManager.release_lock(
            db=db,
            job_or_lease=lease,
            status="FAILED",
        )
        return {
            "status": "failed",
            "target_hour": target_hour_str,
            "error": str(exc),
        }


def recalculate_downstream_dependencies(
    db: Session,
    transformer_id: str,
    modified_hour: datetime,
) -> Dict[str, Any]:
    """
    Recalculates features, predictions, and alerts in chronological order for all
    timestamps affected by an in-place modification to hour H.
    Derives affected timestamps from the 1h/24h/168h lags and 24h/168h rolling windows.
    Strictly preserves the 39-feature formulas and respects inference readiness checks.
    """
    tid = transformer_id.strip().upper()
    affected_timestamps = identify_affected_downstream_timestamps(db, tid, modified_hour)

    logger.info(
        f"Recalculating downstream dependencies for {tid} starting from {modified_hour.isoformat()} "
        f"({len(affected_timestamps)} affected hours identified)."
    )

    recalculated = []
    for ts in affected_timestamps:
        feat_res = generate_live_features(db=db, transformer_id=tid, target_hour=ts)
        if tid in ("T1", "T2", "T3"):
            inf_res = execute_live_inference(db=db, transformer_id=tid, hour_timestamp=ts)
            recalculated.append({
                "timestamp": ts.isoformat(),
                "feature_readiness": feat_res.readiness_status,
                "inference_status": inf_res.status,
                "alert_level": inf_res.alert_level,
            })
        else:
            recalculated.append({
                "timestamp": ts.isoformat(),
                "feature_readiness": feat_res.readiness_status,
                "inference_status": "skipped_non_ml_unit",
                "alert_level": None,
            })

    return {
        "transformer_id": tid,
        "modified_hour": modified_hour.isoformat(),
        "total_recalculated": len(recalculated),
        "recalculated_timestamps": recalculated,
    }


class BackgroundScheduler:
    """
    Lightweight background worker that checks once per minute for completed hours
    past the configured grace period (e.g. 5 minutes after hour end) and executes
    the hourly pipeline with atomic locking.
    """

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Starts the scheduler thread if enabled."""
        if not settings.SCHEDULER_ENABLED:
            logger.info("Background scheduler is disabled via configuration.")
            return

        if self._running:
            logger.warning("Background scheduler is already running.")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="SchedulerWorker")
        self._thread.start()
        logger.info(
            f"Background scheduler started (check interval: {settings.SCHEDULER_CHECK_INTERVAL_SECONDS}s, "
            f"grace period: {settings.SCHEDULER_GRACE_PERIOD_MINUTES}m)."
        )

    def stop(self) -> None:
        """Stops the scheduler thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("Background scheduler stopped.")

    def _run_loop(self) -> None:
        """Periodic evaluation loop."""
        while self._running:
            try:
                now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
                grace_delta = timedelta(minutes=settings.SCHEDULER_GRACE_PERIOD_MINUTES)
                # An hour H (e.g. 10:00 to 11:00) is ready to aggregate once now >= 11:05
                # Previous completed hour window start:
                candidate_hour = (now_utc - grace_delta - timedelta(hours=1)).replace(
                    minute=0, second=0, microsecond=0
                )

                db = SessionLocal()
                try:
                    run_pipeline_for_hour(
                        db=db,
                        target_hour=candidate_hour,
                        job_name="hourly_pipeline",
                        timeout_minutes=settings.SCHEDULER_LOCK_TIMEOUT_MINUTES,
                    )
                finally:
                    db.close()

            except Exception as exc:
                logger.error(f"Unexpected error in background scheduler loop: {exc}", exc_info=True)

            # Sleep until next check interval
            for _ in range(settings.SCHEDULER_CHECK_INTERVAL_SECONDS):
                if not self._running:
                    break
                time.sleep(1.0)


# Global scheduler singleton
background_scheduler = BackgroundScheduler()
