"""
Hourly Telemetry Aggregation Service.
Aggregates high-frequency raw ESP32 telemetry (2-second readings) into 1-hour
averages suitable for subsequent ML inference, monitoring, and historical analysis.

Key Architectural Guarantees:
1. Hourly bucketing: Bins raw observations into [hour_start, hour_end) 60-minute windows.
2. Incomplete hour exclusion: Only finalized/completed hours strictly before the cutoff are aggregated.
3. No phantom rows: Empty hours with zero raw telemetry produce no hourly records.
4. Idempotency: Repeated executions update existing hourly rows without creating duplicate rows.
5. Missing data integrity: Null ambient temperatures remain strictly NULL; never substituted with oil temp.
6. Multi-transformer independence: Aggregation operates per (transformer_id, hour_start).
"""

from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.backend.database import HourlyTelemetry, RawTelemetry

logger = logging.getLogger("transformer_backend.aggregation")


def aggregate_hourly_telemetry(
    db: Session,
    transformer_id: Optional[str] = None,
    reference_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Scans raw telemetry, aggregates completed hours into arithmetic averages,
    and persists or updates hourly records in SQLite.

    Args:
        db: Active SQLAlchemy transactional session.
        transformer_id: Optional transformer identifier filter (e.g. 'T1', 'T2', 'T3').
                        If None, processes all transformers present in raw telemetry.
        reference_time: Optional reference time to evaluate hour completion.
                        Defaults to current UTC time.
                        Only hours where hour_end <= cutoff (i.e. strictly before the
                        current active hour start) are aggregated.

    Returns:
        Dict detailing records_created, records_updated, and execution metadata.
    """
    if reference_time is None:
        ref = datetime.now(timezone.utc)
    else:
        ref = reference_time

    # The cutoff is the start of the current, incomplete hour.
    # Any reading strictly before this cutoff belongs to a completed 1-hour window.
    cutoff = ref.replace(minute=0, second=0, microsecond=0)

    # Build SQL grouping query using SQLite strftime to extract hour buckets
    hour_bucket_expr = func.strftime("%Y-%m-%d %H:00:00", RawTelemetry.timestamp).label("hour_bucket")

    query = (
        select(
            RawTelemetry.transformer_id,
            hour_bucket_expr,
            func.count(RawTelemetry.id).label("reading_count"),
            func.avg(RawTelemetry.voltage_V).label("avg_voltage"),
            func.avg(RawTelemetry.current_A).label("avg_current"),
            func.avg(RawTelemetry.power_W).label("avg_power"),
            func.avg(RawTelemetry.frequency_Hz).label("avg_frequency"),
            func.avg(RawTelemetry.power_factor).label("avg_power_factor"),
            func.avg(RawTelemetry.oil_temperature_C).label("avg_oil_temp"),
            func.avg(RawTelemetry.ambient_temperature_C).label("avg_ambient_temp"),
        )
        .where(RawTelemetry.timestamp < cutoff)
        .group_by(RawTelemetry.transformer_id, "hour_bucket")
        .order_by(RawTelemetry.transformer_id, "hour_bucket")
    )

    if transformer_id:
        clean_tid = transformer_id.strip().upper()
        query = query.where(RawTelemetry.transformer_id == clean_tid)

    results = db.execute(query).all()

    records_created = 0
    records_updated = 0

    for row in results:
        tid = row.transformer_id
        hour_start = datetime.strptime(row.hour_bucket, "%Y-%m-%d %H:%M:%S")
        hour_end = hour_start + timedelta(hours=1)

        # Ambient temperature average: strictly None if all rows were null
        avg_ambient = (
            round(float(row.avg_ambient_temp), 4)
            if row.avg_ambient_temp is not None
            else None
        )

        # Check if record already exists for (transformer_id, hour_start)
        existing = (
            db.query(HourlyTelemetry)
            .filter(
                HourlyTelemetry.transformer_id == tid,
                HourlyTelemetry.hour_start == hour_start,
            )
            .first()
        )

        if existing:
            # Update existing hourly record (Idempotent update)
            existing.voltage_V = round(float(row.avg_voltage), 4)
            existing.current_A = round(float(row.avg_current), 4)
            existing.power_W = round(float(row.avg_power), 4)
            existing.frequency_Hz = round(float(row.avg_frequency), 4)
            existing.power_factor = round(float(row.avg_power_factor), 4)
            existing.oil_temperature_C = round(float(row.avg_oil_temp), 4)
            existing.ambient_temperature_C = avg_ambient
            existing.reading_count = int(row.reading_count)
            existing.hour_end = hour_end
            existing.aggregation_status = "completed"
            records_updated += 1
        else:
            # Insert newly finalized hourly record
            new_record = HourlyTelemetry(
                transformer_id=tid,
                hour_start=hour_start,
                hour_end=hour_end,
                voltage_V=round(float(row.avg_voltage), 4),
                current_A=round(float(row.avg_current), 4),
                power_W=round(float(row.avg_power), 4),
                frequency_Hz=round(float(row.avg_frequency), 4),
                power_factor=round(float(row.avg_power_factor), 4),
                oil_temperature_C=round(float(row.avg_oil_temp), 4),
                ambient_temperature_C=avg_ambient,
                reading_count=int(row.reading_count),
                aggregation_status="completed",
                created_at=datetime.now(timezone.utc),
            )
            db.add(new_record)
            records_created += 1

    db.commit()

    total_processed = records_created + records_updated
    msg = (
        f"Aggregated {total_processed} hourly records for completed windows "
        f"(Created: {records_created}, Updated: {records_updated}, Cutoff: {cutoff.isoformat()})."
    )
    logger.info(msg)

    return {
        "status": "success",
        "records_created": records_created,
        "records_updated": records_updated,
        "total_processed": total_processed,
        "cutoff_time": cutoff.isoformat(),
        "message": msg,
    }


def get_hourly_history(
    db: Session,
    transformer_id: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 100,
    offset: int = 0,
) -> Tuple[int, List[HourlyTelemetry]]:
    """
    Retrieves chronologically sorted hourly aggregated records for a specific transformer.

    Args:
        db: Active SQLAlchemy transactional session.
        transformer_id: Transformer identifier.
        start_time: Optional lower bound filter on hour_start.
        end_time: Optional upper bound filter on hour_start.
        limit: Maximum number of records to return (pagination).
        offset: Number of records to skip (pagination).

    Returns:
        Tuple of (total_count, records_list).
    """
    clean_tid = transformer_id.strip().upper()
    query = select(HourlyTelemetry).where(HourlyTelemetry.transformer_id == clean_tid)

    if start_time:
        query = query.where(HourlyTelemetry.hour_start >= start_time)
    if end_time:
        query = query.where(HourlyTelemetry.hour_start <= end_time)

    # Compute total matching count
    count_query = (
        select(func.count(HourlyTelemetry.id))
        .where(HourlyTelemetry.transformer_id == clean_tid)
    )
    if start_time:
        count_query = count_query.where(HourlyTelemetry.hour_start >= start_time)
    if end_time:
        count_query = count_query.where(HourlyTelemetry.hour_start <= end_time)

    total_count = db.scalar(count_query) or 0

    # Retrieve paginated records ordered chronologically
    records = (
        db.execute(
            query.order_by(HourlyTelemetry.hour_start.asc())
            .offset(offset)
            .limit(limit)
        )
        .scalars()
        .all()
    )

    return total_count, list(records)

