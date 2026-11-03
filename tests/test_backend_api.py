"""
Unit and Integration Test Suite for Milestone 5 Backend Foundation Stage.
Tests:
1. Root metadata endpoint.
2. GET /api/health status, database connectivity, and record counts.
3. POST /api/telemetry/ingest with complete sensor telemetry.
4. POST /api/telemetry/ingest with nullable ambient temperature (strictly NULL in DB, never substituted).
5. Physical boundary validation rejecting invalid electrical sensor signals (HTTP 422 / ValidationError).
6. Direct SQLite database persistence, field types, and retrieval.
7. Simulated multi-transformer 2-second streaming sequence.
"""

from datetime import datetime, timezone
import os
import sys
import unittest
from pydantic import ValidationError

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend.config import settings
from src.backend.database import RawTelemetry, SessionLocal, init_db
from src.backend.schemas import (
    HealthResponse,
    TelemetryIngestRequest,
    TelemetryIngestResponse,
)
from src.backend.main import get_health_status, ingest_raw_telemetry, root_info


class TestBackendFoundation(unittest.TestCase):
    """Test suite for Milestone 5 backend foundation."""

    @classmethod
    def setUpClass(cls):
        """Initialize database tables before running tests."""
        init_db()

    def setUp(self):
        """Create fresh transactional database session for each test."""
        self.db = SessionLocal()

    def tearDown(self):
        """Close database session after each test."""
        self.db.close()

    def test_01_root_endpoint(self):
        """Verify root endpoint returns API metadata and valid documentation links."""
        info = root_info()
        self.assertEqual(info["service"], settings.PROJECT_NAME)
        self.assertEqual(info["version"], settings.VERSION)
        self.assertIn("docs", info)
        self.assertIn("health", info)

    def test_02_health_endpoint(self):
        """Verify health check verifies SQLite connectivity and returns healthy status."""
        health: HealthResponse = get_health_status(db=self.db)
        self.assertEqual(health.status, "healthy")
        self.assertTrue(health.database_connected)
        self.assertEqual(health.service, settings.PROJECT_NAME)
        self.assertGreaterEqual(health.total_raw_readings, 0)
        self.assertIsInstance(health.timestamp, datetime)

    def test_03_ingest_valid_telemetry_with_ambient(self):
        """Verify ingestion and database persistence of valid telemetry with ambient temperature."""
        payload = TelemetryIngestRequest(
            transformer_id="T2",
            timestamp=datetime(2026, 11, 2, 23, 15, 0),
            voltage_V=415.2,
            current_A=128.5,
            power_W=89400.0,
            energy=1540.2,
            frequency_Hz=50.01,
            power_factor=0.91,
            oil_temperature_C=58.7,
            ambient_temperature_C=31.4,
        )
        response: TelemetryIngestResponse = ingest_raw_telemetry(payload=payload, db=self.db)
        self.assertEqual(response.status, "success")
        self.assertEqual(response.transformer_id, "T2")
        self.assertTrue(response.ambient_temperature_supplied)
        self.assertGreater(response.record_id, 0)

        # Verify exact field values in SQLite database
        record = self.db.query(RawTelemetry).filter(RawTelemetry.id == response.record_id).first()
        self.assertIsNotNone(record)
        self.assertEqual(record.transformer_id, "T2")
        self.assertAlmostEqual(record.voltage_V, 415.2, places=1)
        self.assertAlmostEqual(record.current_A, 128.5, places=1)
        self.assertAlmostEqual(record.power_W, 89400.0, places=1)
        self.assertAlmostEqual(record.energy, 1540.2, places=1)
        self.assertAlmostEqual(record.frequency_Hz, 50.01, places=2)
        self.assertAlmostEqual(record.power_factor, 0.91, places=2)
        self.assertAlmostEqual(record.oil_temperature_C, 58.7, places=1)
        self.assertAlmostEqual(record.ambient_temperature_C, 31.4, places=1)

    def test_04_ingest_valid_telemetry_without_ambient(self):
        """
        Verify ingestion with missing/null ambient temperature.
        CRITICAL RULE: ambient_temperature_C must be stored as strictly NULL in SQLite.
        It must NEVER be replaced or fabricated with oil temperature.
        """
        payload = TelemetryIngestRequest(
            transformer_id="t1",  # Test normalization to T1
            timestamp=datetime(2026, 11, 2, 23, 15, 2),
            voltage_V=414.8,
            current_A=65.2,
            power_W=45100.0,
            energy=None,
            frequency_Hz=49.98,
            power_factor=0.89,
            oil_temperature_C=42.5,
            ambient_temperature_C=None,  # Nullable
        )
        response: TelemetryIngestResponse = ingest_raw_telemetry(payload=payload, db=self.db)
        self.assertEqual(response.status, "success")
        self.assertEqual(response.transformer_id, "T1")  # Normalized to uppercase
        self.assertFalse(response.ambient_temperature_supplied)

        # Verify database entry has NULL ambient temperature
        record = self.db.query(RawTelemetry).filter(RawTelemetry.id == response.record_id).first()
        self.assertIsNotNone(record)
        self.assertIsNone(record.ambient_temperature_C, "ambient_temperature_C must be strictly NULL in database")
        self.assertAlmostEqual(record.oil_temperature_C, 42.5, places=1)
        self.assertNotEqual(record.ambient_temperature_C, record.oil_temperature_C)

    def test_05_reject_invalid_frequency(self):
        """Verify validation rejects unrealistic grid frequency (<40 Hz or >65 Hz)."""
        with self.assertRaises(ValidationError):
            TelemetryIngestRequest(
                transformer_id="T3",
                timestamp=datetime.now(),
                voltage_V=415.0,
                current_A=50.0,
                power_W=35000.0,
                frequency_Hz=25.0,  # Invalid
                power_factor=0.90,
                oil_temperature_C=40.0,
            )

    def test_06_reject_invalid_power_factor(self):
        """Verify validation rejects out-of-range power factor (> 1.0)."""
        with self.assertRaises(ValidationError):
            TelemetryIngestRequest(
                transformer_id="T3",
                timestamp=datetime.now(),
                voltage_V=415.0,
                current_A=50.0,
                power_W=35000.0,
                frequency_Hz=50.0,
                power_factor=1.8,  # Mathematically invalid cos phi
                oil_temperature_C=40.0,
            )

    def test_07_reject_negative_voltage(self):
        """Verify validation rejects negative RMS voltage."""
        with self.assertRaises(ValidationError):
            TelemetryIngestRequest(
                transformer_id="T3",
                timestamp=datetime.now(),
                voltage_V=-415.0,  # Invalid negative RMS
                current_A=50.0,
                power_W=35000.0,
                frequency_Hz=50.0,
                power_factor=0.90,
                oil_temperature_C=40.0,
            )

    def test_08_reject_empty_transformer_id(self):
        """Verify validation rejects empty or blank transformer identifier."""
        with self.assertRaises(ValidationError):
            TelemetryIngestRequest(
                transformer_id="   ",
                timestamp=datetime.now(),
                voltage_V=415.0,
                current_A=50.0,
                power_W=35000.0,
                frequency_Hz=50.0,
                power_factor=0.90,
                oil_temperature_C=40.0,
            )

    def test_09_stream_simulation_multi_transformer(self):
        """Simulate sequential stream of 2-second readings across T1, T2, T3."""
        base_time = datetime(2026, 11, 2, 23, 16, 0)
        stored_ids = []
        for i, t_id in enumerate(["T1", "T2", "T3"]):
            payload = TelemetryIngestRequest(
                transformer_id=t_id,
                timestamp=base_time,
                voltage_V=415.0 + (i * 0.5),
                current_A=70.0 + (i * 15.0),
                power_W=50000.0 + (i * 10000.0),
                energy=100.0 + i,
                frequency_Hz=50.00,
                power_factor=0.90,
                oil_temperature_C=45.0 + i,
            )
            resp = ingest_raw_telemetry(payload=payload, db=self.db)
            self.assertEqual(resp.status, "success")
            self.assertEqual(resp.transformer_id, t_id)
            stored_ids.append(resp.record_id)

        self.assertEqual(len(stored_ids), 3)
        self.assertEqual(len(set(stored_ids)), 3, "All records must have distinct auto-incremented IDs")


if __name__ == "__main__":
    unittest.main()
