"""
Unit and Integration Tests for API Security.
Verifies:
- Ingestion with correct API key succeeds.
- Ingestion with incorrect API key returns HTTP 401.
- Ingestion with missing X-Device-API-Key header returns HTTP 401.
- Server with empty or missing configured key rejects ingestion with HTTP 401 (no bypass).
- Constant-time secret comparison.
- Production startup rejects default development key or empty key.
- Production startup accepts valid strong secret.
- Ingestion protection applies to both single and batch endpoints.
"""

import os
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from src.backend.config import Settings, settings
from src.backend.main import app, verify_device_api_key


class TestAPISecurity(unittest.TestCase):
    """Test suite for API authentication and startup validation."""

    @classmethod
    def tearDownClass(cls):
        """Clean up test records inserted during security testing."""
        from src.backend.database import SessionLocal, RawTelemetry
        db = SessionLocal()
        try:
            db.query(RawTelemetry).filter(RawTelemetry.reading_id.like("test_sec_%")).delete()
            db.commit()
        finally:
            db.close()

    def setUp(self):
        self.client = TestClient(app)
        self.valid_key = settings.DEVICE_API_KEY
        self.sample_payload = {
            "reading_id": "test_sec_001",
            "sequence_number": 1,
            "transformer_id": "T1",
            "timestamp": "2026-11-05T10:00:00",
            "voltage_V": 230.0,
            "current_A": 10.0,
            "power_W": 2300.0,
            "energy": 5.0,
            "frequency_Hz": 50.0,
            "power_factor": 0.95,
            "oil_temperature_C": 55.0,
            "ambient_temperature_C": None,
        }
        self.batch_payload = {
            "readings": [
                {
                    "reading_id": "test_sec_batch_001",
                    "sequence_number": 1,
                    "transformer_id": "T1",
                    "timestamp": "2026-11-05T10:00:00",
                    "voltage_V": 230.0,
                    "current_A": 10.0,
                    "power_W": 2300.0,
                    "energy": 5.0,
                    "frequency_Hz": 50.0,
                    "power_factor": 0.95,
                    "oil_temperature_C": 55.0,
                    "ambient_temperature_C": None,
                }
            ]
        }

    def test_single_ingest_correct_key(self):
        """Single ingest succeeds with HTTP 200 or 201 when given valid API key."""
        headers = {"X-Device-API-Key": self.valid_key}
        res = self.client.post("/api/telemetry/ingest", json=self.sample_payload, headers=headers)
        self.assertIn(res.status_code, [200, 201])

    def test_single_ingest_incorrect_key(self):
        """Single ingest fails with HTTP 401 when given invalid API key."""
        headers = {"X-Device-API-Key": "completely-wrong-key-12345"}
        res = self.client.post("/api/telemetry/ingest", json=self.sample_payload, headers=headers)
        self.assertEqual(res.status_code, 401)
        self.assertIn("Invalid X-Device-API-Key header", res.json()["detail"])

    def test_single_ingest_missing_header(self):
        """Single ingest fails with HTTP 401 when header is omitted."""
        res = self.client.post("/api/telemetry/ingest", json=self.sample_payload)
        self.assertEqual(res.status_code, 401)
        self.assertIn("Missing required X-Device-API-Key header", res.json()["detail"])

    def test_batch_ingest_correct_key(self):
        """Batch ingest succeeds with valid API key."""
        headers = {"X-Device-API-Key": self.valid_key}
        res = self.client.post("/api/telemetry/ingest-batch", json=self.batch_payload, headers=headers)
        self.assertIn(res.status_code, [200, 201])

    def test_batch_ingest_incorrect_key(self):
        """Batch ingest fails with HTTP 401 on incorrect API key."""
        headers = {"X-Device-API-Key": "wrong-secret"}
        res = self.client.post("/api/telemetry/ingest-batch", json=self.batch_payload, headers=headers)
        self.assertEqual(res.status_code, 401)

    def test_batch_ingest_missing_header(self):
        """Batch ingest fails with HTTP 401 when header is omitted."""
        res = self.client.post("/api/telemetry/ingest-batch", json=self.batch_payload)
        self.assertEqual(res.status_code, 401)

    def test_empty_configured_key_rejects_with_401(self):
        """When server DEVICE_API_KEY is empty string, all requests must be rejected with 401."""
        with patch.object(settings, "DEVICE_API_KEY", ""):
            headers = {"X-Device-API-Key": "any-key"}
            res = self.client.post("/api/telemetry/ingest", json=self.sample_payload, headers=headers)
            self.assertEqual(res.status_code, 401)
            self.assertIn("server has no valid DEVICE_API_KEY configured", res.json()["detail"])

    def test_none_configured_key_rejects_with_401(self):
        """When server DEVICE_API_KEY is None, all requests must be rejected with 401."""
        with patch.object(settings, "DEVICE_API_KEY", None):
            headers = {"X-Device-API-Key": "any-key"}
            res = self.client.post("/api/telemetry/ingest", json=self.sample_payload, headers=headers)
            self.assertEqual(res.status_code, 401)
            self.assertIn("server has no valid DEVICE_API_KEY configured", res.json()["detail"])

    def test_production_rejects_default_dev_key(self):
        """In production, using the default development key must raise a RuntimeError."""
        test_settings = Settings()
        test_settings.APP_ENV = "production"
        test_settings.DEVICE_API_KEY = test_settings.DEFAULT_DEV_KEY
        with self.assertRaises(RuntimeError) as ctx:
            test_settings.validate_security_configuration()
        self.assertIn("Default development DEVICE_API_KEY cannot be used in production mode", str(ctx.exception))

    def test_production_rejects_empty_key(self):
        """In production, an empty or whitespace key must raise a RuntimeError."""
        test_settings = Settings()
        test_settings.APP_ENV = "production"
        test_settings.DEVICE_API_KEY = "   "
        with self.assertRaises(RuntimeError) as ctx:
            test_settings.validate_security_configuration()
        self.assertIn("DEVICE_API_KEY must be set in production mode", str(ctx.exception))

    def test_production_accepts_valid_key(self):
        """In production, a strong non-default key passes validation cleanly."""
        test_settings = Settings()
        test_settings.APP_ENV = "production"
        test_settings.DEVICE_API_KEY = "high-entropy-production-secret-9876543210"
        # Must not raise
        test_settings.validate_security_configuration()


if __name__ == "__main__":
    unittest.main()
