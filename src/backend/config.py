"""
Backend Configuration Module.
Defines application settings, SQLite database paths, CORS origins, and server metadata.
"""

import json
import os
from typing import List


class Settings:
    """Application configuration settings."""
    PROJECT_NAME: str = "Transformer AI Monitoring Backend"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    APP_ENV: str = os.getenv("APP_ENV", "development").lower()
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")

    # Host & Port
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # SQLite Database Configuration
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    DB_FILE: str = os.path.join(DATA_DIR, "transformer_monitoring.db")
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{DB_FILE}")

    # CORS configuration for web dashboard access
    @staticmethod
    def _parse_cors_origins() -> List[str]:
        raw = os.getenv("CORS_ORIGINS")
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                return [orig.strip() for orig in raw.split(",") if orig.strip()]
        return [
            "http://localhost",
            "http://localhost:3000",
            "http://localhost:8000",
            "http://localhost:5173",
            "http://127.0.0.1",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000",
            "http://127.0.0.1:5173",
        ]

    CORS_ORIGINS: List[str] = _parse_cors_origins()

    # Device Authentication
    # Used to authenticate ESP32 telemetry ingestion via X-Device-API-Key header
    DEFAULT_DEV_KEY: str = "dev-transformer-secret-key-change-in-prod"
    DEVICE_API_KEY: str = os.getenv("DEVICE_API_KEY", DEFAULT_DEV_KEY)

    def validate_security_configuration(self) -> None:
        """
        Validates API security configuration.
        In production ('production', 'prod'), strictly enforces a non-empty, non-default DEVICE_API_KEY.
        Raises RuntimeError if validation fails.
        """
        is_prod = self.APP_ENV in ("production", "prod")
        key = (self.DEVICE_API_KEY or "").strip()
        if is_prod:
            if not key:
                raise RuntimeError(
                    "CRITICAL SECURITY CONFIGURATION ERROR: DEVICE_API_KEY must be set in production mode."
                )
            if key == self.DEFAULT_DEV_KEY:
                raise RuntimeError(
                    "CRITICAL SECURITY CONFIGURATION ERROR: Default development DEVICE_API_KEY cannot be used in production mode."
                )

    # Scheduler Settings
    SCHEDULER_ENABLED: bool = os.getenv("SCHEDULER_ENABLED", "false").lower() in ("true", "1", "yes")
    SCHEDULER_CHECK_INTERVAL_SECONDS: int = int(os.getenv("SCHEDULER_CHECK_INTERVAL_SECONDS", "60"))
    SCHEDULER_GRACE_PERIOD_MINUTES: int = int(os.getenv("SCHEDULER_GRACE_PERIOD_MINUTES", "5"))
    SCHEDULER_LOCK_TIMEOUT_MINUTES: int = int(os.getenv("SCHEDULER_LOCK_TIMEOUT_MINUTES", "15"))

    # Transformer Nameplate Specs (kVA, phases, power factor)
    # Strictly unverified/unconfigured by default per user constraint.
    # Must remain unconfigured until hardware owner in Madurai provides verified nameplate ratings.
    TRANSFORMER_SPECS = {
        "T1": {"rated_kVA": None, "phases": None, "rated_pf": None, "verified": False},
        "T2": {"rated_kVA": None, "phases": None, "rated_pf": None, "verified": False},
        "T3": {"rated_kVA": None, "phases": None, "rated_pf": None, "verified": False},
    }


settings = Settings()

