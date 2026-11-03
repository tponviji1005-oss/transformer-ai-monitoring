"""
Backend Configuration Module.
Defines application settings, SQLite database paths, CORS origins, and server metadata.
"""

import os
from typing import List


class Settings:
    """Application configuration settings."""
    PROJECT_NAME: str = "Transformer AI Monitoring Backend"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
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
    CORS_ORIGINS: List[str] = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "*",  # Allow local network ESP32 and remote clients
    ]


settings = Settings()

