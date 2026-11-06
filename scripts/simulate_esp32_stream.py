"""
Dedicated ESP32 Telemetry Stream Simulator.
Simulates high-frequency (2-second) sensor telemetry transmission to the backend.

SAFETY GUARDS:
1. Targets a dedicated test backend instance (default: http://127.0.0.1:8001).
2. HARD BLOCKS any execution targeting production port 8000 or the production database.
3. Does NOT fabricate ambient temperature unless explicitly requested via --with-ambient for test validation.
4. Uses monotonically increasing sequence numbers and unique boot session IDs.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
import random
import sys
import time
from typing import Dict, Optional
import urllib.error
import urllib.request

# Safety Configuration
DEFAULT_TARGET_URL = os.getenv("TARGET_BACKEND_URL", "http://127.0.0.1:8001")
DEVICE_API_KEY = os.getenv("DEVICE_API_KEY", "dev-transformer-secret-key-change-in-prod")


def enforce_safety_isolation(target_url: str) -> None:
    """
    Enforces strict physical isolation between simulation and production.
    Blocks execution if the target URL resolves to port 8000 or production environment.
    """
    lower_url = target_url.lower()

    if ":8000" in lower_url:
        raise RuntimeError(
            "SAFETY VIOLATION BLOCKED: The simulation script is attempting to target port 8000 (Production Backend).\n"
            "To run simulations, start a dedicated test server on port 8001:\n"
            "  uvicorn src.backend.main:app --port 8001 --env-file <(echo DATABASE_URL=sqlite:///data/simulation_telemetry.db)\n"
            "Or set TARGET_BACKEND_URL=http://127.0.0.1:8001"
        )

    if "prod" in lower_url or "transformer-monitoring" in lower_url or "transformer_monitoring" in lower_url:
        raise RuntimeError(
            "SAFETY VIOLATION BLOCKED: Simulation is forbidden from targeting production hosts or databases."
        )


class ESP32DeviceSimulator:
    """
    Simulates physical sensor sampling and HTTP POST transmission from an ESP32 microcontroller.
    """

    def __init__(
        self,
        transformer_id: str,
        target_url: str,
        api_key: str,
        with_ambient: bool = False,
    ):
        self.transformer_id = transformer_id.strip().upper()
        self.target_url = target_url.rstrip("/")
        self.api_key = api_key
        self.with_ambient = with_ambient

        # Generate unique boot session ID (8 hex characters)
        self.boot_id = f"{random.randint(0, 0xFFFFFFFF):08X}"
        self.sequence_number = 1

        # Physical thermal state
        self.oil_temp = 48.0
        self.cumulative_energy = 1000.0

        print(f"[SIMULATOR INITIALIZED] Transformer: {self.transformer_id} | Boot ID: {self.boot_id}")
        print(f"[TARGET BACKEND] {self.target_url}")
        print(f"[AMBIENT CHANNEL] {'SUPPLIED (--with-ambient)' if self.with_ambient else 'STRICTLY NULL (Real Hardware Mode)'}")

    def generate_reading(self) -> Dict:
        """
        Synthesizes a realistic physical observation.
        Includes diurnal variation, physical voltage fluctuations, and thermal response.
        """
        now = datetime.now(timezone.utc)
        hour_frac = now.hour + now.minute / 60.0

        # Diurnal load curve (peaks at 11:00 and 19:00)
        daily_curve = 0.5 + 0.3 * math.sin((hour_frac - 6.0) / 24.0 * 2.0 * math.pi)
        base_power = {
            "T1": 50000.0,
            "T2": 80000.0,
            "T3": 60000.0,
        }.get(self.transformer_id, 50000.0)

        jitter = random.uniform(-0.05, 0.05)
        active_power = max(1000.0, base_power * daily_curve * (1.0 + jitter))

        # Electrical measurements
        voltage = round(random.uniform(408.0, 418.0), 1)
        pf = round(random.uniform(0.88, 0.94), 2)
        current = round(active_power / (voltage * pf), 1)
        frequency = round(random.uniform(49.88, 50.12), 2)

        # Thermal inertia: oil temperature slowly tracks power dissipation
        target_oil = 35.0 + (active_power / base_power) * 28.0
        self.oil_temp += (target_oil - self.oil_temp) * 0.05
        oil_temp = round(self.oil_temp + random.uniform(-0.2, 0.2), 1)

        # Ambient temperature (only if requested)
        ambient_temp = None
        if self.with_ambient:
            ambient_temp = round(24.0 + 8.0 * math.sin((hour_frac - 9.0) / 24.0 * 2.0 * math.pi), 1)

        reading_id = f"{self.transformer_id}-{self.boot_id}-{self.sequence_number:08d}"

        reading = {
            "reading_id": reading_id,
            "sequence_number": self.sequence_number,
            "transformer_id": self.transformer_id,
            "timestamp": now.strftime("%Y-%m-%dT%H:%M:%S"),
            "voltage_V": voltage,
            "current_A": current,
            "power_W": active_power,
            "energy": round(self.cumulative_energy, 3),
            "frequency_Hz": frequency,
            "power_factor": pf,
            "oil_temperature_C": oil_temp,
            "ambient_temperature_C": ambient_temp,
        }

        self.sequence_number += 1
        self.cumulative_energy += (active_power * 2.0) / (3600.0 * 1000.0)

        return reading

    def send_reading(self, reading: Dict) -> bool:
        """Transmits a single reading to the backend ingest API using standard library urllib."""
        url = f"{self.target_url}/api/telemetry/ingest"
        data_bytes = json.dumps(reading).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={
                "Content-Type": "application/json",
                "X-Device-API-Key": self.api_key,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                code = resp.getcode()
                status_str = "DUPLICATE_ACK" if code == 200 else "SUCCESS_CREATED"
                print(
                    f"[{status_str} HTTP {code}] ID: {reading['reading_id']} | "
                    f"V={reading['voltage_V']}V, I={reading['current_A']}A, P={reading['power_W']:.0f}W, "
                    f"Oil={reading['oil_temperature_C']}°C"
                )
                return True
        except urllib.error.HTTPError as http_err:
            if http_err.code == 401:
                print("[AUTH ERROR HTTP 401] Ingestion rejected. Invalid X-Device-API-Key.")
            else:
                body = http_err.read().decode("utf-8", errors="replace")
                print(f"[ERROR HTTP {http_err.code}] {body}")
            return False
        except urllib.error.URLError as url_err:
            print(f"[NETWORK ERROR] Failed to connect to {url}: {url_err}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Simulate live ESP32 telemetry stream to backend.")
    parser.add_argument("--transformer", default="T1", choices=["T1", "T2", "T3"], help="Transformer ID")
    parser.add_argument("--target", default=DEFAULT_TARGET_URL, help="Target backend URL (default: http://127.0.0.1:8001)")
    parser.add_argument("--interval", type=float, default=2.0, help="Sampling interval in seconds (default: 2.0s)")
    parser.add_argument("--count", type=int, default=30, help="Number of telemetry packets to transmit (default: 30)")
    parser.add_argument("--with-ambient", action="store_true", help="Include synthetic ambient temperature for ML testing")
    parser.add_argument("--api-key", default=DEVICE_API_KEY, help="Device API key for X-Device-API-Key header")
    args = parser.parse_args()

    # Enforce database & port isolation
    enforce_safety_isolation(args.target)

    simulator = ESP32DeviceSimulator(
        transformer_id=args.transformer,
        target_url=args.target,
        api_key=args.api_key,
        with_ambient=args.with_ambient,
    )

    print(f"\nStarting stream of {args.count} observations at {args.interval}s interval...\n")

    transmitted = 0
    for i in range(args.count):
        reading = simulator.generate_reading()
        ok = simulator.send_reading(reading)
        if ok:
            transmitted += 1
        if i < args.count - 1:
            time.sleep(args.interval)

    print(f"\n[STREAM COMPLETE] Transmitted {transmitted}/{args.count} observations successfully.")


if __name__ == "__main__":
    main()
