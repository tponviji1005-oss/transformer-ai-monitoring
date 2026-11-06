# Milestone 5 – Stage 5: Live Hardware-to-Backend Integration Audit Report

> **Project:** AI-Based Transformer Load, Overload and Efficiency Prediction System  
> **Repository:** `transformer-ai-monitoring`  
> **Milestone:** Milestone 5 – Production-Grade Backend & Inference Engine  
> **Stage:** Stage 5 – Live Hardware-to-Backend Integration  
> **Date:** November 5, 2026  
> **Author:** Software & ML Engineering Team (Chennai)  
> **Hardware Partner:** Independent Hardware Team (Madurai)

---

## 1. Executive Summary

Milestone 5 Stage 5 establishes the production-grade communication link between the remote ESP32 microcontroller (located in Madurai) and the central FastAPI monitoring backend. It bridges physical sensor acquisition with automated machine learning inference while maintaining strict physical safety boundaries, data integrity, and idempotency.

All 50 unit and integration tests across the backend foundation, hourly aggregation, live feature generation, live inference, and hardware integration pass with zero errors (100% pass rate). Existing historical records (5 raw telemetry records and 169 baseline hourly records) in `data/transformer_monitoring.db` were preserved intact through safe, non-destructive database migrations.

---

## 2. Hardware Architecture & Confirmed Pinout

The physical hardware integration has been implemented and tested against the exact circuit design confirmed by the hardware partner in Madurai:

| Component | Role | Pin / Interface | ESP32 GPIO (DOIT DEVKIT V1 38-Pin) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Microcontroller** | Sensor Acquisition & Transmission | ESP32 DOIT DEVKIT V1 | — | 38-Pin variant, 2.4 GHz 802.11 b/g/n Wi-Fi |
| **PZEM-004T V4.0** | AC Electrical Measurements | Hardware Serial UART2 | **RX2: GPIO 16** (connects to PZEM TX)<br>**TX2: GPIO 17** (connects to PZEM RX) | Reads V, I, P, E, Freq, PF every 2 seconds |
| **MAX6675 Module** | Top Oil Temperature Measurement | Dedicated SPI Bus | **SCK: GPIO 18** (Clock)<br>**CS: GPIO 5** (Chip Select)<br>**SO: GPIO 19** (MISO) | Cold-junction compensated K-thermocouple (0–1024°C) |
| **Ambient Temp** | Outdoor Weather Channel | Unconfigured / Unconnected | **Strictly None (NULL)** | Preserved as NULL; never fabricated or substituted with oil temp |

---

## 3. Key Architectural Implementations

### 3.1. Collision-Free Reading Identifiers Across Boots & Power Cycles
To eliminate duplicate readings caused by lost HTTP ACKs, packet retransmissions, or power brownouts:
* At boot, the ESP32 generates an 8-character hex string `boot_id` using the hardware true random number generator (`esp_random()`).
* The firmware maintains a 32-bit monotonically increasing `sequence_number` initialized to 1.
* Each observation constructs a composite identifier:
  $$\text{reading\_id} = \text{transformer\_id} + \text{"-"} + \text{boot\_id} + \text{"-"} + \text{String(sequence\_number, DEC)}$$
  *(Example: `T1-B94A7C10-00000042`)*
* Even if the ESP32 reboots and sequence counter restarts from 1, the new `boot_id` guarantees zero collision with pre-reboot readings.

### 3.2. Idempotent Ingestion Deduplication
* `raw_telemetry` table in SQLite is indexed via a partial unique index:
  ```sql
  CREATE UNIQUE INDEX idx_raw_telemetry_reading_id ON raw_telemetry(reading_id) WHERE reading_id IS NOT NULL;
  ```
* When an existing `reading_id` arrives:
  1. Backend skips row insertion.
  2. Backend acknowledges receipt with **HTTP 200 OK** and body:
     ```json
     {"status": "duplicate", "message": "Duplicate reading T1-B94A7C10-00000042 already recorded."}
     ```
  3. The ESP32 receives HTTP 200, safely evicts the packet from its buffer, and advances.

### 3.3. Ingestion Authentication Security
* Ingestion endpoints (`/api/telemetry/ingest` and `/api/telemetry/ingest-batch`) require the `X-Device-API-Key` HTTP header.
* Missing or incorrect API keys return **HTTP 401 Unauthorized**.
* All real credentials remain outside version control (`.env` and `firmware/config.h` are gitignored; templates provided in `.env.example` and `firmware/config.h.example`).

### 3.4. HTTP 401 Buffer Preservation & Backoff on ESP32
* If the backend returns `HTTP 401`:
  1. **Zero Data Loss:** Valid sensor readings in the RAM ring buffer are **preserved** (never discarded or popped).
  2. **Serial Diagnostic:** Detailed warning logged to Serial: `[AUTH ERROR] HTTP 401: Invalid X-Device-API-Key. Please verify firmware/config.h.`
  3. **Rate Limiting:** Transmission enters a 60-second backoff loop (checking once per minute rather than spamming the backend).
  4. Once credentials are corrected, transmission resumes and flushes preserved packets chronologically.

### 3.5. RAM Ring Buffer & Offline Resilience
* **Capacity:** 60 readings in volatile RAM ring buffer.
* **Duration:** $60 \text{ readings} \times 2.0\text{ seconds} = 120\text{ seconds}$ ($2.0\text{ minutes}$ of continuous offline telemetry).
* **Limitation:** If Wi-Fi outage exceeds 2 minutes, oldest readings are overwritten (FIFO) to prevent heap exhaustion.
* **Flush:** Upon reconnect, buffered readings are transmitted in chronological order.

### 3.6. Scheduler Concurrency, Stale-Lock Recovery & Late Telemetry
* Table `scheduler_locks` coordinates atomic execution for hourly windows `(job_name, target_hour)`:
  - Active lock within 15 minutes: Concurrent workers cleanly abort without duplicate runs.
  - Completed lock (`status = 'COMPLETED'`): Never re-executed.
  - Stale lock ($> 15\text{ minutes}$ old): Reclaimed by recovery worker if a previous process died or hung.
* **Late Telemetry Recalculation:** When delayed packets arrive for an already aggregated hour $H$, `aggregate_hourly_telemetry` updates the row in place (`records_updated += 1`), recalculating arithmetic averages and updating `reading_count`. Zero duplicate rows are created in `hourly_telemetry`, `live_features`, or `predictions`.

### 3.7. Simulation Database & Port Isolation
* The stream simulator (`scripts/simulate_esp32_stream.py`) targets `http://127.0.0.1:8001` (`data/simulation_telemetry.db`).
* Contains a hardcoded safety assertion that raises `RuntimeError` immediately if targeted at port `8000` or production hosts, ensuring synthetic data never contaminates real telemetry.

---

## 4. Database Verification & Migration Safety

* **Pre-Migration Backup:** `data/backups/transformer_monitoring_pre_stage5.db` created prior to schema modification.
* **Schema Evolution:**
  - `RawTelemetry.reading_id`: Added `VARCHAR(64)` nullable column.
  - `RawTelemetry.sequence_number`: Added `INTEGER` nullable column.
  - Partial unique index: `idx_raw_telemetry_reading_id` on `raw_telemetry(reading_id) WHERE reading_id IS NOT NULL`.
  - Table `scheduler_locks`: Created with columns `job_name`, `target_hour`, `worker_pid`, `status`, `locked_at`, `completed_at`.
* **Record Count Audit:**
  - `raw_telemetry`: **5 records** (100% preserved).
  - `hourly_telemetry`: **169 records** (100% preserved).
  - `live_features`: **0 records** (clean state).
  - `predictions`: **0 records** (clean state).
  - `scheduler_locks`: **0 records** (clean state).

---

## 5. API Endpoints Specification

### 5.1. Single Telemetry Ingestion
* **Endpoint:** `POST /api/telemetry/ingest`
* **Headers:** `Content-Type: application/json`, `X-Device-API-Key: <secret>`
* **Request Payload Example:**
  ```json
  {
    "reading_id": "T1-B94A7C10-00000042",
    "sequence_number": 42,
    "transformer_id": "T1",
    "timestamp": "2026-11-20T10:00:00",
    "voltage_V": 415.2,
    "current_A": 65.4,
    "power_W": 24500.0,
    "energy": 1250.45,
    "frequency_Hz": 50.01,
    "power_factor": 0.90,
    "oil_temperature_C": 52.4,
    "ambient_temperature_C": null
  }
  ```
* **Success Response (HTTP 201 Created):**
  ```json
  {
    "status": "success",
    "message": "Raw telemetry for T1 persisted successfully.",
    "record_id": 6,
    "transformer_id": "T1",
    "timestamp": "2026-11-20T10:00:00",
    "reading_id": "T1-B94A7C10-00000042",
    "ambient_temperature_supplied": false
  }
  ```
* **Duplicate Acknowledgment (HTTP 200 OK):**
  ```json
  {
    "status": "duplicate",
    "message": "Duplicate reading T1-B94A7C10-00000042 already recorded.",
    "record_id": 6,
    "transformer_id": "T1",
    "timestamp": "2026-11-20T10:00:00",
    "reading_id": "T1-B94A7C10-00000042",
    "ambient_temperature_supplied": false
  }
  ```

### 5.2. Micro-Batch Telemetry Ingestion (Buffer Flush)
* **Endpoint:** `POST /api/telemetry/ingest-batch`
* **Headers:** `Content-Type: application/json`, `X-Device-API-Key: <secret>`
* **Request Payload Example:**
  ```json
  {
    "readings": [
      {
        "reading_id": "T1-B94A7C10-00000043",
        "sequence_number": 43,
        "transformer_id": "T1",
        "timestamp": "2026-11-20T10:00:02",
        "voltage_V": 415.0,
        "current_A": 65.2,
        "power_W": 24450.0,
        "frequency_Hz": 50.0,
        "power_factor": 0.90,
        "oil_temperature_C": 52.4
      },
      {
        "reading_id": "T1-B94A7C10-00000044",
        "sequence_number": 44,
        "transformer_id": "T1",
        "timestamp": "2026-11-20T10:00:04",
        "voltage_V": 414.8,
        "current_A": 65.5,
        "power_W": 24510.0,
        "frequency_Hz": 50.01,
        "power_factor": 0.90,
        "oil_temperature_C": 52.5
      }
    ]
  }
  ```
* **Response (HTTP 200 OK):**
  ```json
  {
    "status": "success",
    "total_received": 2,
    "inserted_count": 2,
    "duplicate_count": 0,
    "record_ids": [7, 8]
  }
  ```

---

## 6. Verification & Test Suite Results

The complete test suite comprising **50 unit and integration tests** was executed:

| Test Suite | File | Tests Run | Pass | Fail | Errors | Coverage Focus |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Backend Foundation** | `tests/test_backend_api.py` | 9 | 9 | 0 | 0 | Root info, health check, electrical boundary validations, nullable ambient temperature persistence |
| **Hourly Aggregation** | `tests/test_aggregation_service.py` | 7 | 7 | 0 | 0 | 1-hour bucketing, incomplete window exclusion, idempotent update in place, missing ambient NULL preservation |
| **Live Feature Engine** | `tests/test_feature_service.py` | 13 | 13 | 0 | 0 | Exact 39 features, 168h cold start, gap detection, IST calendar/festivals, readiness gates |
| **Live ML Inference** | `tests/test_inference_service.py` | 13 | 13 | 0 | 0 | ModelRegistry singleton, T1 anomaly detection, T2 $\tau^*=0.3382$ classifier, T3 threshold monitor, advisory alerts, no efficiency |
| **Hardware Integration** | `tests/test_hardware_integration.py` | 8 | 8 | 0 | 0 | Safe migrations, API key enforcement, reading_id deduplication, batch ingest, scheduler concurrency lock, 15m crash recovery, late packet recalculation, simulation port guard |
| **TOTAL** | — | **50** | **50** | **0** | **0** | **100% Pass Rate** |

---

## 7. ESP32 Firmware Setup Runbook for Madurai Partner

To flash and test the ESP32 in Madurai:

1. **Prerequisites in Arduino IDE:**
   - Install ESP32 Board Package (by Espressif Systems).
   - Install libraries:
     - `PZEM004Tv30` (by Mandar Koske).
     - `MAX6675` (by Adafruit).
2. **Configuration:**
   - Navigate to `firmware/` directory.
   - Copy `config.h.example` to `config.h`:
     ```bash
     cp firmware/config.h.example firmware/config.h
     ```
   - In `config.h`, set:
     - `WIFI_SSID` and `WIFI_PASSWORD`.
     - `BACKEND_HOST`: Server IP (e.g. `192.168.1.50` on LAN, or public Cloudflare Tunnel hostname).
     - `BACKEND_PORT`: `8000`.
     - `DEVICE_API_KEY`: Matching `DEVICE_API_KEY` in backend `.env`.
     - `TRANSFORMER_ID`: `"T1"`, `"T2"`, or `"T3"`.
3. **Flashing:**
   - Select Board: `DOIT ESP32 DEVKIT V1`.
   - Select COM Port.
   - Click **Upload**.
4. **Serial Monitor Verification:**
   - Open Serial Monitor at **115200 baud**.
   - Verify boot message:
     ```
     ==================================================
     Transformer AI Monitoring - ESP32 Firmware v1.0
     ==================================================
     Boot Session ID: B94A7C10
     Transformer ID:  T1
     [WIFI] Connecting to MyNetwork.....
     [WIFI] Connected! IP Address: 192.168.1.105
     [NTP] Time synchronized successfully: 2026-11-20T10:00:00
     [OBS] T1-B94A7C10-00000001 | V=415.2V | I=65.2A | P=24500W | OilTemp=52.4C | Buf=1
     [INGEST SUCCESS] Record: T1-B94A7C10-00000001 | Buffered remaining: 0
     ```

---

## 8. Remaining Hardware Confirmations Required from Madurai

1. **Transformer Nameplate Capacity:**
   - Exact rated kVA for the physical transformer under test.
   - Confirmed phase configuration (single-phase or three-phase line-to-line vs line-to-neutral).
   - Once confirmed, register via `POST /api/transformers/{id}/specs` to clear the nameplate readiness blocker.
2. **PZEM CT Clamp Orientation:**
   - Confirm current transformer (CT) clamp arrow points in the direction of power flow (grid $\to$ transformer) to ensure positive active power and positive power factor readings.
3. **MAX6675 Thermocouple Placement:**
   - Confirm thermal probe is firmly affixed to the transformer top oil thermowell or outer enclosure with adequate thermal paste.
4. **Physical Network Access:**
   - Whether the ESP32 in Madurai connects via a local Wi-Fi router on the same subnet as the laptop, or via an internet gateway (requiring Cloudflare Tunnel / ngrok / static cloud IP).

---

## 9. Known Limitations

1. **60-Reading RAM Buffer:**
   - Outages longer than 120 seconds ($2.0\text{ minutes}$) result in FIFO eviction of oldest readings. If multi-hour outages are expected, external SPI Flash / SD card logging will be required.
2. **Cold-Start 168-Hour Requirement:**
   - Feature readiness gates will block inference (`is_ready_for_inference = 0`) until 168 continuous hours of hourly telemetry have accumulated on the live transformer.
3. **Ambient Temperature Absence:**
   - Physical hardware does not currently have an outdoor ambient temperature sensor connected. All 6 ambient-derived features remain NULL, and inference status will report `STATUS_BLOCKED_MISSING_AMBIENT` until a physical sensor is added.
4. **No Efficiency Calculation:**
   - Single-phase PZEM measures input side only. Transformer efficiency is physically unmeasurable without calibrated secondary-side power or heat-run loss certificates.

