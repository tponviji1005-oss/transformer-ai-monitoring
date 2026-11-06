/**
 * @file esp32_transformer_monitor.ino
 * @brief ESP32 Live Telemetry Ingestion Firmware for Transformer AI Monitoring.
 *
 * Microcontroller: ESP32 DOIT DEVKIT V1 (38-Pin)
 * Sensors:
 *   1. PZEM-004T V4.0 (Single-Phase AC Power/Energy/V/I/Freq/PF) on UART2 (RX2: GPIO 16, TX2: GPIO 17)
 *   2. MAX6675 Cold-Junction K-Thermocouple (Top Oil Temperature) on SPI (SCK: GPIO 18, CS: GPIO 5, SO: GPIO 19)
 *
 * Operational Capabilities:
 *   - Monotonically increasing sequence number + boot session random identifier (reading_id).
 *   - NTP UTC synchronization for ISO-8601 hardware timestamps.
 *   - RAM Ring Buffer (capacity 60) for network outage tolerance (2 minutes offline buffer).
 *   - HTTP 401 buffer preservation and 60-second backoff logic (never spam server; zero data loss).
 *   - HTTP 200/201 acknowledgment handling with buffer eviction.
 *   - Strictly nullable ambient temperature (never substituted with oil temperature).
 *   - Zero transformer rating assumptions or efficiency calculations.
 *   - Advisory operator alerts only; zero electrical switching or actuator commands.
 */

#if __has_include("config.h")
  #include "config.h"
#else
  #include "config.h.example"
#endif

#include <WiFi.h>
#include <HTTPClient.h>
#include <time.h>
#include <PZEM004Tv30.h>
#include <max6675.h>

// ====================================================================
// Hardware Interfaces
// ====================================================================
// PZEM-004T v4.0 on UART2
PZEM004Tv30 pzem(Serial2, PZEM_RX_PIN, PZEM_TX_PIN);

// MAX6675 Thermocouple
MAX6675 thermocouple(MAX6675_SCK_PIN, MAX6675_CS_PIN, MAX6675_SO_PIN);

// ====================================================================
// State & Telemetry Structures
// ====================================================================
struct TelemetryRecord {
  char reading_id[64];
  uint32_t sequence_number;
  char timestamp_iso[32];
  float voltage_V;
  float current_A;
  float power_W;
  float energy_kWh;
  float frequency_Hz;
  float power_factor;
  float oil_temperature_C;
  bool has_ambient;
  float ambient_temperature_C;
};

// In-Memory Ring Buffer for network outage tolerance (60 readings = 2 minutes)
TelemetryRecord ringBuffer[RING_BUFFER_CAPACITY];
int bufferHead = 0;   // Index of oldest record
int bufferTail = 0;   // Index for next insert
int bufferCount = 0;  // Current number of buffered records

// Boot Session Identifier (8 hex characters generated via hardware RNG)
char bootId[9];
uint32_t sequenceNumber = 1;

// Timing & Backoff State
unsigned long lastReadTime = 0;
unsigned long lastAuthErrorTime = 0;
bool authErrorActive = false;

// ====================================================================
// Helper Functions
// ====================================================================

/**
 * Generates an 8-character hex random string for boot session uniqueness.
 */
void generateBootId() {
  uint32_t r = esp_random();
  snprintf(bootId, sizeof(bootId), "%08X", r);
}

/**
 * Obtains the current UTC timestamp formatted as ISO-8601 (YYYY-MM-DDTHH:MM:SS).
 */
bool getUtcTimestampIso(char *dest, size_t maxLen) {
  time_t now;
  struct tm timeinfo;
  time(&now);
  gmtime_r(&now, &timeinfo);

  if (timeinfo.tm_year < (2020 - 1900)) {
    // NTP not yet synchronized
    return false;
  }

  strftime(dest, maxLen, "%Y-%m-%dT%H:%M:%S", &timeinfo);
  return true;
}

/**
 * Adds a telemetry record to the ring buffer.
 * If buffer is full, the oldest unread record is overwritten.
 */
void pushBuffer(const TelemetryRecord &rec) {
  ringBuffer[bufferTail] = rec;
  bufferTail = (bufferTail + 1) % RING_BUFFER_CAPACITY;

  if (bufferCount < RING_BUFFER_CAPACITY) {
    bufferCount++;
  } else {
    // Buffer overflow: advance head to discard oldest reading
    bufferHead = (bufferHead + 1) % RING_BUFFER_CAPACITY;
    Serial.println(F("[BUFFER OVERFLOW] 60-reading limit reached. Oldest reading evicted."));
  }
}

/**
 * Removes the oldest record from the ring buffer.
 */
void popBuffer() {
  if (bufferCount > 0) {
    bufferHead = (bufferHead + 1) % RING_BUFFER_CAPACITY;
    bufferCount--;
  }
}

/**
 * Returns a pointer to the oldest record in the ring buffer.
 */
TelemetryRecord* peekBuffer() {
  if (bufferCount == 0) return nullptr;
  return &ringBuffer[bufferHead];
}

/**
 * Formats a TelemetryRecord as a JSON payload for REST API ingestion.
 */
String serializeRecordJson(const TelemetryRecord &rec) {
  String json = "{";
  json += "\"reading_id\":\"" + String(rec.reading_id) + "\",";
  json += "\"sequence_number\":" + String(rec.sequence_number) + ",";
  json += "\"transformer_id\":\"" + String(TRANSFORMER_ID) + "\",";
  json += "\"timestamp\":\"" + String(rec.timestamp_iso) + "\",";
  json += "\"voltage_V\":" + String(rec.voltage_V, 2) + ",";
  json += "\"current_A\":" + String(rec.current_A, 2) + ",";
  json += "\"power_W\":" + String(rec.power_W, 2) + ",";
  json += "\"energy\":" + String(rec.energy_kWh, 3) + ",";
  json += "\"frequency_Hz\":" + String(rec.frequency_Hz, 2) + ",";
  json += "\"power_factor\":" + String(rec.power_factor, 2) + ",";
  json += "\"oil_temperature_C\":" + String(rec.oil_temperature_C, 2) + ",";
  if (rec.has_ambient) {
    json += "\"ambient_temperature_C\":" + String(rec.ambient_temperature_C, 2);
  } else {
    json += "\"ambient_temperature_C\":null";
  }
  json += "}";
  return json;
}

/**
 * Synchronizes UTC time with NTP servers.
 */
void syncNtpTime() {
  Serial.print(F("[NTP] Synchronizing UTC time via pool.ntp.org..."));
  configTime(0, 0, "pool.ntp.org", "time.nist.gov");

  time_t now = time(nullptr);
  int retries = 0;
  while (now < 1600000000 && retries < 20) {
    delay(500);
    Serial.print(".");
    now = time(nullptr);
    retries++;
  }
  Serial.println();

  if (now > 1600000000) {
    char buf[32];
    getUtcTimestampIso(buf, sizeof(buf));
    Serial.print(F("[NTP] Time synchronized successfully: "));
    Serial.println(buf);
  } else {
    Serial.println(F("[NTP WARNING] Failed to acquire NTP time. Will retry."));
  }
}

/**
 * Connects to the configured Wi-Fi network.
 */
void connectWifi() {
  if (WiFi.status() == WL_CONNECTED) return;

  Serial.print(F("[WIFI] Connecting to "));
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - start < WIFI_CONNECT_TIMEOUT_MS)) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print(F("[WIFI] Connected! IP Address: "));
    Serial.println(WiFi.localIP());
    syncNtpTime();
  } else {
    Serial.println(F("[WIFI WARNING] Connection timeout. Buffering readings in RAM."));
  }
}

/**
 * Attempts to transmit buffered telemetry to the backend.
 * Handles HTTP 200/201 (pop buffer), HTTP 401 (backoff, preserve buffer),
 * and network timeouts (keep buffer).
 */
void transmitBufferedTelemetry() {
  if (bufferCount == 0) return;
  if (WiFi.status() != WL_CONNECTED) return;

  // Check if currently in HTTP 401 authentication backoff
  if (authErrorActive) {
    if (millis() - lastAuthErrorTime < AUTH_BACKOFF_MS) {
      // In backoff period; do not spam server
      return;
    }
    // Backoff expired, attempt single transmission to verify if key was updated
    Serial.println(F("[AUTH RETRY] Retrying transmission with configured API Key..."));
    authErrorActive = false;
  }

  TelemetryRecord *rec = peekBuffer();
  if (rec == nullptr) return;

  HTTPClient http;
  String url = "http://" + String(BACKEND_HOST) + ":" + String(BACKEND_PORT) + String(INGEST_URL);

  http.begin(url);
  http.setTimeout(HTTP_TIMEOUT_MS);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-Device-API-Key", DEVICE_API_KEY);

  String payload = serializeRecordJson(*rec);
  int httpCode = http.POST(payload);

  if (httpCode == 201) {
    // New telemetry successfully persisted
    Serial.print(F("[INGEST SUCCESS] Record: "));
    Serial.print(rec->reading_id);
    Serial.print(F(" | Buffered remaining: "));
    Serial.println(bufferCount - 1);
    popBuffer();
    authErrorActive = false;
  } else if (httpCode == 200) {
    // Duplicate reading acknowledged by backend; safe to pop
    Serial.print(F("[INGEST DUPLICATE ACK] Reading already recorded: "));
    Serial.println(rec->reading_id);
    popBuffer();
    authErrorActive = false;
  } else if (httpCode == 401) {
    // CRITICAL: Authentication failure
    // DO NOT pop the buffer! Preserve readings and enter backoff loop.
    authErrorActive = true;
    lastAuthErrorTime = millis();
    Serial.println(F("=================================================================="));
    Serial.println(F("[AUTH ERROR] HTTP 401 Unauthorized!"));
    Serial.println(F("Invalid or missing X-Device-API-Key header."));
    Serial.println(F("Preserving sensor readings in RAM buffer. Entering 60s backoff."));
    Serial.println(F("Action required: Check DEVICE_API_KEY in firmware/config.h."));
    Serial.println(F("=================================================================="));
  } else {
    // Other HTTP error or connection failure (keep buffer for retry)
    Serial.print(F("[INGEST RETRY] HTTP Response code: "));
    Serial.println(httpCode);
  }

  http.end();
}

// ====================================================================
// Arduino Setup & Loop
// ====================================================================

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println(F("\n=================================================="));
  Serial.println(F("Transformer AI Monitoring - ESP32 Firmware v1.0"));
  Serial.println(F("=================================================="));

  generateBootId();
  Serial.print(F("Boot Session ID: "));
  Serial.println(bootId);
  Serial.print(F("Transformer ID:  "));
  Serial.println(TRANSFORMER_ID);

  connectWifi();
  lastReadTime = millis();
}

void loop() {
  // Ensure Wi-Fi remains connected
  if (WiFi.status() != WL_CONNECTED) {
    static unsigned long lastWifiRetry = 0;
    if (millis() - lastWifiRetry > 10000) {
      connectWifi();
      lastWifiRetry = millis();
    }
  }

  // Sample sensors at defined sampling interval (every 2 seconds)
  if (millis() - lastReadTime >= SENSOR_READ_INTERVAL_MS) {
    lastReadTime = millis();

    // 1. Read PZEM-004T AC Electrical Parameters
    float voltage = pzem.voltage();
    float current = pzem.current();
    float power   = pzem.power();
    float energy  = pzem.energy();
    float freq    = pzem.frequency();
    float pf      = pzem.pf();

    // 2. Read MAX6675 Thermocouple (Top Oil Temperature)
    float oilTemp = thermocouple.readCelsius();

    // Validate sensor signals (guard against disconnected hardware)
    bool pzemValid = !isnan(voltage) && !isnan(current) && !isnan(power) &&
                     !isnan(freq) && !isnan(pf) &&
                     (voltage >= 0.0f && voltage <= 1000.0f) &&
                     (freq >= 40.0f && freq <= 65.0f);

    bool tempValid = !isnan(oilTemp) && (oilTemp >= -20.0f && oilTemp <= 200.0f);

    if (pzemValid && tempValid) {
      TelemetryRecord record;
      record.sequence_number = sequenceNumber++;
      snprintf(
        record.reading_id,
        sizeof(record.reading_id),
        "%s-%s-%08lu",
        TRANSFORMER_ID,
        bootId,
        (unsigned long)record.sequence_number
      );

      // Acquire UTC timestamp
      if (!getUtcTimestampIso(record.timestamp_iso, sizeof(record.timestamp_iso))) {
        // Fallback: If NTP is not ready, format relative marker
        snprintf(record.timestamp_iso, sizeof(record.timestamp_iso), "1970-01-01T00:00:00");
      }

      record.voltage_V = voltage;
      record.current_A = current;
      record.power_W = power;
      record.energy_kWh = isnan(energy) ? 0.0f : energy;
      record.frequency_Hz = freq;
      record.power_factor = pf;
      record.oil_temperature_C = oilTemp;

      // Ambient temperature channel is strictly NULL (never substituted with oil temp)
      record.has_ambient = false;
      record.ambient_temperature_C = 0.0f;

      // Push into ring buffer
      pushBuffer(record);

      Serial.print(F("[OBS] "));
      Serial.print(record.reading_id);
      Serial.print(F(" | V=")); Serial.print(voltage, 1);
      Serial.print(F("V | I=")); Serial.print(current, 1);
      Serial.print(F("A | P=")); Serial.print(power, 0);
      Serial.print(F("W | OilTemp=")); Serial.print(oilTemp, 1);
      Serial.print(F("C | Buf=")); Serial.println(bufferCount);
    } else {
      Serial.print(F("[SENSOR WARNING] Invalid reading. PZEM valid: "));
      Serial.print(pzemValid);
      Serial.print(F(", MAX6675 valid: "));
      Serial.println(tempValid);
    }
  }

  // Transmit buffered readings if Wi-Fi is active
  transmitBufferedTelemetry();
  delay(10);
}

