// ============================================================
//  TEST 05 — UART Loopback: ESP32 ↔ Raspberry Pi
//  Purpose: Verify UART2 TX/RX wiring between ESP32 and Pi
//
//  On the Pi, run: python3 uart_test.py /dev/ttyS0 115200
//  (see tests/05_uart_loopback/uart_test.py)
//
//  Expected behaviour:
//    - ESP32 sends "PING\n" every 1 second
//    - Pi responds with "PONG\n"
//    - ESP32 prints received echoes to USB Serial (Serial)
//    - Round-trip latency should be < 5ms
// ============================================================
#include <Arduino.h>

#define RX_PIN   16
#define TX_PIN   17
#define BAUD     115200

HardwareSerial PiSerial(2);  // UART2

char rxBuf[128];
int rxLen = 0;
uint32_t pingTimer = 0;
uint32_t pingCount = 0;
uint32_t pongCount = 0;

void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("=== TEST 05: UART Loopback ===");
    Serial.printf("Serial2 RX=GPIO%d TX=GPIO%d @ %d baud\n", RX_PIN, TX_PIN, BAUD);

    PiSerial.begin(BAUD, SERIAL_8N1, RX_PIN, TX_PIN);
    pingTimer = millis();
    Serial.println("Waiting for Pi response...\n");
}

void loop() {
    // ── Send PING every 1 second ──────────────────────────────
    if (millis() - pingTimer >= 1000) {
        pingTimer = millis();
        pingCount++;
        PiSerial.printf("PING %lu\n", pingCount);
        Serial.printf("[TX] PING %lu\n", pingCount);
    }

    // ── Receive any bytes from Pi ─────────────────────────────
    while (PiSerial.available()) {
        char c = PiSerial.read();
        if (c == '\n') {
            rxBuf[rxLen] = '\0';
            if (strncmp(rxBuf, "PONG", 4) == 0) {
                pongCount++;
                Serial.printf("[RX] %s  (success %lu/%lu)\n", rxBuf, pongCount, pingCount);
            } else {
                Serial.printf("[RX] %s\n", rxBuf);
            }
            rxLen = 0;
        } else if (rxLen < 126) {
            rxBuf[rxLen++] = c;
        }
    }
}
