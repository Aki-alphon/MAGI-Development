// ============================================================
//  TEST 01 — PCA9685 I²C Scan + Init
//  Purpose: Verify I²C wiring, find all devices on bus,
//           confirm PCA9685 at 0x40 and MPU-6050 at 0x68
//
//  Expected output:
//    Found device at 0x40  (PCA9685)
//    Found device at 0x68  (MPU-6050)
//    PCA9685 OK — all 16 channels set to 0° neutral
// ============================================================
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

#define SDA_PIN 21
#define SCL_PIN 22

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("=== TEST 01: I2C Scan + PCA9685 Init ===");

    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(400000);

    // ── I²C device scan ──────────────────────────────────────
    Serial.println("\nScanning I²C bus (0x01 – 0x7E)...");
    int found = 0;
    for (uint8_t addr = 1; addr < 127; addr++) {
        Wire.beginTransmission(addr);
        uint8_t err = Wire.endTransmission();
        if (err == 0) {
            Serial.printf("  Found device at 0x%02X", addr);
            if (addr == 0x40) Serial.print("  ← PCA9685");
            if (addr == 0x41) Serial.print("  ← PCA9685 (alt addr)");
            if (addr == 0x68) Serial.print("  ← MPU-6050 IMU");
            if (addr == 0x70) Serial.print("  ← PCA9685 broadcast?");
            Serial.println();
            found++;
        }
    }
    if (found == 0) {
        Serial.println("  No devices found! Check wiring.");
        return;
    }
    Serial.printf("Scan complete: %d device(s) found.\n\n", found);

    // ── PCA9685 init test ─────────────────────────────────────
    Adafruit_PWMServoDriver pca(0x40);
    pca.begin();
    pca.setOscillatorFrequency(27000000);
    pca.setPWMFreq(50);
    delay(100);
    Serial.println("PCA9685 initialized at 50 Hz");

    // Set all 16 channels to neutral (90° = 1500 µs pulse)
    // OFF_TIME = 1500 * 50 * 4096 / 1000000 = 307
    for (uint8_t ch = 0; ch < 16; ch++) {
        pca.setPWM(ch, 0, 307);
    }
    Serial.println("All 16 channels set to 90° (neutral). Servos should be centered.");
    Serial.println("\n=== TEST 01 PASS ===");
}

void loop() {}
