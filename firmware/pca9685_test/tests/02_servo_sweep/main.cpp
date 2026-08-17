// ============================================================
//  TEST 02 — Single Servo Full Range Sweep
//  Purpose: Verify one servo moves smoothly from 0° to 180°
//           Checks calibration of SERVO_US_MIN / SERVO_US_MAX
//
//  ⚠ WARNING: Run with ONE servo connected only!
//             Start with CH=0 (FR Coxa) to verify range.
//
//  Expected output:
//    Servo ch0 → 0°  (servo at minimum)
//    Servo ch0 → 90° (servo centered)
//    Servo ch0 → 180° (servo at maximum)
//    Repeating sweep 0° → 180° → 0°...
// ============================================================
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

#define SDA_PIN      21
#define SCL_PIN      22
#define TEST_CHANNEL 0     // PCA9685 channel to test
#define SERVO_FREQ   50
#define US_MIN       500   // 0°
#define US_MAX       2500  // 180°

Adafruit_PWMServoDriver pca(0x40);

uint16_t angleToPulse(float deg) {
    float us = US_MIN + (deg / 180.0f) * (US_MAX - US_MIN);
    return (uint16_t)(us * SERVO_FREQ * 4096.0f / 1000000.0f);
}

void setAngle(float deg) {
    uint16_t tick = angleToPulse(deg);
    pca.setPWM(TEST_CHANNEL, 0, tick);
    Serial.printf("  ch%d → %.0f° (tick=%d)\n", TEST_CHANNEL, deg, tick);
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.printf("=== TEST 02: Servo Sweep — Channel %d ===\n", TEST_CHANNEL);

    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(400000);
    pca.begin();
    pca.setOscillatorFrequency(27000000);
    pca.setPWMFreq(SERVO_FREQ);
    delay(100);

    // ── One-time calibration check ────────────────────────────
    Serial.println("\nCalibration check:");
    setAngle(0);    delay(2000);
    setAngle(90);   delay(2000);
    setAngle(180);  delay(2000);
    setAngle(90);   delay(1000);

    Serial.println("\nStarting continuous sweep (Ctrl+C to stop)...");
}

void loop() {
    // Sweep 0° → 180°
    for (float a = 0; a <= 180; a += 2.0f) {
        setAngle(a);
        delay(20);
    }
    // Sweep 180° → 0°
    for (float a = 180; a >= 0; a -= 2.0f) {
        setAngle(a);
        delay(20);
    }
}
