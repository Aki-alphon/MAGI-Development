// ============================================================
//  MAGI ESP32 — main.cpp
//  Entry point: setup() + loop()
//  
//  Hardware: ESP32 DevKit V1
//    GPIO21/22 — I²C SDA/SCL (PCA9685 + MPU-6050)
//    GPIO16/17 — UART2 RX/TX (Raspberry Pi 4B)
//    GPIO2     — Built-in LED (status)
//
//  Loop structure (target 100 Hz):
//    ├── IMU read + filter
//    ├── UART receive + parse commands from Pi
//    ├── Gait engine update (advance phase, compute IK)
//    ├── Servo update (sent to PCA9685 via I²C)
//    └── Serial debug @ 10 Hz
// ============================================================
#include <Arduino.h>
#include <Wire.h>

#include "config.h"
#include "servo_controller.h"
#include "kinematics.h"
#include "gait_engine.h"
#include "imu_driver.h"
#include "uart_protocol.h"

// ---- Global objects ----------------------------------------
ServoController servo;
GaitEngine      gait(servo);
ImuDriver       imu;
UartProtocol    uart(gait, imu);

// ---- Timing ------------------------------------------------
static uint32_t lastLoopUs = 0;
static uint32_t debugTimerMs = 0;
static uint32_t imuTimerUs = 0;
static float    dt = 0.01f;  // seconds

// ---- LED state ---------------------------------------------
static bool ledState = false;
static uint32_t ledBlinkMs = 0;

// ---- Forward declarations ----------------------------------
void blinkLed(uint32_t intervalMs);
void printStatus();

// ============================================================
void setup() {
    // Debug serial (USB)
    Serial.begin(115200);
    delay(500);
    Serial.println("\n╔══════════════════════════════════╗");
    Serial.println("║  MAGI ESP32 Firmware  v1.0       ║");
    Serial.println("║  12-DOF Quadruped Controller     ║");
    Serial.println("╚══════════════════════════════════╝");

    // LED
    pinMode(LED_BUILTIN_PIN, OUTPUT);
    digitalWrite(LED_BUILTIN_PIN, HIGH);

    // I²C bus
    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(I2C_FREQ);
    Serial.printf("[I2C] Bus started at %lu Hz on SDA=%d SCL=%d\n",
                  I2C_FREQ, SDA_PIN, SCL_PIN);

    // PCA9685 servo driver
    if (!servo.begin()) {
        Serial.println("[FATAL] PCA9685 init failed — check wiring!");
        while (true) { blinkLed(100); }  // rapid blink = fatal error
    }

    // Move to home pose
    servo.goHome();
    delay(500);

    // MPU-6050 IMU
    if (!imu.begin()) {
        Serial.println("[WARN] IMU not found — continuing without IMU");
    } else {
        Serial.println("[IMU] Calibrating zero offsets...");
        imu.calibrate(500);
    }

    // UART to Raspberry Pi
    uart.begin();

    // Gait engine
    gait.begin();
    gait.setGait(GAIT_CRAWL);

    // Set default forward crawl command
    GaitCommand defaultCmd;
    defaultCmd.vx = 1.0f;
    defaultCmd.vz = 0.0f;
    defaultCmd.yawRate = 0.0f;
    defaultCmd.speed = 0.5f;
    gait.setCommand(defaultCmd);

    lastLoopUs  = micros();
    imuTimerUs  = micros();
    debugTimerMs = millis();

    Serial.println("[MAIN] Setup complete — entering main loop");
    digitalWrite(LED_BUILTIN_PIN, LOW);  // LED off = ready
}

// ============================================================
void loop() {
    uint32_t now = micros();
    dt = (now - lastLoopUs) / 1e6f;
    if (dt < 0.001f) dt = 0.001f;
    if (dt > 0.05f)  dt = 0.05f;  // cap at 50 ms (20 Hz min)
    lastLoopUs = now;

    // ── IMU update @ 100 Hz ──────────────────────────────────
    if ((now - imuTimerUs) >= (1000000UL / IMU_SAMPLE_HZ)) {
        imu.update(dt);
        imuTimerUs = now;
    }

    // ── UART command processing ───────────────────────────────
    uart.update();

    // ── Gait engine tick ──────────────────────────────────────
    gait.update(dt);

    // ── Debug print @ 10 Hz ──────────────────────────────────
    uint32_t nowMs = millis();
    if (nowMs - debugTimerMs >= 100) {
        debugTimerMs = nowMs;
        printStatus();
    }

    // ── LED heartbeat (0.5 Hz) ────────────────────────────────
    blinkLed(1000);

    // Target loop rate ~100 Hz
    // PCA9685 I²C burst takes ~2–3 ms, so headroom is limited
}

// ============================================================
void blinkLed(uint32_t intervalMs) {
    uint32_t t = millis();
    if (t - ledBlinkMs >= intervalMs) {
        ledBlinkMs = t;
        ledState = !ledState;
        digitalWrite(LED_BUILTIN_PIN, ledState ? HIGH : LOW);
    }
}

void printStatus() {
    const LegAngles* a = gait.getLegAngles();
    Serial.printf("[DBG] ph=%.2f IMU pitch=%.1f° roll=%.1f° | "
                  "FR[%.0f,%.0f,%.0f] FL[%.0f,%.0f,%.0f]\n",
                  gait.getPhase(),
                  imu.data().pitch, imu.data().roll,
                  a[0].coxa, a[0].femur, a[0].tibia,
                  a[1].coxa, a[1].femur, a[1].tibia);
}
