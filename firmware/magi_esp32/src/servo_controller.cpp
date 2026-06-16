// ============================================================
//  MAGI ESP32 — servo_controller.cpp
//  PCA9685 servo driver implementation
// ============================================================
#include "servo_controller.h"
#include <Arduino.h>
#include <Wire.h>

ServoController::ServoController()
    : _pca(PCA9685_ADDR), _enabled(false) {}

bool ServoController::begin() {
    // Wire must be initialized before this
    _pca.begin();
    _pca.setOscillatorFrequency(27000000);  // trim to your board's crystal
    _pca.setPWMFreq(SERVO_FREQ_HZ);
    delay(10);
    _enabled = true;
    Serial.println("[SERVO] PCA9685 initialized at 50 Hz");
    return true;
}

// Convert angle (0–180°) to PCA9685 tick count
// Formula from report: OFF_TIME = tpulse[µs] × 50 × 4096 / 1e6
uint16_t ServoController::degreesToTick(float deg) {
    float us = degreesToUs(deg);
    // OFF_TIME = us * SERVO_FREQ_HZ * 4096 / 1000000
    return (uint16_t)((us * SERVO_FREQ_HZ * 4096.0f) / 1000000.0f);
}

uint16_t ServoController::degreesToUs(float deg) {
    // Clamp
    if (deg < 0.0f) deg = 0.0f;
    if (deg > 180.0f) deg = 180.0f;
    // Linear interpolation between min and max pulse
    return (uint16_t)(SERVO_US_MIN + (deg / 180.0f) * (SERVO_US_MAX - SERVO_US_MIN));
}

void ServoController::setChannel(uint8_t ch, float degrees) {
    if (!_enabled || ch >= 16) return;
    uint16_t tick = degreesToTick(degrees);
    _pca.setPWM(ch, 0, tick);
}

void ServoController::setLeg(LegID leg, const LegAngles& a) {
    uint8_t base = (uint8_t)leg * DOF_PER_LEG;
    setChannel(base + COXA,  a.coxa);
    setChannel(base + FEMUR, a.femur);
    setChannel(base + TIBIA, a.tibia);
}

void ServoController::setAllLegs(const LegAngles angles[NUM_LEGS]) {
    for (int i = 0; i < NUM_LEGS; i++) {
        setLeg((LegID)i, angles[i]);
    }
}

void ServoController::goHome() {
    LegAngles home = { HOME_COXA, HOME_FEMUR, HOME_TIBIA };
    LegAngles allHome[NUM_LEGS] = { home, home, home, home };
    setAllLegs(allHome);
    Serial.println("[SERVO] All servos → HOME");
}

void ServoController::disable() {
    for (uint8_t ch = 0; ch < NUM_SERVOS; ch++) {
        _pca.setPWM(ch, 0, 0);  // 0 duty = no pulse = servo released
    }
    _enabled = false;
    Serial.println("[SERVO] PWM disabled (freewheel)");
}
