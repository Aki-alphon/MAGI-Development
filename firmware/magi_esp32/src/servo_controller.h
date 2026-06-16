// ============================================================
//  MAGI ESP32 — servo_controller.h
//  PCA9685 servo abstraction layer
// ============================================================
#pragma once
#include <Adafruit_PWMServoDriver.h>
#include "config.h"

// Joint indices within a leg
enum Joint { COXA = 0, FEMUR = 1, TIBIA = 2 };

// Leg order
enum LegID { FR = 0, FL = 1, BR = 2, BL = 3 };

struct LegAngles {
    float coxa;   // degrees, 0–180
    float femur;  // degrees, 0–180
    float tibia;  // degrees, 0–180
};

class ServoController {
public:
    ServoController();

    // Call once in setup()
    bool begin();

    // Set all 12 joints at once from a 4-element LegAngles array
    void setAllLegs(const LegAngles angles[NUM_LEGS]);

    // Set a single leg
    void setLeg(LegID leg, const LegAngles& a);

    // Set a single joint by channel number (0-11)
    void setChannel(uint8_t ch, float degrees);

    // Move all servos to home (neutral) position
    void goHome();

    // Disable all servo PWM (let them freewheel — power saving)
    void disable();

    // Convert degrees to PCA9685 OFF_TIME count
    static uint16_t degreesToTick(float deg);

    // Convert degrees to pulse width in microseconds
    static uint16_t degreesToUs(float deg);

private:
    Adafruit_PWMServoDriver _pca;
    bool _enabled;
};
