// ============================================================
//  MAGI ESP32 — gait_engine.h
//  Bézier-arc crawl / trot gait generator
//  Matches the JavaScript gait.js implementation
// ============================================================
#pragma once
#include <Arduino.h>
#include "config.h"
#include "kinematics.h"
#include "servo_controller.h"

// Available gait types
enum GaitType {
    GAIT_STAND = 0,
    GAIT_CRAWL,
    GAIT_TROT
};

// Direction vector for omnidirectional movement
struct GaitCommand {
    float vx;        // forward velocity [-1, +1] (normalised)
    float vz;        // lateral velocity [-1, +1] (normalised, +right)
    float yawRate;   // turn rate        [-1, +1] (normalised, +right)
    float speed;     // overall speed    [0, 1]
};

class GaitEngine {
public:
    GaitEngine(ServoController& servo);

    // Call once in setup()
    void begin();

    // Call every loop() iteration — advances gait by dt seconds
    void update(float dt);

    // Set movement command
    void setCommand(const GaitCommand& cmd);

    // Set gait type
    void setGait(GaitType type);

    // Emergency stop — hold current pose, stop advancing
    void stop();

    // Return to home pose
    void goHome();

    // Get current gait phase [0, 1)
    float getPhase() const { return _phase; }

    // Get current leg angles (latest computed)
    const LegAngles* getLegAngles() const { return _angles; }

private:
    ServoController& _servo;

    GaitType _gait;
    GaitCommand _cmd;
    float _phase;        // gait cycle phase [0, 1)
    float _frequency;    // Hz
    float _strideLength; // mm
    float _stepHeight;   // mm
    float _globalX;      // body world X (mm)
    float _globalZ;      // body world Z (mm)
    float _bodyYaw;      // degrees

    LegAngles _angles[NUM_LEGS];
    bool _stopped;

    // Swing windows (phase start, phase end) per leg for each gait
    // Crawl: FR→FL→BL→BR sequential
    struct SwingWindow { float sw0, sw1; };
    static const SwingWindow CRAWL_SWING[NUM_LEGS];
    static const SwingWindow TROT_SWING[NUM_LEGS];

    // Bézier step height profile
    float bezierHeight(float t);

    // Compute foot world positions for current phase
    void computeFeet(float feet[NUM_LEGS][3]);  // [leg][x,y,z]

    // Solve IK and apply to servos
    void applyFeet(float feet[NUM_LEGS][3]);
};
