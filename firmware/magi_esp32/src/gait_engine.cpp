// ============================================================
//  MAGI ESP32 — gait_engine.cpp
//  Bézier-arc crawl/trot gait with IK leg placement
// ============================================================
#include "gait_engine.h"
#include <math.h>
#include <Arduino.h>

#ifndef M_PI
#define M_PI 3.14159265358979f
#endif

// ------------------------------------------------------------
// Crawl gait: one leg lifts at a time (static stability)
// Sequence: FR(0), FL(1), BL(3), BR(2)
// Duty cycle ~75% (swing = 25% of period)
const GaitEngine::SwingWindow GaitEngine::CRAWL_SWING[NUM_LEGS] = {
    { 0.00f, 0.25f },  // FR
    { 0.25f, 0.50f },  // FL
    { 0.50f, 0.75f },  // BL
    { 0.75f, 1.00f },  // BR
};

// Trot gait: diagonal pairs lift simultaneously
// FR+BL together, then FL+BR together
const GaitEngine::SwingWindow GaitEngine::TROT_SWING[NUM_LEGS] = {
    { 0.00f, 0.40f },  // FR
    { 0.50f, 0.90f },  // FL
    { 0.50f, 0.90f },  // BR
    { 0.00f, 0.40f },  // BL
};

// ------------------------------------------------------------
GaitEngine::GaitEngine(ServoController& servo)
    : _servo(servo)
    , _gait(GAIT_STAND)
    , _phase(0.0f)
    , _frequency(GAIT_STEP_HZ_DEFAULT)
    , _strideLength(GAIT_STRIDE_MM_DEFAULT)
    , _stepHeight(GAIT_STEP_HEIGHT_MM)
    , _globalX(0.0f)
    , _globalZ(0.0f)
    , _bodyYaw(0.0f)
    , _stopped(false) {
    _cmd = { 1.0f, 0.0f, 0.0f, 0.5f };
    for (int i = 0; i < NUM_LEGS; i++) {
        _angles[i] = { HOME_COXA, HOME_FEMUR, HOME_TIBIA };
    }
}

void GaitEngine::begin() {
    goHome();
}

// Cubic Bézier arc — smooth foot lift and lower
float GaitEngine::bezierHeight(float t) {
    // Control points: 0, 0.15*H, 1.1*H, 0
    float u  = 1.0f - t;
    float h0 = 0.0f;
    float h1 = _stepHeight * 0.15f;
    float h2 = _stepHeight * 1.1f;
    float h3 = 0.0f;
    return u*u*u*h0 + 3*u*u*t*h1 + 3*u*t*t*h2 + t*t*t*h3;
}

void GaitEngine::setCommand(const GaitCommand& cmd) {
    _cmd = cmd;
    _frequency = GAIT_STEP_HZ_DEFAULT * (0.5f + cmd.speed * 1.5f);
    _strideLength = GAIT_STRIDE_MM_DEFAULT * cmd.speed;
}

void GaitEngine::setGait(GaitType type) {
    _gait = type;
    _phase = 0.0f;
    if (type == GAIT_STAND) goHome();
}

void GaitEngine::stop() {
    _stopped = true;
}

void GaitEngine::goHome() {
    _servo.goHome();
    for (int i = 0; i < NUM_LEGS; i++) {
        _angles[i] = { HOME_COXA, HOME_FEMUR, HOME_TIBIA };
    }
}

// Neutral foot positions relative to body centre (world-frame offsets in mm)
static const float NEUTRAL_FOOT[4][2] = {
    {  120.0f,  140.0f },  // FR: forward, right
    {  120.0f, -140.0f },  // FL: forward, left
    { -120.0f,  140.0f },  // BR: back,    right
    { -120.0f, -140.0f },  // BL: back,    left
};

void GaitEngine::computeFeet(float feet[NUM_LEGS][3]) {
    const SwingWindow* windows = (_gait == GAIT_TROT) ? TROT_SWING : CRAWL_SWING;
    const float groundY = -(BODY_HEIGHT_MM);  // relative to shoulder

    // Actual stride in each axis
    float strideX = _strideLength * _cmd.vx;
    float strideZ = _strideLength * _cmd.vz;

    for (int i = 0; i < NUM_LEGS; i++) {
        // Neutral foot in world frame
        float nfX = _globalX + NEUTRAL_FOOT[i][0];
        float nfZ = _globalZ + NEUTRAL_FOOT[i][1];

        const SwingWindow& w = windows[i];
        float ph = _phase;
        bool isSwing;
        float legPhase;

        if (w.sw1 > w.sw0) {
            isSwing  = (ph >= w.sw0 && ph < w.sw1);
            legPhase = isSwing ? (ph - w.sw0) / (w.sw1 - w.sw0) : -1.0f;
        } else {
            // Wrapping swing window
            isSwing = (ph >= w.sw0 || ph < w.sw1);
            if (isSwing) {
                legPhase = (ph >= w.sw0)
                    ? (ph - w.sw0) / (1.0f - w.sw0 + w.sw1)
                    : (ph + 1.0f - w.sw0) / (1.0f - w.sw0 + w.sw1);
            } else {
                legPhase = -1.0f;
            }
        }

        float footX, footZ, footY;

        if (isSwing) {
            // Swing: arc from back to front using linear lerp + Bézier height
            footX = nfX - strideX * 0.5f + strideX * legPhase;
            footZ = nfZ - strideZ * 0.5f + strideZ * legPhase;
            footY = groundY + bezierHeight(legPhase);
        } else {
            // Stance: foot pushes backward relative to body motion
            float swingDur   = (w.sw1 > w.sw0) ? (w.sw1 - w.sw0) : (1.0f - w.sw0 + w.sw1);
            float stanceDur  = 1.0f - swingDur;
            float stancePhase;
            if (w.sw1 > w.sw0) {
                stancePhase = (ph >= w.sw1) ? (ph - w.sw1) / stanceDur : (ph + 1.0f - w.sw1) / stanceDur;
            } else {
                stancePhase = (ph >= w.sw1 && ph < w.sw0) ? (ph - w.sw1) / stanceDur : 0.0f;
            }
            stancePhase = fmaxf(0.0f, fminf(1.0f, stancePhase));

            footX = nfX + strideX * 0.5f - strideX * stancePhase;
            footZ = nfZ + strideZ * 0.5f - strideZ * stancePhase;
            footY = groundY;
        }

        feet[i][0] = footX;
        feet[i][1] = footY;
        feet[i][2] = footZ;
    }
}

void GaitEngine::applyFeet(float feet[NUM_LEGS][3]) {
    for (int i = 0; i < NUM_LEGS; i++) {
        float sx, sy, sz;
        Kinematics::getShoulderLocal(i, sx, sy, sz);

        // Convert world foot to shoulder-relative
        FootPos fp;
        fp.x = (feet[i][0] - _globalX) - sx;
        fp.y = feet[i][1] - sy;
        fp.z = (feet[i][2] - _globalZ) - sz;

        LegAngles ang;
        if (Kinematics::solveFromFoot(fp, ang)) {
            _angles[i] = ang;
            _servo.setLeg((LegID)i, ang);
        }
    }
}

void GaitEngine::update(float dt) {
    if (_stopped || _gait == GAIT_STAND) return;

    // Advance phase
    _phase += _frequency * dt;
    if (_phase >= 1.0f) _phase -= 1.0f;

    // Advance world position
    _globalX += _strideLength * _cmd.vx * _frequency * dt;
    _globalZ += _strideLength * _cmd.vz * _frequency * dt;
    _bodyYaw += _cmd.yawRate * 30.0f * dt;  // 30 deg/s max turn

    // Compute and apply feet
    float feet[NUM_LEGS][3];
    computeFeet(feet);
    applyFeet(feet);
}
