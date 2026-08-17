// ============================================================
//  TEST 07 — Full Crawl Gait on Real Hardware
//  Purpose: End-to-end hardware test of the complete system
//           Robot crawls forward for 5 steps, pauses, crawls back
//
//  Prerequisites:
//    ✅ TEST 01 passed (PCA9685 found)
//    ✅ TEST 02 passed (servo calibration OK)
//    ✅ TEST 03 passed (all 12 servos respond)
//    ✅ TEST 06 passed (IK solver correct)
//    ✅ Robot assembled and on flat surface
//    ✅ Battery fully charged (≥11.1V for 3S)
//    ✅ Buck converter set to 6.0V
//
//  ⚠ Keep hand near power switch — robot WILL move!
// ============================================================
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <math.h>

#define SDA_PIN   21
#define SCL_PIN   22
#define NUM_LEGS  4
#define DOF       3

// Link lengths (mm)
#define LC 30.0f
#define LF 90.0f
#define LT 90.0f
#define M_PIf 3.14159265f

// Gait parameters
#define STEP_HZ      1.0f    // gait cycles per second
#define STRIDE_MM    50.0f   // foot travel per cycle
#define STEP_H_MM    25.0f   // foot lift height
#define BODY_H_MM    100.0f  // body above ground

Adafruit_PWMServoDriver pca(0x40);

// ---- Servo output ----
uint16_t angleTick(float deg) {
    float us = 500.0f + (deg / 180.0f) * 2000.0f;
    return (uint16_t)(us * 50.0f * 4096.0f / 1000000.0f);
}

void setJoint(uint8_t leg, uint8_t joint, float deg) {
    pca.setPWM(leg * DOF + joint, 0, angleTick(deg));
}

// ---- IK ----
struct Angles { float coxa, femur, tibia; };

Angles solveIK(float outward, float forward, float down) {
    Angles a;
    a.coxa = atan2f(forward, outward) * 180.0f / M_PIf;
    float r = sqrtf(outward*outward + forward*forward);
    float xp = r - LC, zp = down;
    float Ld = sqrtf(xp*xp + zp*zp);
    Ld = fmaxf(fabsf(LF-LT)+2.0f, fminf((LF+LT)*0.98f, Ld));
    float cosK = (LF*LF + LT*LT - Ld*Ld) / (2.0f*LF*LT);
    a.tibia = acosf(fmaxf(-1.0f, fminf(1.0f, cosK))) * 180.0f / M_PIf;
    float alpha = atan2f(zp,xp);
    float cosH = (LF*LF + Ld*Ld - LT*LT) / (2.0f*LF*Ld);
    a.femur = (alpha - acosf(fmaxf(-1.0f, fminf(1.0f, cosH)))) * 180.0f / M_PIf;
    a.coxa  = 90.0f + a.coxa;
    a.femur = 90.0f + a.femur;
    a.tibia = 180.0f - a.tibia;
    return a;
}

// Neutral foot positions (offset from body centre, mm)
const float NEU[4][2] = {
    {  120.0f,  140.0f },   // FR
    {  120.0f, -140.0f },   // FL
    { -120.0f,  140.0f },   // BR
    { -120.0f, -140.0f },   // BL
};

const float SHOULDER[4][2] = {
    {  80.0f,  40.0f },  // FR
    {  80.0f, -40.0f },  // FL
    { -80.0f,  40.0f },  // BR
    { -80.0f, -40.0f },  // BL
};

const int SIDE[4] = { 1, -1, 1, -1 };

void setFoot(int leg, float footX, float footZ, float footY_world) {
    // foot relative to shoulder
    float dx = footX - SHOULDER[leg][0];
    float dz = footZ - SHOULDER[leg][1];
    float dy = footY_world;  // negative = below shoulder

    float outward = fabsf(dz);
    float forward = dx;
    float down    = -dy;

    Angles a = solveIK(outward, forward, down);
    setJoint(leg, 0, a.coxa);
    setJoint(leg, 1, a.femur);
    setJoint(leg, 2, a.tibia);
}

void goHome() {
    for (int leg = 0; leg < NUM_LEGS; leg++) {
        setFoot(leg, NEU[leg][0], NEU[leg][1], -BODY_H_MM);
    }
    Serial.println("[GAIT] Home pose");
}

// Crawl sequence: FR → FL → BL → BR
const int CRAWL_ORDER[] = { 0, 1, 3, 2 };

void crawlStep(int stepLeg, float strideX, float dir) {
    // Other 3 legs stay put (stance), chosen leg swings
    Serial.printf("[GAIT] Step leg %d dir=%.0f\n", stepLeg, dir * 90.0f);

    int steps = 30;
    for (int s = 0; s <= steps; s++) {
        float t = (float)s / steps;
        for (int leg = 0; leg < NUM_LEGS; leg++) {
            float fX = NEU[leg][0];
            float fZ = NEU[leg][1];
            float fY = -BODY_H_MM;

            if (leg == stepLeg) {
                // Bézier arc: start back, swing forward
                fX = NEU[leg][0] + strideX * dir * (t - 0.5f) * 2.0f;
                float arc = -4.0f * STEP_H_MM * t * (t - 1.0f);  // parabola up
                fY = -BODY_H_MM + arc;
            } else {
                // Stance: body drifts forward
                fX = NEU[leg][0] - strideX * dir * (t - 0.5f) * 0.33f;
            }

            setFoot(leg, fX, fZ, fY);
        }
        delay((int)(1000.0f / STEP_HZ / steps));
    }
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("=== TEST 07: Full Crawl Gait ===");
    Serial.println("Robot will crawl forward 4 steps then stop.\n");

    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(400000);
    pca.begin();
    pca.setOscillatorFrequency(27000000);
    pca.setPWMFreq(50);
    delay(100);

    // Home position
    goHome();
    delay(2000);
    Serial.println("[GAIT] Starting crawl — 4 steps forward...");

    // 4 crawl steps forward
    for (int step = 0; step < 4; step++) {
        int swingLeg = CRAWL_ORDER[step % 4];
        crawlStep(swingLeg, STRIDE_MM, 1.0f);
        delay(200);
    }

    Serial.println("[GAIT] Pause 2s...");
    goHome();
    delay(2000);

    Serial.println("[GAIT] 4 steps backward...");
    for (int step = 0; step < 4; step++) {
        int swingLeg = CRAWL_ORDER[step % 4];
        crawlStep(swingLeg, STRIDE_MM, -1.0f);
        delay(200);
    }

    goHome();
    delay(1000);
    Serial.println("\n=== TEST 07 COMPLETE ===");
}

void loop() {}
