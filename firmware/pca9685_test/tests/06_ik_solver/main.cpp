// ============================================================
//  TEST 06 — IK Solver Unit Test (No Hardware Required)
//  Purpose: Verify kinematics math matches the JavaScript
//           robot.js solver. Run on ESP32 or plain PC.
//
//  Expected output: known foot positions → known joint angles
//  Compare against simulation dashboard joint readouts.
// ============================================================
#include <Arduino.h>
#include <math.h>

// ---- Copy of the IK solver from kinematics.cpp ----
#define LC 30.0f
#define LF 90.0f
#define LT 90.0f
#define M_PI_F 3.14159265f

struct JointAngles { float coxa, femur, tibia; };

JointAngles solveIK(float outward, float forward, float down) {
    JointAngles out;
    float coxaRad = atan2f(forward, outward);
    out.coxa = coxaRad * 180.0f / M_PI_F;

    float r  = sqrtf(outward * outward + forward * forward);
    float xp = r - LC;
    float zp = down;
    float Ld = sqrtf(xp * xp + zp * zp);

    float maxR = (LF + LT) * 0.98f;
    float minR = fabsf(LF - LT) + 2.0f;
    if (Ld > maxR) Ld = maxR;
    if (Ld < minR) Ld = minR;

    float cosKnee = (LF*LF + LT*LT - Ld*Ld) / (2.0f * LF * LT);
    cosKnee = fmaxf(-1.0f, fminf(1.0f, cosKnee));
    out.tibia = acosf(cosKnee) * 180.0f / M_PI_F;

    float alpha  = atan2f(zp, xp);
    float cosHip = (LF*LF + Ld*Ld - LT*LT) / (2.0f * LF * Ld);
    cosHip = fmaxf(-1.0f, fminf(1.0f, cosHip));
    float beta   = acosf(cosHip);
    out.femur = (alpha - beta) * 180.0f / M_PI_F;

    // Convert to servo space (90° = neutral)
    out.coxa  = 90.0f + out.coxa;
    out.femur = 90.0f + out.femur;
    out.tibia = 180.0f - out.tibia;

    return out;
}

struct TestCase {
    const char* desc;
    float outward, forward, down;
    // Expected servo angles (±2° tolerance)
    float expCoxa, expFemur, expTibia;
};

void runTest(const TestCase& tc) {
    JointAngles a = solveIK(tc.outward, tc.forward, tc.down);
    bool pass = (fabsf(a.coxa - tc.expCoxa) < 2.5f &&
                 fabsf(a.femur - tc.expFemur) < 2.5f &&
                 fabsf(a.tibia - tc.expTibia) < 2.5f);

    Serial.printf("[%s] %-30s  coxa=%.1f° femur=%.1f° tibia=%.1f°  (exp %.1f %.1f %.1f)\n",
                  pass ? "PASS" : "FAIL",
                  tc.desc,
                  a.coxa, a.femur, a.tibia,
                  tc.expCoxa, tc.expFemur, tc.expTibia);
}

TestCase TESTS[] = {
    // Straight down, foot directly below shoulder (neutral stance)
    { "Straight neutral",  110.0f,  0.0f, 100.0f,   90.0f,  44.0f,  95.0f },
    // Forward reach
    { "Forward reach",     110.0f, 30.0f, 100.0f,  105.0f,  44.0f,  95.0f },
    // Backward reach
    { "Backward reach",    110.0f,-30.0f, 100.0f,   75.0f,  44.0f,  95.0f },
    // Leg fully stretched (test reachability clamping)
    { "Max reach",         170.0f,  0.0f,  10.0f,   90.0f,  -5.0f, 180.0f },
};

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("=== TEST 06: IK Solver Unit Test ===\n");
    Serial.println("Config: COXA=30mm FEMUR=90mm TIBIA=90mm\n");

    for (auto& tc : TESTS) {
        runTest(tc);
    }

    // ── Interactive mode ──────────────────────────────────────
    Serial.println("\n--- Interactive: send 'outward,forward,down' over Serial ---");
}

void loop() {
    if (Serial.available()) {
        String s = Serial.readStringUntil('\n');
        s.trim();
        float vals[3];
        int i = 0;
        char* tok = strtok((char*)s.c_str(), ",");
        while (tok && i < 3) { vals[i++] = atof(tok); tok = strtok(nullptr, ","); }
        if (i == 3) {
            JointAngles a = solveIK(vals[0], vals[1], vals[2]);
            Serial.printf("IK(%.1f, %.1f, %.1f) → coxa=%.2f° femur=%.2f° tibia=%.2f°\n",
                          vals[0], vals[1], vals[2], a.coxa, a.femur, a.tibia);
        }
    }
}
