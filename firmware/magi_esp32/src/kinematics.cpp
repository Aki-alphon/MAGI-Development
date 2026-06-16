// ============================================================
//  MAGI ESP32 — kinematics.cpp
//  3-DOF Analytical IK — Spider/Crab configuration
//  Matches the JavaScript IK solver in robot.js exactly
// ============================================================
#include "kinematics.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

static inline float RAD2DEG(float r) { return r * (180.0f / M_PI); }
static inline float DEG2RAD(float d) { return d * (M_PI / 180.0f); }

// Shoulder positions in chassis-local frame (mm)
// Same as shoulders[] array in robot.js
static const float SHOULDER[4][3] = {
    {  SHOULDER_X_FRONT, 0,  SHOULDER_Z_RIGHT },  // FR
    {  SHOULDER_X_FRONT, 0,  SHOULDER_Z_LEFT  },  // FL
    {  SHOULDER_X_BACK,  0,  SHOULDER_Z_RIGHT },  // BR
    {  SHOULDER_X_BACK,  0,  SHOULDER_Z_LEFT  },  // BL
};

// Side: +1 = right (+Z), -1 = left (-Z)
static const int SIDE[4] = { 1, -1, 1, -1 };

bool Kinematics::solveIK(float outward, float forward, float down,
                          LegAngles& out) {
    // --- Coxa: horizontal rotation to point foot ---
    float coxaRad = atan2f(forward, outward);
    out.coxa = RAD2DEG(coxaRad);

    // --- Project into sagittal plane ---
    float r  = sqrtf(outward * outward + forward * forward);
    float xp = r - LC;     // horizontal distance after coxa
    float zp = down;        // vertical distance (positive = down)

    float Ld = sqrtf(xp * xp + zp * zp);

    // Clamp to reachable range
    float maxR = (LF + LT) * 0.98f;
    float minR = fabsf(LF - LT) + 2.0f;
    if (Ld > maxR) Ld = maxR;
    if (Ld < minR) Ld = minR;

    // --- Tibia (knee) — law of cosines ---
    float cosKnee = (LF * LF + LT * LT - Ld * Ld) / (2.0f * LF * LT);
    cosKnee = fmaxf(-1.0f, fminf(1.0f, cosKnee));
    out.tibia = RAD2DEG(acosf(cosKnee));

    // --- Femur (hip vertical swing) ---
    float alpha   = atan2f(zp, xp);
    float cosHip  = (LF * LF + Ld * Ld - LT * LT) / (2.0f * LF * Ld);
    cosHip = fmaxf(-1.0f, fminf(1.0f, cosHip));
    float beta    = acosf(cosHip);
    out.femur = RAD2DEG(alpha - beta);

    // Shift angles into servo frame (servo 0° = neutral)
    // The simulation uses signed IK angles directly as offsets
    // from neutral. We need to convert to absolute servo angles:
    //   servo_angle = 90 + ik_angle   (for femur/tibia offsets)
    // Coxa: servo 90° = straight out (no swing), ±30° swing range
    out.coxa  = clampAngle(90.0f + out.coxa);
    out.femur = clampAngle(90.0f + out.femur);
    // Tibia: supplementary angle (knee bends inward)
    out.tibia = clampAngle(180.0f - out.tibia);

    return true;  // solver always produces a result (clamped if needed)
}

bool Kinematics::solveFromFoot(const FootPos& foot, LegAngles& out) {
    float outward = fabsf(foot.z);  // lateral (always positive for IK)
    float forward = foot.x;
    float down    = -foot.y;        // foot.y positive = up, IK wants down
    return solveIK(outward, forward, down, out);
}

FootPos Kinematics::computeFootLocal(int legIndex,
                                     float globalFootX, float globalFootZ,
                                     float globalBodyX, float globalBodyZ,
                                     float globalBodyY) {
    float sx = SHOULDER[legIndex][0];
    float sz = SHOULDER[legIndex][2];

    // Foot position relative to shoulder in chassis frame
    FootPos fp;
    fp.x = (globalFootX - globalBodyX) - sx;
    fp.z = (globalFootZ - globalBodyZ) - sz;
    fp.y = -globalBodyY;  // below shoulder = negative y in foot frame
    return fp;
}

void Kinematics::getShoulderLocal(int legIndex, float& sx, float& sy, float& sz) {
    sx = SHOULDER[legIndex][0];
    sy = SHOULDER[legIndex][1];
    sz = SHOULDER[legIndex][2];
}

float Kinematics::clampAngle(float deg) {
    if (deg < 0.0f)   deg = 0.0f;
    if (deg > 180.0f) deg = 180.0f;
    return deg;
}
