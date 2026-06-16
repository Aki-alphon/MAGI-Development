// ============================================================
//  MAGI ESP32 — kinematics.h
//  3-DOF Analytical Inverse Kinematics Solver
//  Spider/Crab configuration — legs point OUTWARD from body
// ============================================================
#pragma once
#include <math.h>
#include "config.h"
#include "servo_controller.h"

// 3D foot position relative to shoulder mount
struct FootPos {
    float x;   // forward/back from shoulder (mm) — +forward
    float y;   // up/down from shoulder (mm)       — +up
    float z;   // lateral from shoulder (mm)        — always positive (outward)
};

// Body pose
struct BodyPose {
    float pitch;  // degrees — nose up positive
    float roll;   // degrees — right side down positive
    float yaw;    // degrees — turn right positive
    float height; // mm above ground
};

class Kinematics {
public:
    // Solve IK for one leg: returns joint angles in degrees
    // outward = lateral distance from shoulder (positive)
    // forward = forward/back from shoulder
    // down    = distance below shoulder (positive = below shoulder)
    // Returns false if position is unreachable
    static bool solveIK(float outward, float forward, float down,
                        LegAngles& out);

    // Solve IK from a foot position relative to shoulder
    static bool solveFromFoot(const FootPos& foot, LegAngles& out);

    // Compute foot position for a given body pose and neutral stance
    // leg: 0=FR,1=FL,2=BR,3=BL  globalFootXZ: world X,Z of target foot
    static FootPos computeFootLocal(int legIndex,
                                    float globalFootX, float globalFootZ,
                                    float globalBodyX, float globalBodyZ,
                                    float globalBodyY);

    // Get shoulder mount position for a given leg (chassis-local, mm)
    static void getShoulderLocal(int legIndex, float& sx, float& sy, float& sz);

    // Clamp angle to servo physical limits
    static float clampAngle(float deg);

private:
    // Link lengths from config.h
    static constexpr float LC = LINK_COXA_MM;
    static constexpr float LF = LINK_FEMUR_MM;
    static constexpr float LT = LINK_TIBIA_MM;
};
