"""
MAGI OS — Locomotion Kinematics
src/locomotion/kinematics.py

Implements Forward and Inverse Kinematics (IK) for the 3-DOF quadruped leg.
Maps Cartesian coordinates (x, y, z) in the leg shoulder frame to joint angles
(coxa, femur, tibia) and translates them to PCA9685 PWM duty cycles.
"""

import math
from typing import Tuple, Dict

class LegKinematics:
    # Physical dimensions of the MAGI leg links (in mm)
    L_COXA = 30.0    # Coxa offset length
    L_FEMUR = 90.0   # Femur link length
    L_TIBIA = 90.0   # Tibia link length

    # Servo PWM mapping configurations
    # MG996R servo limits: 0° to 180° mapped to 500µs to 2500µs pulse duration
    PWM_MIN_US = 500
    PWM_MAX_US = 2500
    PWM_FREQ_HZ = 50
    PCA_RESOLUTION = 4096

    def __init__(self, leg_id: int, is_left: bool):
        self.leg_id = leg_id
        self.is_left = is_left

    def solve_ik(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Solves Inverse Kinematics for a single foot target (x, y, z) relative
        to the leg shoulder joint pivot.
        
        Coordinates:
          +X: Forward, -X: Backward
          +Y: Outward lateral (relative to side), -Y: Inward lateral
          +Z: Upward vertical, -Z: Downward vertical (normal ground is -150mm)
          
        Returns:
          Tuple of joint angles in radians: (theta_coxa, theta_femur, theta_tibia)
        """
        # 1. Coxa Angle (horizontal rotation)
        # Direct angle to the target coordinate in the horizontal plane
        theta_coxa = math.atan2(y, x)

        # 2. Projected coordinates in the vertical sagittal plane
        r = math.sqrt(x**2 + y**2)
        xp = r - self.L_COXA
        zp = z

        # Straight-line distance from femur pivot to foot tip
        Ld = math.sqrt(xp**2 + zp**2)

        # Reachability safety limits check
        max_reach = (self.L_FEMUR + self.L_TIBIA) * 0.99
        min_reach = abs(self.L_FEMUR - self.L_TIBIA) + 5.0

        if Ld > max_reach:
            scale = max_reach / Ld
            xp *= scale
            zp *= scale
            Ld = max_reach
        elif Ld < min_reach:
            scale = min_reach / Ld
            xp *= scale
            zp *= scale
            Ld = min_reach

        # 3. Tibia Joint Angle (Knee)
        # Solved using Law of Cosines: cos(C) = (a^2 + b^2 - c^2) / (2ab)
        cos_tibia = (self.L_FEMUR**2 + self.L_TIBIA**2 - Ld**2) / (2.0 * self.L_FEMUR * self.L_TIBIA)
        cos_tibia = min(max(cos_tibia, -1.0), 1.0)
        
        # Knee angle relative to femur extension
        theta_tibia = math.pi - math.acos(cos_tibia)

        # 4. Femur Joint Angle (Hip vertical swing)
        alpha = math.atan2(zp, xp)
        cos_femur = (self.L_FEMUR**2 + Ld**2 - self.L_TIBIA**2) / (2.0 * self.L_FEMUR * Ld)
        cos_femur = min(max(cos_femur, -1.0), 1.0)
        beta = math.acos(cos_femur)
        
        theta_femur = alpha + beta

        return theta_coxa, theta_femur, theta_tibia

    def angles_to_pwm(self, coxa: float, femur: float, tibia: float) -> Tuple[int, int, int]:
        """
        Converts joint angles (in radians) into PCA9685 12-bit register duty values (0 - 4095).
        Nominal joint position (0.0 rad) centers the servo around 90 degrees (1500 µs pulse).
        """
        # Convert radians to degrees
        c_deg = math.degrees(coxa)
        f_deg = math.degrees(femur)
        t_deg = math.degrees(tibia)

        # Map to servo angles (nominal offsets)
        # Adjust direction conventions based on side mounting
        servo_c = 90.0 + c_deg
        servo_f = 90.0 - f_deg
        servo_t = 90.0 + t_deg

        # Constrain to mechanical safety limits
        servo_c = min(max(servo_c, 0.0), 180.0)
        servo_f = min(max(servo_f, 0.0), 180.0)
        servo_t = min(max(servo_t, 0.0), 180.0)

        # Convert servo degrees (0-180) to PWM pulse width (500 - 2500 µs)
        def deg_to_tick(deg: float) -> int:
            us = self.PWM_MIN_US + (deg / 180.0) * (self.PWM_MAX_US - self.PWM_MIN_US)
            # TICK = (Pulse Width in seconds) * Frequency * Resolution
            tick = int((us / 1e6) * self.PWM_FREQ_HZ * self.PCA_RESOLUTION)
            return min(max(tick, 0), self.PCA_RESOLUTION - 1)

        return deg_to_tick(servo_c), deg_to_tick(servo_f), deg_to_tick(servo_t)

    def solve_fk(self, coxa: float, femur: float, tibia: float) -> Tuple[float, float, float]:
        """
        Forward Kinematics: Computes foot position (x, y, z) relative to shoulder pivot
        given joint angles (in radians).
        """
        # Projected length in horizontal plane
        r = self.L_COXA + self.L_FEMUR * math.cos(femur) + self.L_TIBIA * math.cos(femur - tibia)
        
        x = r * math.cos(coxa)
        y = r * math.sin(coxa)
        z = self.L_FEMUR * math.sin(femur) + self.L_TIBIA * math.sin(femur - tibia)
        
        return x, y, z
