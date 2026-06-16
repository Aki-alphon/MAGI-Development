"""
MAGI OS — Locomotion Gait Controller
src/locomotion/gait_controller.py

Manages gait states (Static Crawl, Trot) and generates the continuous foot trajectory
vectors. Feeds the 3D targets to the LegKinematics solver to yield 12 joint angle ticks
ready to be serialized and transmitted to the ESP32 microcontroller.
"""

import time
import math
from typing import List, Dict, Tuple
from locomotion.kinematics import LegKinematics

class GaitController:
    def __init__(self, frequency: float = 1.5, stride: float = 60.0, step_height: float = 30.0, body_height: float = 150.0):
        self.gait_type = "crawl"  # crawl, trot, stand
        self.frequency = frequency # Hz
        self.stride = stride       # mm
        self.step_height = step_height # mm
        self.body_height = body_height # mm

        # Body posture offsets
        self.body_offset_x = 0.0
        self.body_offset_y = 0.0
        self.body_offset_z = 0.0
        
        this_pitch = 0.0 # deg
        this_roll = 0.0  # deg

        # Initialize kinematics engines for the four legs
        # Leg ordering matching: 0:FR, 1:FL, 2:BR, 3:BL
        self.legs = [
            LegKinematics(0, is_left=False), # FR
            LegKinematics(1, is_left=True),  # FL
            LegKinematics(2, is_left=False), # BR
            LegKinematics(3, is_left=True)   # BL
        ]

        # Neutral coordinates relative to the chassis center (shoulder pivots)
        # FR: x=90, y=-40, z=0
        # FL: x=90, y=40, z=0
        # BR: x=-90, y=-40, z=0
        # BL: x=-90, y=40, z=0
        self.shoulders = [
            {"x": 90.0,  "y": -40.0, "z": 0.0}, # FR
            {"x": 90.0,  "y": 40.0,  "z": 0.0}, # FL
            {"x": -90.0, "y": -40.0, "z": 0.0}, # BR
            {"x": -90.0, "y": 40.0,  "z": 0.0}  # BL
        ]

        # Stance offsets from shoulders (nominal footprint)
        self.footprint_offsets = [
            {"x": 0.0, "y": -60.0, "z": -self.body_height}, # FR
            {"x": 0.0, "y": 60.0,  "z": -self.body_height}, # FL
            {"x": 0.0, "y": -60.0, "z": -self.body_height}, # BR
            {"x": 0.0, "y": 60.0,  "z": -self.body_height}  # BL
        ]

    def set_gait(self, gait_type: str):
        if gait_type in ["crawl", "trot", "stand"]:
            self.gait_type = gait_type

    def calculate_gait_step(self, t: float) -> List[Tuple[int, int, int]]:
        """
        Calculates the joint ticks (0 - 4095) for all 12 servos for the given time t.
        """
        phase = (t * self.frequency) % 1.0
        foot_targets = []

        # 1. Posture & offsets calculation
        h_offset = -self.body_height + self.body_offset_z
        
        # Symmetrical footprints
        current_footprints = []
        for i in range(4):
            fp = {
                "x": self.footprint_offsets[i]["x"] + self.body_offset_x,
                "y": self.footprint_offsets[i]["y"] + self.body_offset_y,
                "z": h_offset
            }
            current_footprints.append(fp)

        # 2. Sequence foot coordinates
        if self.gait_type == "crawl":
            # 4-phase static crawl
            swing_idx = -1
            leg_phase = 0.0

            if phase < 0.25:
                swing_idx = 0  # FR
                leg_phase = (phase - 0.00) / 0.25
            elif phase < 0.50:
                swing_idx = 1  # FL
                leg_phase = (phase - 0.25) / 0.25
            elif phase < 0.75:
                swing_idx = 3  # BL
                leg_phase = (phase - 0.50) / 0.25
            else:
                swing_idx = 2  # BR
                leg_phase = (phase - 0.75) / 0.25

            for i in range(4):
                fp = current_footprints[i]
                if i == swing_idx:
                    # Swing phase (Reaches forward)
                    start_x = fp["x"] - self.stride / 2.0
                    end_x = fp["x"] + self.stride / 2.0
                    
                    target_x = start_x + (end_x - start_x) * leg_phase
                    target_z = fp["z"] + math.sin(leg_phase * math.pi) * self.step_height
                    foot_targets.append((target_x, fp["y"], target_z))
                else:
                    # Stance phase (Pushes ground backward)
                    # Stance normalisation time mapping
                    norm_stance = 0.0
                    if phase < 0.25:
                        norm_stance = 0.0 if i == 0 else (phase / 0.25)
                    elif phase < 0.50:
                        norm_stance = 0.0 if i == 1 else ((phase - 0.25) / 0.25)
                    elif phase < 0.75:
                        norm_stance = 0.0 if i == 3 else ((phase - 0.50) / 0.25)
                    else:
                        norm_stance = 0.0 if i == 2 else ((phase - 0.75) / 0.25)

                    target_x = fp["x"] + (self.stride / 2.0) - self.stride * norm_stance
                    foot_targets.append((target_x, fp["y"], fp["z"]))

        elif self.gait_type == "trot":
            # 2-phase diagonal trot
            is_phase_0 = (phase < 0.5)
            trot_phase = (phase / 0.5) if is_phase_0 else ((phase - 0.5) / 0.5)

            for i in range(4):
                fp = current_footprints[i]
                is_swing = False
                if is_phase_0 and (i == 0 or i == 3): is_swing = True
                if not is_phase_0 and (i == 1 or i == 2): is_swing = True

                if is_swing:
                    start_x = fp["x"] - self.stride / 2.0
                    end_x = fp["x"] + self.stride / 2.0
                    target_x = start_x + (end_x - start_x) * trot_phase
                    target_z = fp["z"] + math.sin(trot_phase * math.pi) * self.step_height
                    foot_targets.append((target_x, fp["y"], target_z))
                else:
                    target_x = fp["x"] + (self.stride / 2.0) - self.stride * trot_phase
                    foot_targets.append((target_x, fp["y"], fp["z"]))

        else:  # stand
            for i in range(4):
                foot_targets.append((current_footprints[i]["x"], current_footprints[i]["y"], current_footprints[i]["z"]))

        # 3. Solve IK for all legs and convert to 12 servo PCA9685 ticks
        pwm_commands = []
        for i in range(4):
            tx, ty, tz = foot_targets[i]
            
            # Map target coordinates relative to shoulder pivot
            # Outward lateral mapping
            rel_x = tx
            rel_y = ty if self.legs[i].is_left else -ty
            rel_z = tz

            # Run analytical IK solver
            c_rad, f_rad, t_rad = self.legs[i].solve_ik(rel_x, rel_y, rel_z)
            
            # Map joint angles in radians to 12-bit register ticks
            ticks = self.legs[i].angles_to_pwm(c_rad, f_rad, t_rad)
            pwm_commands.append(ticks)

        return pwm_commands
