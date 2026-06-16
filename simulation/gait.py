import numpy as np

class Gait:
    def __init__(self, step_length=10, step_height=5, cycle_time=2.0):
        self.L = step_length
        self.H = step_height
        self.T = cycle_time

    def foot_trajectory(self, t, phase=0):
        """
        t: time
        phase: offset for each leg
        """

        t = (t + phase) % self.T

        # split cycle
        half = self.T / 2

        # --- X motion ---
        if t < half:  # stance
            x = -self.L/2 + self.L * (t / half)
            z = 0
        else:  # swing
            x = self.L/2 - self.L * ((t - half) / half)
            z = self.H * np.sin(np.pi * (t - half) / half)

        # --- Y stays constant (no sideways motion yet) ---
        y = 0

        return x, y, z