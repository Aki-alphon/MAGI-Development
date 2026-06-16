import numpy as np

class LegIK3DOF:
    def __init__(self, L_coxa, L_femur, L_tibia):
        self.Lc = L_coxa
        self.L1 = L_femur
        self.L2 = L_tibia

    def solve(self, x, y, z):
        theta0 = np.arctan2(y, x)

        r = np.sqrt(x**2 + y**2)
        x_prime = r - self.Lc

        H = np.sqrt(x_prime**2 + z**2)
        H = np.clip(H, abs(self.L1 - self.L2), self.L1 + self.L2)

        cos_theta2 = (H**2 - self.L1**2 - self.L2**2) / (2 * self.L1 * self.L2)
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)

        theta2 = np.arccos(cos_theta2)

        phi = np.arctan2(z, x_prime)

        psi = np.arctan2(
            self.L2 * np.sin(theta2),
            self.L1 + self.L2 * np.cos(theta2)
        )

        theta1 = phi - psi

        return theta0, theta1, theta2

    def to_degrees(self, t0, t1, t2):
        return np.degrees(t0), np.degrees(t1), np.degrees(t2)