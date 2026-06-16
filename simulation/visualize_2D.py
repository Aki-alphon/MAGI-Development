import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ik import LegIK3DOF
from gait import Gait

# --- Robot dimensions ---
L_coxa = 6.3
L_femur = 10.15
L_tibia = 19.45

ik = LegIK3DOF(L_coxa, L_femur, L_tibia)
gait = Gait(step_length=10, step_height=5, cycle_time=2.0)

# phase offsets (crawl)
phases = {
    "LF": 0.0,
    "RB": 0.5,
    "RF": 1.0,
    "LB": 1.5
}

# leg positions (top view offsets)
leg_offsets = {
    "LF": (10, 5),
    "RF": (10, -5),
    "LB": (-10, 5),
    "RB": (-10, -5)
}

# matplotlib setup
fig, ax = plt.subplots()
ax.set_xlim(-30, 30)
ax.set_ylim(-30, 10)
ax.set_aspect('equal')
ax.grid()

lines = {}
for leg in phases:
    line, = ax.plot([], [], 'o-', label=leg)
    lines[leg] = line

ax.legend()


def compute_leg_points(x, y, z, t0, t1, t2):
    """
    Forward kinematics (approx visualization)
    """

    # hip
    hip = np.array([0, 0])

    # project to 2D plane (ignore y after rotation)
    r = np.sqrt(x**2 + y**2) - L_coxa

    # femur end
    knee = np.array([
        r * np.cos(t1),
        r * np.sin(t1)
    ])

    # tibia end (foot)
    foot = np.array([
        knee[0] + L_tibia * np.cos(t1 + t2),
        knee[1] + L_tibia * np.sin(t1 + t2)
    ])

    return hip, knee, foot


def update(frame):
    t = frame * 0.05

    for leg in phases:
        phase = phases[leg]

        x, y, z = gait.foot_trajectory(t, phase)

        # apply body offsets
        ox, oy = leg_offsets[leg]
        x += ox
        y += oy
        z -= 10

        t0, t1, t2 = ik.solve(x, y, z)

        hip, knee, foot = compute_leg_points(x, y, z, t0, t1, t2)

        xs = [hip[0], knee[0], foot[0]]
        ys = [hip[1], knee[1], foot[1]]

        lines[leg].set_data(xs, ys)

    return lines.values()


ani = FuncAnimation(fig, update, frames=200, interval=50)
plt.show()
print("RUNNING...")