import time
from ik import LegIK3DOF
from gait import Gait

# --- Robot dimensions ---
L_coxa = 6.3
L_femur = 10.15
L_tibia = 19.45

ik = LegIK3DOF(L_coxa, L_femur, L_tibia)
gait = Gait(step_length=10, step_height=5, cycle_time=2.0)

# --- Phase offsets (crawl gait) ---
phases = {
    "LF": 0.0,
    "RB": 0.5,
    "RF": 1.0,
    "LB": 1.5
}

start_time = time.time()

while True:
    t = time.time() - start_time

    print("\nTime:", round(t, 2))

    for leg, phase in phases.items():
        x, y, z = gait.foot_trajectory(t, phase)

        # shift leg position (important)
        # assume body offset for each leg
        if leg in ["LF", "LB"]:
            y_offset = 5
        else:
            y_offset = -5

        x += 10  # forward offset
        y += y_offset
        z -= 10  # keep leg below body

        t0, t1, t2 = ik.solve(x, y, z)

        print(f"{leg}:",
              f"{round(t0,2)}",
              f"{round(t1,2)}",
              f"{round(t2,2)}")

    time.sleep(0.05)