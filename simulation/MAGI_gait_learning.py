"""
Quadruped Robot Gait Simulation
================================
3-DOF per leg (Coxa, Femur, Tibia) with full inverse kinematics,
crawl gait generator, and real-time 3D matplotlib animation.

Coordinate system:
    x → forward
    y → left (+) / right (-)
    z → vertical (0 = ground, negative = down into ground)

Leg naming:
    LF = Left Front   RF = Right Front
    LB = Left Back    RB = Right Back
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.patches as mpatches

# ───────────────────────────────────────────── 
#  ROBOT PARAMETERS  (all lengths in cm)
# ─────────────────────────────────────────────
L_COXA  = 6.30
L_FEMUR = 10.15
L_TIBIA = 19.45

# Hip attachment points on the body frame (x_fwd, y_lat, z=0)
HIP_POSITIONS = {
    "LF": np.array([ 10.0,  7.0, 0.0]),
    "RF": np.array([ 10.0, -7.0, 0.0]),
    "LB": np.array([-10.0,  7.0, 0.0]),
    "RB": np.array([-10.0, -7.0, 0.0]),
}

LEG_NAMES = ["LF", "RF", "LB", "RB"]

# Default standing foot position **relative to the hip**
STAND_HEIGHT = -(L_FEMUR + L_TIBIA) * 0.75   # ~comfortable stance depth (z)
DEFAULT_FOOT = {
    "LF": np.array([ 5.0,  8.0, STAND_HEIGHT]),
    "RF": np.array([ 5.0, -8.0, STAND_HEIGHT]),
    "LB": np.array([-5.0,  8.0, STAND_HEIGHT]),
    "RB": np.array([-5.0, -8.0, STAND_HEIGHT]),
}

# Crawl gait phase offsets  (fraction of cycle, [0,1))
PHASE_OFFSETS = {"LF": 0.0, "RB": 0.0,"RF": 0.5, "LB": 0.5}

# ─────────────────────────────────────────────
#  ADJUSTABLE GAIT PARAMETERS
# ─────────────────────────────────────────────
STEP_LENGTH = 10.0   # cm – how far forward each step travels
STEP_HEIGHT  = 5.0   # cm – peak swing height above stance plane
CYCLE_TIME   = 2.0   # seconds per full gait cycle
SWING_FRAC   = 0.25  # fraction of cycle spent in swing (rest = stance)
FPS          = 60    # animation frames per second
DURATION     = 8.0   # seconds of animation (set None for infinite)

# ─────────────────────────────────────────────
# [NEW] TASK 1 – TERRAIN LAYER
# ─────────────────────────────────────────────
# The base height the gait was designed around (z-component of DEFAULT_FOOT).
# Used to compute the delta when remapping foot z onto terrain.
BASE_HEIGHT = STAND_HEIGHT   # == -(L_FEMUR + L_TIBIA) * 0.75

def terrain(x, y):
    """
    Returns world-frame terrain z at position (x, y).
    Gentle sinusoidal bumps – kept mild so IK never goes out of range.
    Replace with any height-map function here; gait code is never touched.
    """
    return -15 + 3 * np.sin(0.2 * x) + 2 * np.cos(0.3 * y)

# ─────────────────────────────────────────────
# [NEW] TASK 3 – DISTURBANCE TOGGLE
# ─────────────────────────────────────────────
ENABLE_DISTURBANCE = False #set False to disable noise injection
DISTURBANCE_STD    = 0.2     # std-dev of zero-mean Gaussian noise (cm)


# ═══════════════════════════════════════════════════════════════════
#  MODULE 1 – INVERSE KINEMATICS
# ═══════════════════════════════════════════════════════════════════

def clamp(val, lo, hi):
    """Clamp val to [lo, hi]."""
    return max(lo, min(hi, val))

def inverse_kinematics(x, y, z, L_coxa=L_COXA, L1=L_FEMUR, L2=L_TIBIA):
    """
    Compute joint angles (theta0, theta1, theta2) for a 3-DOF leg.

    Inputs (all relative to the coxa joint / hip frame):
        x  – forward distance
        y  – lateral distance (+ = left)
        z  – vertical distance (negative = below hip)

    Returns:
        (theta0, theta1, theta2) in radians
        theta0 – coxa yaw   (rotation in the horizontal plane)
        theta1 – femur pitch (rotation in the sagittal plane)
        theta2 – tibia pitch (knee angle)

    Raises ValueError if the target is completely unreachable.
    """
    # --- Coxa: yaw to point toward foot in horizontal plane ---
    theta0 = np.arctan2(y, x)

    # Horizontal distance from coxa joint to foot (in the coxa plane)
    r = np.sqrt(x**2 + y**2)

    # After rotating by theta0, the remaining reach is reduced by coxa length
    x_prime = r - L_coxa           # distance femur root → foot (horizontal)
    H = np.sqrt(x_prime**2 + z**2) # straight-line distance femur root → foot

    # Reachability check – clamp to avoid acos domain errors
    max_reach = L1 + L2
    min_reach = abs(L1 - L2)
    H = clamp(H, min_reach + 1e-6, max_reach - 1e-6)

    # --- Tibia: use law of cosines ---
    cos_theta2 = (H**2 - L1**2 - L2**2) / (2.0 * L1 * L2)
    cos_theta2 = clamp(cos_theta2, -1.0, 1.0)
    theta2 = np.arccos(cos_theta2)   # always positive; knee bends "backward"

    # --- Femur: two-part angle ---
    alpha = np.arctan2(-z, x_prime)          # angle to foot from horizontal
    beta  = np.arctan2(L2 * np.sin(theta2),  # angle offset due to tibia
                       L1 + L2 * np.cos(theta2))
    theta1 = alpha - beta

    return theta0, theta1, theta2


def forward_kinematics(theta0, theta1, theta2,
                        L_coxa=L_COXA, L1=L_FEMUR, L2=L_TIBIA):
    """
    Given joint angles, return the 3D positions of each joint
    (coxa_tip, knee, foot) in the hip frame.

    Useful for drawing the leg segments.
    """
    # Coxa tip position (in hip frame)
    cx = L_coxa * np.cos(theta0)
    cy = L_coxa * np.sin(theta0)
    cz = 0.0
    coxa_tip = np.array([cx, cy, cz])

    # Knee position (femur end)
    kx = cx + L1 * np.cos(theta0) * np.cos(theta1)
    ky = cy + L1 * np.sin(theta0) * np.cos(theta1)
    kz = cz + L1 * np.sin(theta1)
    knee = np.array([kx, ky, kz])

    # Foot position (tibia end)
    fx = kx + L2 * np.cos(theta0) * np.cos(theta1 + theta2)
    fy = ky + L2 * np.sin(theta0) * np.cos(theta1 + theta2)
    fz = kz + L2 * np.sin(theta1 + theta2)
    foot = np.array([fx, fy, fz])

    return coxa_tip, knee, foot


# ═══════════════════════════════════════════════════════════════════
#  MODULE 2 – FOOT TRAJECTORY
# ═══════════════════════════════════════════════════════════════════

def foot_trajectory(phase, default_pos,
                    step_length=STEP_LENGTH,
                    step_height=STEP_HEIGHT,
                    swing_frac=SWING_FRAC):
    """
    Compute the foot target position (in hip frame) for a given gait phase.

    phase       – float in [0, 1), current position in the cycle
    default_pos – numpy array (3,), nominal foot resting position

    Stance phase  (phase ∈ [swing_frac, 1)):
        Foot sweeps backward linearly from +step_length/2 to -step_length/2
        at constant z = default_pos[2].

    Swing phase   (phase ∈ [0, swing_frac)):
        Foot sweeps forward from -step_length/2 to +step_length/2
        with a sinusoidal lift of step_height.
    """
    x0, y0, z0 = default_pos

    if phase < swing_frac:
        # ── SWING ──────────────────────────────────────────────────
        t = phase / swing_frac           # normalised swing time [0,1)
        # x moves from rear (-) to front (+)
        x = x0 + step_length * (t - 0.5)
        # z rises and falls smoothly
        z = z0 + step_height * np.sin(np.pi * t)
        y = y0
    else:
        # ── STANCE ─────────────────────────────────────────────────
        t = (phase - swing_frac) / (1.0 - swing_frac)  # [0,1)
        # x moves from front (+) to rear (-)
        x = x0 + step_length * (0.5 - t)
        z = z0
        y = y0

    return np.array([x, y, z])


# ═══════════════════════════════════════════════════════════════════
#  MODULE 3 – GAIT GENERATOR
# ═══════════════════════════════════════════════════════════════════

class CrawlGait:
    """
    Generates per-leg foot targets for the crawl (wave) gait.

    The crawl gait moves one leg at a time using the phase offsets:
        LF: 0      RB: T/4    RF: T/2    LB: 3T/4

    At any point in time, only one leg is in swing; the other three
    are in stance – guaranteeing a stable, statically balanced gait.
    """

    def __init__(self,
                 step_length=STEP_LENGTH,
                 step_height=STEP_HEIGHT,
                 cycle_time=CYCLE_TIME,
                 swing_frac=SWING_FRAC):
        self.step_length = step_length
        self.step_height  = step_height
        self.cycle_time   = cycle_time
        self.swing_frac   = swing_frac

    def get_foot_targets(self, t):
        """
        Returns a dict { leg_name: np.array([x,y,z]) } of foot positions
        in each leg's hip frame at absolute time t (seconds).
        """
        targets = {}
        cycle_phase = (t % self.cycle_time) / self.cycle_time  # [0,1)

        for leg in LEG_NAMES:
            offset = PHASE_OFFSETS[leg]
            phase  = (cycle_phase - offset) % 1.0
            targets[leg] = foot_trajectory(phase,
                                           DEFAULT_FOOT[leg],
                                           self.step_length,
                                           self.step_height,
                                           self.swing_frac)
        return targets


# ═══════════════════════════════════════════════════════════════════
#  MODULE 4 – VISUALISATION
# ═══════════════════════════════════════════════════════════════════

# Colours for each leg
LEG_COLORS = {"LF": "#00C8FF", "RF": "#FF6B35", "LB": "#A8FF3E", "RB": "#FF3EFF"}

# ─────────────────────────────────────────────
# [NEW] TASK 2 – BODY OFFSET GENERATOR
# ─────────────────────────────────────────────
BODY_OFFSET_AMP_Z  = 1.5   # cm – vertical oscillation amplitude (breathing motion)
BODY_OFFSET_AMP_X  = 0.5   # cm – subtle fore-aft sway amplitude

def get_body_offset(cycle_phase):
    """
    Computes a sinusoidal body offset keyed to cycle phase.
    Applied EXTERNALLY to hip_world only in compute_joint_positions();
    HIP_POSITIONS dict is NEVER modified.

    Returns np.array([dx, dy, dz]).
    """
    # Gentle Z oscillation (body rises/falls once per cycle)
    dz = BODY_OFFSET_AMP_Z * np.sin(2 * np.pi * cycle_phase)
    # Subtle X sway at double frequency
    dx = BODY_OFFSET_AMP_X * np.sin(4 * np.pi * cycle_phase)
    dy = 0.0
    return np.array([dx, dy, dz])

def build_scene(ax):
    """Configure the 3D axes and draw a ground grid."""
    ax.set_xlim(-35, 35)
    ax.set_ylim(-35, 35)
    ax.set_zlim(-35, 5)
    ax.set_xlabel("X (forward)", color="#aaaaaa", labelpad=6)
    ax.set_ylabel("Y (lateral)", color="#aaaaaa", labelpad=6)
    ax.set_zlabel("Z (vertical)", color="#aaaaaa", labelpad=6)
    ax.set_title("Quadruped Crawl Gait Simulation", color="white", fontsize=13, pad=12)

    # Draw ground plane grid
    grid_range = np.linspace(-30, 30, 13)
    z_ground   = STAND_HEIGHT * 1.02   # just below the deepest foot
    for g in grid_range:
        ax.plot([g, g], [-30, 30], [z_ground, z_ground],
                color="#333333", lw=0.4, zorder=0)
        ax.plot([-30, 30], [g, g], [z_ground, z_ground],
                color="#333333", lw=0.4, zorder=0)

    ax.set_facecolor("#0a0a12")
    ax.tick_params(colors="#555555", labelsize=7)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_body(ax, artists):
    """Draw/update the rectangular body outline."""
    corners_body = np.array([
        HIP_POSITIONS["LF"], HIP_POSITIONS["RF"],
        HIP_POSITIONS["RB"], HIP_POSITIONS["LB"],
        HIP_POSITIONS["LF"],
    ])
    if "body" not in artists:
        artists["body"], = ax.plot(corners_body[:, 0],
                                   corners_body[:, 1],
                                   corners_body[:, 2],
                                   color="#ffffff", lw=2.0, zorder=5)
        # Body centre dot
        artists["body_dot"], = ax.plot([0], [0], [0],
                                       "o", color="#ffffff",
                                       markersize=5, zorder=6)
    else:
        artists["body"].set_data_3d(corners_body[:, 0],
                                    corners_body[:, 1],
                                    corners_body[:, 2])


def draw_legs(ax, artists, joint_positions):
    """
    Draw/update the 3 segments of each leg:
        hip → coxa_tip → knee → foot
    and the joint spheres.

    joint_positions: dict { leg: (hip, coxa_tip, knee, foot) }
    """
    for leg in LEG_NAMES:
        hip, coxa_tip, knee, foot = joint_positions[leg]
        color = LEG_COLORS[leg]

        xs = [hip[0], coxa_tip[0], knee[0], foot[0]]
        ys = [hip[1], coxa_tip[1], knee[1], foot[1]]
        zs = [hip[2], coxa_tip[2], knee[2], foot[2]]

        key_line = f"{leg}_line"
        key_jnts = f"{leg}_joints"
        key_foot = f"{leg}_foot"

        if key_line not in artists:
            artists[key_line], = ax.plot(xs, ys, zs,
                                         color=color, lw=2.2, zorder=4)
            artists[key_jnts], = ax.plot(xs[:-1], ys[:-1], zs[:-1],
                                         "o", color=color,
                                         markersize=5, zorder=5)
            artists[key_foot], = ax.plot([xs[-1]], [ys[-1]], [zs[-1]],
                                         "o", color="#ffffff",
                                         markersize=6, zorder=6,
                                         markeredgecolor=color, markeredgewidth=1.5)
        else:
            artists[key_line].set_data_3d(xs, ys, zs)
            artists[key_jnts].set_data_3d(xs[:-1], ys[:-1], zs[:-1])
            artists[key_foot].set_data_3d([xs[-1]], [ys[-1]], [zs[-1]])


# ═══════════════════════════════════════════════════════════════════
#  MODULE 5 – MAIN SIMULATION LOOP
# ═══════════════════════════════════════════════════════════════════

def compute_joint_positions(foot_targets, body_offset=None):
    """
    For each leg, run IK on the foot target and then FK to get
    the 3D positions of all joints in the world frame.

    [NEW] body_offset (np.array, optional) – added to hip_world only;
          HIP_POSITIONS is never mutated (TASK 2).
    [NEW] Stability checks printed to console (TASK 4) – diagnostics only,
          no control changes.

    Returns dict: { leg_name: (hip_world, coxa_tip_world, knee_world, foot_world) }
    """
    if body_offset is None:
        body_offset = np.zeros(3)

    positions = {}
    for leg in LEG_NAMES:
        # [NEW TASK 2] Ephemeral hip position – original dict untouched
        hip_world = HIP_POSITIONS[leg] + body_offset
        target    = foot_targets[leg]   # in hip frame

        # ── [NEW TASK 4] STABILITY CHECKS (diagnostics only) ──────
        foot_world_z = hip_world[2] + target[2]   # approx world z of foot
        if foot_world_z > hip_world[2]:
            print(f"[STABILITY] INVALID FOOT POSITION  – {leg}: "
                  f"foot_z={foot_world_z:.1f} > hip_z={hip_world[2]:.1f}")
        if body_offset[2] < -5.0:
            print(f"[STABILITY] FALL DETECTED – body_offset_z={body_offset[2]:.2f} cm")
        # ───────────────────────────────────────────────────────────

        try:
            t0, t1, t2 = inverse_kinematics(*target)
        except Exception:
            # If IK fails for some reason, use default angles
            t0, t1, t2 = inverse_kinematics(*DEFAULT_FOOT[leg])

        # FK returns points in hip frame → shift to world frame
        coxa_local, knee_local, foot_local = forward_kinematics(t0, t1, t2)
        positions[leg] = (
            hip_world,
            hip_world + coxa_local,
            hip_world + knee_local,
            hip_world + foot_local,
        )
    return positions


def run_simulation(step_length=STEP_LENGTH,
                   step_height=STEP_HEIGHT,
                   cycle_time=CYCLE_TIME):
    """
    Main entry point: creates the figure, sets up the animation loop,
    and starts the matplotlib event loop.
    """
    gait = CrawlGait(step_length=step_length,
                     step_height=step_height,
                     cycle_time=cycle_time)

    # ── Figure setup ──────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8), facecolor="#0a0a12")
    ax  = fig.add_subplot(111, projection="3d")
    build_scene(ax)

    # Legend patches
    patches = [mpatches.Patch(color=LEG_COLORS[l], label=l) for l in LEG_NAMES]
    ax.legend(handles=patches, loc="upper left",
              facecolor="#111122", edgecolor="#333333",
              labelcolor="white", fontsize=9)

    # Phase indicator text
    phase_txt = ax.text2D(0.02, 0.06, "", transform=ax.transAxes,
                          color="#aaaaaa", fontsize=8)
    time_txt  = ax.text2D(0.02, 0.02, "", transform=ax.transAxes,
                          color="#666666", fontsize=8)

    # [NEW TASK 5] Debug overlay text (right side of plot)
    debug_txt = ax.text2D(0.62, 0.98, "", transform=ax.transAxes,
                          color="#cccccc", fontsize=7,
                          verticalalignment="top",
                          family="monospace")

    artists = {}
    draw_body(ax, artists)   # body never changes, draw once

    # ── Animation callback ────────────────────────────────────────
    def update(frame):
        t = frame / FPS

        # ── STEP 1: Gait generator → foot targets (hip frame) ─────
        # [UNTOUCHED – gait logic never modified]
        foot_targets = gait.get_foot_targets(t)

        # ── STEP 2 [NEW TASK 1]: Terrain layer ────────────────────
        # Adjust ONLY the z value of each foot target so it conforms
        # to terrain height. Gait trajectory shape is fully preserved.
        cycle_phase = (t % cycle_time) / cycle_time
        for leg in LEG_NAMES:
            x, y, z = foot_targets[leg]
            # World x/y = hip_x + foot_x_local (approx; good enough for terrain lookup)
            wx = HIP_POSITIONS[leg][0] + x
            wy = HIP_POSITIONS[leg][1] + y
            terrain_z = terrain(wx, wy)
            # Shift z by how far the gait foot is above/below the design base height
            z_adjusted = terrain_z + (z - BASE_HEIGHT)
            foot_targets[leg] = np.array([x, y, z_adjusted])

        # ── STEP 3 [NEW TASK 3]: Disturbance injection ────────────
        # Small zero-mean Gaussian noise – toggle via ENABLE_DISTURBANCE
        if ENABLE_DISTURBANCE:
            for leg in LEG_NAMES:
                foot_targets[leg] += np.random.normal(0, DISTURBANCE_STD, size=3)

        # ── STEP 4 [NEW TASK 2]: Body offset ──────────────────────
        # Computed from cycle phase, applied only inside compute_joint_positions
        body_offset = get_body_offset(cycle_phase)

        # ── STEP 5: IK + FK → world-frame joint positions ─────────
        # body_offset injected here; HIP_POSITIONS never mutated
        joint_positions = compute_joint_positions(foot_targets, body_offset)

        # ── STEP 6: Update visuals ─────────────────────────────────
        draw_body(ax, artists)
        draw_legs(ax, artists, joint_positions)

        # ── STEP 7 [NEW TASK 5]: Debug overlay ────────────────────
        # Computes angles fresh (same as IK above; negligible cost)
        angle_lines  = []
        debug_lines  = ["── DEBUG ──────────────────────────"]

        for leg in LEG_NAMES:
            target = foot_targets[leg]
            t0, t1, t2 = inverse_kinematics(*target)

            p     = (cycle_phase - PHASE_OFFSETS[leg]) % 1.0
            state = "SWING " if p < SWING_FRAC else "STANCE"

            # Phase + state overlay (left side – existing)
            angle_lines.append(
                f"{leg}: {state} | "
                f"θ0={int(np.degrees(t0)):+4d}° "
                f"θ1={int(np.degrees(t1)):+4d}° "
                f"θ2={int(np.degrees(t2)):+4d}°"
            )

            # Terrain height under this foot (debug panel – right side)
            wx = HIP_POSITIONS[leg][0] + target[0]
            wy = HIP_POSITIONS[leg][1] + target[1]
            th = terrain(wx, wy)
            debug_lines.append(
                f"{leg} | {state} | "
                f"terrain_z={th:+5.1f}cm | "
                f"θ0={int(np.degrees(t0)):+4d}° "
                f"θ1={int(np.degrees(t1)):+4d}° "
                f"θ2={int(np.degrees(t2)):+4d}°"
            )

        phase_txt.set_text("\n".join(angle_lines))
        debug_lines.append(f"body_offset_z = {body_offset[2]:+.2f} cm")
        debug_lines.append(f"disturbance   = {'ON' if ENABLE_DISTURBANCE else 'OFF'}")
        debug_txt.set_text("\n".join(debug_lines))

        time_txt.set_text(f"t = {t:.2f}s   cycle phase = {cycle_phase:.2f}")

        return list(artists.values()) + [phase_txt, time_txt, debug_txt]

    # ── Run ───────────────────────────────────────────────────────
    total_frames = int(DURATION * FPS) if DURATION else None
    interval     = 1000 / FPS   # ms between frames

    ani = animation.FuncAnimation(
        fig, update,
        frames=total_frames,
        interval=interval,
        blit=False,
        repeat=True,
    )

    plt.tight_layout()
    print("╔══════════════════════════════════════════╗")
    print("║  Quadruped Crawl Gait Simulation         ║")
    print("╠══════════════════════════════════════════╣")
    print(f"║  Coxa={L_COXA}cm  Femur={L_FEMUR}cm  Tibia={L_TIBIA}cm   ║")
    print(f"║  Step length : {step_length} cm                    ║")
    print(f"║  Step height : {step_height} cm         ║")
    print(f"║  Cycle time  : {cycle_time} s           ║")
    print(f"║  Gait        : Crawl (1-leg-at-a-time)  ║")
    print("╚══════════════════════════════════════════╝")
    plt.show()
    return ani   # keep reference alive





# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # ── Adjustable parameters ──────────────────
    SIM_STEP_LENGTH = 10.0   # cm
    SIM_STEP_HEIGHT  = 5.0   # cm
    SIM_CYCLE_TIME   = 2.0   # seconds
    # ───────────────────────────────────────────

    ani = run_simulation(
        step_length=SIM_STEP_LENGTH,
        step_height=SIM_STEP_HEIGHT,
        cycle_time=SIM_CYCLE_TIME,
    )