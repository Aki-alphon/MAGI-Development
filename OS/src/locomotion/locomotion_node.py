"""
MAGI OS — Locomotion Node
src/locomotion/locomotion_node.py

A LifecycleNode that:
  1. Loads the RL-trained gait config (gait_config.yaml).
  2. Subscribes to /decision (Lugia) and /sensors (SensorHub).
  3. Selects gait mode and body posture based on the active decision.
  4. Drives 12 MG996R servos at 50Hz via PCA9685 over I2C.
  5. Publishes /locomotion_state diagnostic telemetry.

CPU Core pinning: Core 0 (same as sensor hub) to avoid interfering
with AI inference cores 1-3.
"""

import os, sys, time, threading, math
sys.path.insert(0, "/opt/magi/src")

try:
    os.sched_setaffinity(0, {0})
except (AttributeError, OSError):
    pass

import yaml
from core.lifecycle import LifecycleNode
from core.messages import DiagStatus
from locomotion.gait_controller import GaitController

# PCA9685 channel mapping for 12 servos (4 legs × 3 joints)
# Each row: [coxa_ch, femur_ch, tibia_ch]
SERVO_CHANNELS = [
    [0,  1,  2],   # Leg 0: FR
    [4,  5,  6],   # Leg 1: FL
    [8,  9,  10],  # Leg 2: BR
    [12, 13, 14],  # Leg 3: BL
]

GAIT_CONFIG_PATH = "/opt/magi/config/gait_config.yaml"
CONTROL_HZ = 50  # Servo update rate


class PCA9685Driver:
    """Thin I2C wrapper for PCA9685 PWM servo controller."""

    ADDR = 0x40
    MODE1 = 0x00
    PRESCALE = 0xFE
    LED0_ON_L = 0x06

    def __init__(self, bus_id: int = 1, freq_hz: int = 50):
        try:
            import smbus2
            self.bus = smbus2.SMBus(bus_id)
            self._init_chip(freq_hz)
            self.available = True
        except Exception as e:
            self.available = False
            self.bus = None
            print(f"[LOCO] PCA9685 init failed (sim mode): {e}")

    def _init_chip(self, freq_hz):
        import smbus2
        self.bus.write_byte_data(self.ADDR, self.MODE1, 0x10)  # sleep
        time.sleep(0.01)
        prescale = round(25_000_000.0 / (4096 * freq_hz)) - 1
        self.bus.write_byte_data(self.ADDR, self.PRESCALE, prescale)
        self.bus.write_byte_data(self.ADDR, self.MODE1, 0x20)  # wake, auto-increment
        time.sleep(0.01)

    def set_pwm(self, channel: int, on: int, off: int):
        if not self.available:
            return
        reg = self.LED0_ON_L + 4 * channel
        self.bus.write_i2c_block_data(self.ADDR, reg,
            [on & 0xFF, on >> 8, off & 0xFF, off >> 8])

    def set_tick(self, channel: int, tick: int):
        self.set_pwm(channel, 0, tick)


class LocomotionNode(LifecycleNode):

    # Map Lugia action strings → gait type + speed
    ACTION_MAP = {
        "IDLE":      {"gait": "stand",  "freq": 0.0, "stride_scale": 1.0},
        "TRACK":     {"gait": "crawl",  "freq": 1.0, "stride_scale": 0.8},
        "ALERT":     {"gait": "crawl",  "freq": 0.6, "stride_scale": 0.5},
        "ANALYZE":   {"gait": "crawl",  "freq": 0.8, "stride_scale": 0.6},
        "EMERGENCY": {"gait": "stand",  "freq": 0.0, "stride_scale": 0.0},
    }

    def on_configure(self):
        # Load RL-trained gait config
        cfg_path = GAIT_CONFIG_PATH
        if not os.path.exists(cfg_path):
            # Fallback to workspace path during development
            cfg_path = "/home/aki/Downloads/MAGI/OS/src/common/gait_config.yaml"

        with open(cfg_path) as f:
            raw = yaml.safe_load(f)

        gp = raw["gait_parameters"]
        self._trained_phases = [
            gp["phase_offsets"]["front_right"],
            gp["phase_offsets"]["front_left"],
            gp["phase_offsets"]["back_right"],
            gp["phase_offsets"]["back_left"],
        ]
        self._trained_stride  = gp["target_stride_mm"]
        self._trained_height  = gp["neutral_height_mm"]
        self.log.info(f"Loaded gait config: fitness={gp['fitness_score']}, "
                      f"stride={self._trained_stride}mm, height={self._trained_height}mm")

        # Initialise gait controller with trained params
        self._gait = GaitController(
            frequency=1.5,
            stride=self._trained_stride,
            step_height=30.0,
            body_height=self._trained_height,
        )

        # PCA9685
        self._pca = PCA9685Driver(bus_id=1, freq_hz=50)

        # State
        self._action        = "IDLE"
        self._tof_mm        = 9999
        self._lock          = threading.Lock()
        self._cycle_count   = 0
        self._start_time    = time.monotonic()

        # Subscriptions
        self.create_subscription("/decision", self._on_decision)
        self.create_subscription("/sensors",  self._on_sensors)

        # Publisher for locomotion telemetry
        self.create_publisher("/locomotion")
        self.log.info("LocomotionNode configured")

    def on_activate(self):
        self._running = True
        threading.Thread(target=self._servo_loop, daemon=True).start()
        self.log.info(f"Servo loop started @ {CONTROL_HZ} Hz")

    def on_deactivate(self):
        self._running = False

    def on_cleanup(self):
        # Return all servos to neutral (90°) before shutdown
        self._set_all_neutral()

    # ── Subscription callbacks ──────────────────────────────────────────────

    def _on_decision(self, data: dict):
        with self._lock:
            self._action = data.get("action", "IDLE")

    def _on_sensors(self, data: dict):
        with self._lock:
            tof = (data.get("tof") or {}).get("distance_mm", 9999)
            self._tof_mm = tof

    # ── Main servo control loop ─────────────────────────────────────────────

    def _servo_loop(self):
        dt = 1.0 / CONTROL_HZ

        while self._running:
            t0 = time.monotonic()

            with self._lock:
                action  = self._action
                tof     = self._tof_mm

            # Select gait parameters from action
            params = self.ACTION_MAP.get(action, self.ACTION_MAP["IDLE"])
            self._gait.set_gait(params["gait"])
            self._gait.frequency = params["freq"] if params["freq"] > 0 else 1.5
            self._gait.stride    = self._trained_stride * params["stride_scale"]

            # Emergency: freeze immediately regardless of decision bus
            if action == "EMERGENCY" or (0 < tof < 180):
                self._set_all_neutral()
                time.sleep(dt)
                continue

            # Calculate all 12 servo tick values
            elapsed = time.monotonic() - self._start_time
            pwm_cmds = self._gait.calculate_gait_step(elapsed)

            # Send PWM ticks to PCA9685
            for leg_idx, (c_tick, f_tick, t_tick) in enumerate(pwm_cmds):
                ch = SERVO_CHANNELS[leg_idx]
                self._pca.set_tick(ch[0], c_tick)
                self._pca.set_tick(ch[1], f_tick)
                self._pca.set_tick(ch[2], t_tick)

            self._cycle_count += 1

            # Publish telemetry every 50 cycles (1 Hz)
            if self._cycle_count % CONTROL_HZ == 0:
                self.publish("/locomotion", {
                    "action":      action,
                    "gait_type":   params["gait"],
                    "frequency_hz": params["freq"],
                    "stride_mm":   self._gait.stride,
                    "tof_mm":      tof,
                    "cycle":       self._cycle_count,
                })
                self.publish_diag(DiagStatus.OK, f"Gait={params['gait']}", {
                    "cycles": self._cycle_count,
                    "action": action,
                })

            elapsed_loop = time.monotonic() - t0
            sleep_t = dt - elapsed_loop
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                self.log.debug(f"Servo loop overrun by {-sleep_t*1000:.1f} ms")

    def _set_all_neutral(self):
        """Send all servos to 90° (neutral/safe stand position)."""
        # 90° = 1500µs pulse → tick = (1500/1e6) * 50 * 4096 = 307
        NEUTRAL_TICK = 307
        for leg_idx in range(4):
            for ch_idx in range(3):
                self._pca.set_tick(SERVO_CHANNELS[leg_idx][ch_idx], NEUTRAL_TICK)


if __name__ == "__main__":
    LocomotionNode(node_id="locomotion", cpu_core=0).boot()
