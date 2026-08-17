"""
MAGI OS v2 — MAGI-3 Lugia: Fusion & Decision Node
src/magi3/lugia.py

Upgraded to LifecycleNode. Subscribes to /detections, /scene, /sensors.
Publishes typed DecisionMsg to /decision (TRANSIENT_LOCAL QoS).
Pinned to CPU Core 3.
"""

import os, sys, time, threading
sys.path.insert(0, "/opt/magi/src")

try:
    os.sched_setaffinity(0, {3})
except (AttributeError, OSError):
    pass

import yaml
from core.lifecycle import LifecycleNode
from core.messages import (
    DecisionMsg, DetectionMsg, SceneMsg, SensorMsg, DiagStatus,
    CanopyHealthMsg,
)
from core.param_server import ParamClient


class Lugia(LifecycleNode):

    IDLE      = "IDLE"
    ALERT     = "ALERT"
    TRACK     = "TRACK"
    EMERGENCY = "EMERGENCY"
    ANALYZE   = "ANALYZE"

    def on_configure(self):
        with open("/opt/magi/config/config.yaml") as f:
            cfg = yaml.safe_load(f)
        self._out_cfg    = cfg.get("output", {})
        self._sensor_cfg = cfg.get("sensors", {})
        self._params     = ParamClient("magi3")

        # State cache — latest from each upstream node
        self._detections    = None
        self._scene         = None
        self._sensors       = None
        self._canopy_health = None   # ← NEW: plant heatmap from Celebi
        self._lock          = threading.Lock()
        self._decision_count = 0
        self._prev_action    = None

        # GPIO output
        self._gpio      = None
        self._alert_pin = None
        self._init_gpio()

        # Log file
        self._log_file = None
        if self._out_cfg.get("log_detections"):
            import os as _os
            _os.makedirs("/opt/magi/logs", exist_ok=True)
            self._log_file = open(self._out_cfg["log_file"], "a", buffering=1)

        # Publishers / subscribers
        self.create_publisher("/decision")
        self.create_subscription("/detections",    self._on_detections)
        self.create_subscription("/scene",         self._on_scene)
        self.create_subscription("/sensors",       self._on_sensors)
        self.create_subscription("/canopy_health", self._on_canopy_health)  # ← NEW
        self.log.info("Lugia configured")

    def _init_gpio(self):
        if not self._out_cfg.get("gpio_alert", False):
            return
        try:
            import pigpio
            self._gpio = pigpio.pi()
            for g in self._sensor_cfg.get("gpio", []):
                if g["name"] == "alert_out" and g.get("enabled"):
                    self._alert_pin = g["pin"]
            self.log.info(f"Alert GPIO pin: {self._alert_pin}")
        except Exception as e:
            self.log.warning(f"GPIO init failed: {e}")

    def on_activate(self):
        self.log.info("Lugia active — decision loop running")

    def on_deactivate(self):
        if self._log_file:
            self._log_file.flush()

    def on_cleanup(self):
        if self._gpio:
            self._gpio.stop()
        if self._log_file:
            self._log_file.close()

    # ── Subscription callbacks ──────────────────────────────────────────────

    def _on_detections(self, data: dict):
        with self._lock:
            self._detections = data
        self._fuse_and_decide()

    def _on_scene(self, data: dict):
        with self._lock:
            self._scene = data

    def _on_sensors(self, data: dict):
        with self._lock:
            self._sensors = data

    def _on_canopy_health(self, data: dict):
        """Receive plant health heatmap from Celebi — trigger a new decision."""
        with self._lock:
            self._canopy_health = data
        self._fuse_and_decide()

    # ── Decision Engine ─────────────────────────────────────────────────────

    def _fuse_and_decide(self):
        with self._lock:
            det    = self._detections    or {}
            sc     = self._scene         or {}
            sen    = self._sensors       or {}
            canopy = self._canopy_health or {}

        action   = self.IDLE
        priority = 0
        targets  = []
        reason   = "No significant events"

        dets    = det.get("detections", [])
        d_count = det.get("count", 0)
        scene   = sc.get("scene",         "unknown")
        anomaly = sc.get("anomaly_score", 0.0)
        is_mov  = sc.get("motion",        {}).get("is_moving", False)
        tof_mm  = (sen.get("tof") or {}).get("distance_mm", -1)
        gpio_ev = sen.get("gpio", {})

        # Canopy health fields (from Celebi)
        mean_health      = canopy.get("mean_health",        1.0)
        veg_coverage     = canopy.get("vegetation_coverage", 0.0)
        plant_action_rec = canopy.get("recommended_action", "IDLE")

        # ── Priority-ordered decision rules ────────────────────────────────

        # Rule P10: Emergency obstacle (ToF)
        if 0 < tof_mm < 200:
            action, priority = self.EMERGENCY, 10
            reason = f"Obstacle at {tof_mm:.0f} mm"

        # Rule P9: Hardware trigger
        elif gpio_ev.get("trigger_in", False):
            action, priority = self.EMERGENCY, 9
            reason = "Hardware trigger received"

        # Rule P8: Person in restricted zone
        elif scene == "restricted_zone" and any(d.get("label") == "person" for d in dets):
            action, priority = self.EMERGENCY, 8
            targets = [d for d in dets if d.get("label") == "person"]
            reason  = "Person in restricted zone"

        # Rule P6: High anomaly score (scene-level)
        elif anomaly > 0.7:
            action, priority = self.ALERT, 6
            reason = f"High anomaly: {anomaly:.2f}"

        # Rule P5: Scene emergency flag
        elif scene in ("emergency", "obstacle_close"):
            action, priority = self.ANALYZE, 5
            reason = f"Scene flag: {scene}"

        # Rule P4: Object detections (legacy path — e.g. from Gengar)
        elif d_count > 0:
            action, priority = self.TRACK, 4
            targets = dets
            reason  = f"Tracking {d_count} object(s)"

        # Rule P3: Plant health heatmap (MAGI-1 Celebi)
        elif veg_coverage > 0.1:   # at least 10% of frame is vegetation
            action, priority = plant_action_rec, 3
            reason = (
                f"Canopy health={mean_health:.2f} "
                f"veg={veg_coverage*100:.0f}% → {plant_action_rec}"
            )

        msg = DecisionMsg(
            header        = self.next_header("base_link"),
            action        = action,
            priority      = priority,
            targets       = targets,
            reason        = reason,
            scene         = scene,
            anomaly_score = anomaly,
            tof_mm        = tof_mm,
            is_moving     = is_mov,
            confidence    = float(canopy.get("confidence", 0.0)),
        )
        self.publish("/decision", msg)
        self._decision_count += 1

        # Side effects on state change
        if action != self._prev_action:
            self.log.info(f"Decision → {action} (p={priority}) | {reason}")
            self._prev_action = action
            if action in (self.EMERGENCY, self.ALERT):
                self._pulse_alert()
            if self._log_file:
                self._log_file.write(
                    f"{time.strftime('%Y-%m-%dT%H:%M:%S')} | {action} | {reason}\n"
                )

        # Diagnostics (include canopy health metrics)
        self.publish_diag(
            DiagStatus.OK,
            f"Decision={action}",
            {
                "action":      action,
                "priority":    priority,
                "decisions":   self._decision_count,
                "mean_health": round(mean_health, 3),
                "veg_%":       round(veg_coverage * 100, 1),
            },
        )

    def _pulse_alert(self, ms: int = 300):
        if self._gpio and self._alert_pin is not None:
            self._gpio.write(self._alert_pin, 1)
            threading.Timer(ms / 1000.0, lambda: self._gpio.write(self._alert_pin, 0)).start()


if __name__ == "__main__":
    Lugia(node_id="magi3", cpu_core=3).boot()
