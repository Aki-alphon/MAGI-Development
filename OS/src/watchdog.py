"""
MAGI OS — Watchdog & Process Supervisor
src/watchdog.py

Monitors all MAGI processes and auto-restarts on failure.
Subscribes to /diagnostics and monitors 1 Hz heartbeats.
If a node hangs, issues SIGTERM, flushes POSIX shared memory,
and restarts only that specific process via the lifecycle node.
"""

import os
import sys
import time
import signal
import subprocess
import yaml
import threading
import zmq

sys.path.insert(0, "/opt/magi/src")

from common.logger import get_logger
from core.messages import decode

log = get_logger("watchdog")

with open("/opt/magi/config/config.yaml") as f:
    CFG = yaml.safe_load(f)

VENV_PYTHON = "/opt/magi/venv/bin/python3"
SRC         = "/opt/magi/src"

SERVICES = [
    {
        "name":    "message_bus",
        "cmd":     [VENV_PYTHON, f"{SRC}/core/message_bus.py"],
        "delay":   0,     # Start FIRST — all nodes depend on it
    },
    {
        "name":    "param_server",
        "cmd":     [VENV_PYTHON, f"{SRC}/core/param_server.py"],
        "delay":   1,     # Start after bus
    },
    {
        "name":    "sensor_hub",
        "cmd":     [VENV_PYTHON, f"{SRC}/sensors/sensor_hub.py"],
        "delay":   2,
    },
    {
        "name":    "camera",
        "cmd":     [VENV_PYTHON, f"{SRC}/sensors/camera_capture.py"],
        "delay":   3,
    },
    {
        "name":    "magi1_celebi",
        "cmd":     [VENV_PYTHON, f"{SRC}/magi1/celebi.py"],
        "delay":   4,
    },
    {
        "name":    "magi2_gengar",
        "cmd":     [VENV_PYTHON, f"{SRC}/magi2/gengar.py"],
        "delay":   4,
    },
    {
        "name":    "magi3_lugia",
        "cmd":     [VENV_PYTHON, f"{SRC}/magi3/lugia.py"],
        "delay":   6,     # Start last — depends on magi1+magi2
    },
    {
        "name":    "batch_manager",
        "cmd":     [VENV_PYTHON, f"{SRC}/core/batch_manager.py"],
        "delay":   5,
    },
    {
        "name":    "dashboard",
        "cmd":     [VENV_PYTHON, f"{SRC}/core/dashboard.py"],
        "delay":   6,
    },
]

MAX_RESTARTS   = 10     # Max restarts before giving up
BACKOFF_BASE   = 2.0    # Exponential backoff base (seconds)
CHECK_INTERVAL = 2.0    # Faster monitoring interval
HEARTBEAT_TIMEOUT = 5.0  # Max seconds before node is declared hung

_running = True


def _shutdown(sig, frame):
    global _running
    log.info("Watchdog shutting down — stopping all MAGI processes...")
    _running = False


signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT,  _shutdown)


class HeartbeatMonitor(threading.Thread):
    """Subscribes to ZMQ diagnostics and records timestamps of node heartbeats."""
    def __init__(self):
        super().__init__(daemon=True)
        self.last_heartbeat = {}
        self._running = True

    def run(self):
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        # Wait a bit for bus socket to be created
        time.sleep(0.5)
        try:
            sock.connect("ipc:///tmp/magi/bus_sub.sock")
            sock.setsockopt(zmq.SUBSCRIBE, b"/diagnostics")
            sock.setsockopt(zmq.RCVTIMEO, 500)
            log.info("HeartbeatMonitor connected to bus_sub.sock")
        except Exception as e:
            log.error(f"HeartbeatMonitor ZMQ subscription failed: {e}")
            return

        while self._running:
            try:
                parts = sock.recv_multipart()
                if len(parts) >= 2:
                    msg = decode(parts[1])
                    node_id = msg.node_id
                    self.last_heartbeat[node_id] = time.time()
            except zmq.Again:
                pass
            except Exception as e:
                log.debug(f"Heartbeat decoding error: {e}")

        sock.close()

    def stop(self):
        self._running = False


class ManagedProcess:
    def __init__(self, spec: dict):
        self.name      = spec["name"]
        self.cmd       = spec["cmd"]
        self.delay     = spec["delay"]
        self.proc      = None
        self.restarts  = 0
        self.last_fail = 0.0
        self.start_time = 0.0

        # Map service name to ZMQ node_id
        node_map = {
            "magi1_celebi": "magi1",
            "magi2_gengar": "magi2",
            "magi3_lugia": "magi3",
            "sensor_hub": "sensor_hub",
            "camera": "camera",
            "batch_manager": "batch_manager",
            "dashboard": "dashboard",
        }
        self.node_id = node_map.get(self.name)

    def start(self):
        log.info(f"Starting {self.name}...")
        env = os.environ.copy()
        env["PYTHONPATH"] = SRC
        self.proc = subprocess.Popen(
            self.cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.start_time = time.time()
        log.info(f"{self.name} started (PID {self.proc.pid})")

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.proc and self.is_alive():
            log.info(f"Terminating {self.name} (PID {self.proc.pid})...")
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2.0)
                log.info(f"{self.name} terminated cleanly.")
            except subprocess.TimeoutExpired:
                log.warning(f"{self.name} failed to terminate, killing...")
                self.proc.kill()
                self.proc.wait()

    def flush_shared_memory(self):
        """Cleanly unlink shared memory to prevent corruption or locking on restart."""
        if self.name in ["camera", "magi1_celebi", "magi2_gengar"]:
            try:
                import posix_ipc
                posix_ipc.unlink_shared_memory("/magi_camera_frame")
                log.warning(f"Watchdog: Flushed shared memory block '/magi_camera_frame' for {self.name}")
            except Exception as e:
                # Normal if it's already unlinked or mock layer is active
                log.debug(f"Shared memory unlinking skipped/failed for {self.name}: {e}")

    def restart(self, reason: str = "crashed"):
        self.stop()
        self.flush_shared_memory()
        self.restarts += 1
        backoff = min(BACKOFF_BASE ** self.restarts, 60.0)
        log.warning(f"{self.name} {reason} (restart #{self.restarts}) — backoff {backoff:.1f}s")
        time.sleep(backoff)
        self.start()
        self.last_fail = time.time()


def main():
    log.info("MAGI Watchdog starting...")

    procs = [ManagedProcess(s) for s in SERVICES]

    # Start all processes with their configured delays
    for p in procs:
        time.sleep(p.delay)
        p.start()

    # Start heartbeat monitor
    monitor = HeartbeatMonitor()
    monitor.start()

    log.info("All MAGI processes started — watchdog monitoring active")

    while _running:
        time.sleep(CHECK_INTERVAL)
        now = time.time()
        for p in procs:
            # 1. Process crash check
            if not p.is_alive():
                if p.restarts >= MAX_RESTARTS:
                    log.error(f"{p.name} exceeded max restarts ({MAX_RESTARTS}). Giving up.")
                    continue
                p.restart(reason="crashed")
                # Reset heartbeat record on restart
                if p.node_id in monitor.last_heartbeat:
                    monitor.last_heartbeat[p.node_id] = now
                continue

            # 2. Heartbeat hang check (only for processes that publish heartbeats)
            if p.node_id:
                # Wait 10 seconds post-start before enforcing heartbeats (avoids boot race conditions)
                if now - p.start_time > 10.0:
                    last_h = monitor.last_heartbeat.get(p.node_id)
                    if last_h is None:
                        # Process is running but has NEVER sent a heartbeat
                        log.warning(f"{p.name} (node: {p.node_id}) has never published a ZMQ heartbeat! Declaring hung.")
                        p.restart(reason="hung (no initial heartbeat)")
                    elif now - last_h > HEARTBEAT_TIMEOUT:
                        log.error(f"{p.name} (node: {p.node_id}) has hung! Last heartbeat was {now - last_h:.1f} seconds ago.")
                        p.restart(reason=f"hung (heartbeat timeout {now - last_h:.1f}s)")
                        monitor.last_heartbeat[p.node_id] = now

    # Shutdown all
    log.info("Shutting down watchdog and child services...")
    monitor.stop()
    for p in procs:
        p.stop()
        p.flush_shared_memory()
    monitor.join(timeout=1.0)
    log.info("All MAGI processes stopped.")


if __name__ == "__main__":
    main()
