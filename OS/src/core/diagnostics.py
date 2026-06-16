"""
MAGI OS v2 — Diagnostics Monitor
src/core/diagnostics.py

Aggregates /diagnostics messages from all nodes.
Reports system health and alerts on STALE / ERROR nodes.
Equivalent to ROS2 ros_diagnostics.
"""

import os, sys, time, signal, zmq, msgpack
sys.path.insert(0, "/opt/magi/src")

try:
    os.sched_setaffinity(0, {0})
except (AttributeError, OSError):
    pass

from common.logger import get_logger

log = get_logger("diagnostics")

BUS_SUB  = "tcp://localhost:5556"
STALE_TIMEOUT = 5.0     # Mark node STALE if no diag in 5 seconds
EXPECTED_NODES = {"sensor_hub", "magi1", "magi2", "magi3", "camera"}


class DiagnosticsMonitor:
    """
    Subscribes to /diagnostics from all nodes and maintains
    a health dashboard. Alerts on ERROR/STALE nodes.
    """

    def __init__(self):
        self._ctx     = zmq.Context()
        self._sock    = self._ctx.socket(zmq.SUB)
        self._sock.connect(BUS_SUB)
        self._sock.setsockopt(zmq.SUBSCRIBE, b"/diagnostics")
        self._sock.setsockopt(zmq.RCVTIMEO, 500)
        self._running = True
        self._nodes: dict[str, dict] = {}    # node_id → last diag

        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT,  self._shutdown)

        log.info("Diagnostics monitor running")

    def _check_stale(self):
        """Mark nodes as STALE if no heartbeat received recently."""
        now = time.time()
        for node_id, info in self._nodes.items():
            age = now - info.get("ts", 0)
            if age > STALE_TIMEOUT and info.get("status") != "STALE":
                self._nodes[node_id]["status"]  = "STALE"
                self._nodes[node_id]["message"] = f"No heartbeat for {age:.1f}s"
                log.warning(f"Node '{node_id}' is STALE ({age:.1f}s)")

    def run(self):
        last_report = 0.0
        while self._running:
            try:
                parts = self._sock.recv_multipart()
                data  = msgpack.unpackb(parts[1], raw=False)
                nid   = data.get("node_id", "unknown")
                self._nodes[nid] = {
                    "status":  data.get("status", "OK"),
                    "message": data.get("message", ""),
                    "state":   data.get("state", "?"),
                    "values":  data.get("values", []),
                    "ts":      time.time(),
                }
                if data.get("status") == "ERROR":
                    log.error(f"[DIAG] {nid}: ERROR — {data.get('message','')}")
            except zmq.Again:
                pass

            self._check_stale()

            if time.time() - last_report > 10.0:
                self._print_summary()
                last_report = time.time()

        self._sock.close()

    def _print_summary(self):
        log.info("─── Diagnostics Summary ──────────────────────")
        for nid in sorted(EXPECTED_NODES | set(self._nodes.keys())):
            info = self._nodes.get(nid, {"status": "MISSING", "state": "?", "message": "Never reported"})
            icon = {"OK":"✅","WARN":"⚠️","ERROR":"🔴","STALE":"🟡","MISSING":"❌"}.get(info["status"],"?")
            log.info(f"  {icon} {nid:<16} [{info['state']:<14}] {info['message']}")
        log.info("──────────────────────────────────────────────")

    def get_status(self) -> dict:
        return dict(self._nodes)

    def _shutdown(self, sig, frame):
        self._running = False


if __name__ == "__main__":
    DiagnosticsMonitor().run()
