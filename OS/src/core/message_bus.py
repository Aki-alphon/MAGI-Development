"""
MAGI OS v2 — Message Bus (XPUB/XSUB Broker)
src/core/message_bus.py

Central pub/sub broker — equivalent to roscore + DDS.
Uses ZeroMQ XPUB/XSUB proxy for many-to-many topic routing.

Architecture:
    Publishers  → XSUB frontend (port 5555)
    Subscribers ← XPUB backend  (port 5556)
    Control     ← REP socket    (port 5557) for stats/management

Features:
    - Dynamic many-to-many topic routing
    - TRANSIENT_LOCAL: last-value cache per topic
    - Message rate tracking per topic
    - Topic filtering (namespace support)
"""

import os
import sys
import time
import signal
import threading
import collections
import yaml
import zmq
import msgpack

sys.path.insert(0, "/opt/magi/src")

try:
    os.sched_setaffinity(0, {0})
except (AttributeError, OSError):
    pass

from common.logger import get_logger
from core.qos import TOPIC_QOS, Durability

log = get_logger("message_bus")

# Default socket addresses
FRONTEND_ADDR = "ipc:///tmp/magi/bus_pub.sock"   # Publishers connect here
BACKEND_ADDR  = "ipc:///tmp/magi/bus_sub.sock"   # Subscribers connect here
CONTROL_ADDR  = "ipc:///tmp/magi/bus_ctl.sock"   # CLI / stats connect here


class MessageBus:
    """
    XPUB/XSUB broker with TRANSIENT_LOCAL last-value cache,
    message rate tracking, and a control interface.
    """

    def __init__(self):
        self.ctx      = zmq.Context()
        self._running = True

        # XSUB — receives from all publishers
        self.frontend = self.ctx.socket(zmq.XSUB)
        self.frontend.bind(FRONTEND_ADDR)

        # XPUB — forwards to all subscribers
        self.backend  = self.ctx.socket(zmq.XPUB)
        self.backend.setsockopt(zmq.XPUB_VERBOSE, 1)   # Get all sub events
        self.backend.bind(BACKEND_ADDR)

        # Control socket — for CLI queries
        self.control  = self.ctx.socket(zmq.REP)
        self.control.bind(CONTROL_ADDR)

        # TRANSIENT_LOCAL cache: topic → last raw message bytes
        self._last_msg: dict[str, bytes] = {}

        # Rate tracking: topic → deque of timestamps
        self._rates: dict[str, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=100)
        )

        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT,  self._shutdown)

        log.info(f"Message bus ready:")
        log.info(f"  Publishers  → {FRONTEND_ADDR}")
        log.info(f"  Subscribers ← {BACKEND_ADDR}")
        log.info(f"  Control     ← {CONTROL_ADDR}")

    def _extract_topic(self, frame: bytes) -> str:
        """Extract topic string from raw ZMQ message frame."""
        try:
            return frame.split(b"\x00")[0].decode("utf-8", errors="ignore").lstrip("/")
        except Exception:
            return ""

    def _handle_control(self):
        """Handle CLI/stats control requests (runs in thread)."""
        while self._running:
            try:
                if not self.control.poll(timeout=500):
                    continue
                req = msgpack.unpackb(self.control.recv(), raw=False)
                cmd = req.get("cmd", "")

                if cmd == "topic_list":
                    topics = list(self._last_msg.keys())
                    self.control.send(msgpack.packb({"topics": topics}, use_bin_type=True))

                elif cmd == "topic_hz":
                    topic = req.get("topic", "")
                    dq    = self._rates.get(topic, collections.deque())
                    now   = time.monotonic()
                    recent = [t for t in dq if now - t < 1.0]
                    hz     = len(recent)
                    self.control.send(msgpack.packb({"hz": hz, "topic": topic}, use_bin_type=True))

                elif cmd == "stats":
                    stats = {
                        t: {"hz": len([x for x in dq if time.monotonic() - x < 1.0])}
                        for t, dq in self._rates.items()
                    }
                    self.control.send(msgpack.packb({"stats": stats}, use_bin_type=True))

                else:
                    self.control.send(msgpack.packb({"error": f"Unknown cmd: {cmd}"}, use_bin_type=True))

            except Exception as e:
                log.warning(f"Control handler error: {e}")

    def run(self):
        """Run the XPUB/XSUB proxy with last-value cache."""
        log.info("Message bus running...")

        # Start control handler thread
        ctrl_thread = threading.Thread(target=self._handle_control, daemon=True)
        ctrl_thread.start()

        poller = zmq.Poller()
        poller.register(self.frontend, zmq.POLLIN)
        poller.register(self.backend,  zmq.POLLIN)

        while self._running:
            try:
                events = dict(poller.poll(timeout=100))
            except zmq.ZMQError:
                break

            # ── Publisher → Broker (XSUB receives) ───────────────────────
            if self.frontend in events:
                frames = self.frontend.recv_multipart()
                topic  = frames[0].decode("utf-8", errors="ignore") if frames else ""

                # Update rate tracker
                self._rates[topic].append(time.monotonic())

                # Cache for TRANSIENT_LOCAL topics
                from core.qos import TOPIC_QOS, Durability
                qos = TOPIC_QOS.get(f"/{topic}")
                if qos and qos.durability == Durability.TRANSIENT_LOCAL:
                    self._last_msg[topic] = frames

                # Forward to all subscribers
                self.backend.send_multipart(frames)

            # ── Subscriber → Broker (XPUB receives subscription) ─────────
            if self.backend in events:
                msg    = self.backend.recv()
                is_sub = msg[0] == 1
                topic  = msg[1:].decode("utf-8", errors="ignore")

                # Forward subscription to frontend
                self.frontend.send(msg)

                # TRANSIENT_LOCAL: send last cached message to new subscriber
                if is_sub and topic in self._last_msg:
                    log.debug(f"TRANSIENT_LOCAL: replaying /{topic} to new subscriber")
                    self.backend.send_multipart(self._last_msg[topic])

        self._cleanup()

    def _shutdown(self, sig, frame):
        log.info("Message bus shutting down...")
        self._running = False

    def _cleanup(self):
        self.frontend.close()
        self.backend.close()
        self.control.close()
        self.ctx.term()
        log.info("Message bus stopped.")


if __name__ == "__main__":
    bus = MessageBus()
    bus.run()
