"""
MAGI OS v2 — Lifecycle Node Base Class
src/core/lifecycle.py

Implements the ROS2 Managed Node lifecycle state machine.
All MAGI nodes inherit from this instead of MAGIBaseNode.

States:  UNCONFIGURED → INACTIVE → ACTIVE → FINALIZED
                              ↑         ↓
                         deactivate() ←┘
"""

from __future__ import annotations
import os, sys, time, signal, threading, abc, zmq, msgpack
sys.path.insert(0, "/opt/magi/src")

from common.logger import get_logger
from core.qos import get_qos, Durability, Reliability

# ─── Lifecycle States ────────────────────────────────────────────────────────

class State:
    UNCONFIGURED = "UNCONFIGURED"
    INACTIVE     = "INACTIVE"
    ACTIVE       = "ACTIVE"
    FINALIZED    = "FINALIZED"

BUS_PUB = "ipc:///tmp/magi/bus_pub.sock"   # Connect publishers here
BUS_SUB = "ipc:///tmp/magi/bus_sub.sock"   # Connect subscribers here


class LifecycleNode(abc.ABC):
    """
    ROS2-style Managed Node base class.
    
    Subclasses implement:
        on_configure()   → allocate resources, load model
        on_activate()    → start publishing / processing
        on_deactivate()  → stop publishing, keep resources
        on_cleanup()     → release all resources
        on_shutdown()    → emergency stop
    """

    def __init__(self, node_id: str, cpu_core: int = 0):
        self.node_id   = node_id
        self.cpu_core  = cpu_core
        self.log       = get_logger(node_id)
        self._state    = State.UNCONFIGURED
        self._running  = False
        self._subs     = {}     # topic → (socket, callback)
        self._pubs     = {}     # topic → socket
        self._ctx      = zmq.Context.instance()
        self._seq      = 0
        self._lock     = threading.Lock()
        self._diag_pub = None
        self._running_heartbeat = False
        self._heartbeat_thread = None

        # CPU pinning
        try:
            os.sched_setaffinity(0, {cpu_core})
            self.log.info(f"Pinned to CPU core {cpu_core}")
        except (AttributeError, OSError):
            pass

        signal.signal(signal.SIGTERM, lambda s, f: self.shutdown())
        signal.signal(signal.SIGINT,  lambda s, f: self.shutdown())

        self.log.info(f"Node '{node_id}' created — state: {self._state}")

    # ─── State Machine ───────────────────────────────────────────────────────

    def configure(self) -> bool:
        if self._state != State.UNCONFIGURED:
            self.log.warning(f"configure() called in state {self._state}")
            return False
        self.log.info("Configuring...")
        try:
            self._setup_diag_pub()
            self.on_configure()
            self._state = State.INACTIVE
            self._running = True
            
            # Start 1 Hz heartbeat diagnostics
            self._running_heartbeat = True
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat_thread.start()
            
            self.log.info(f"State → {self._state}")
            return True
        except Exception as e:
            self.log.error(f"configure() failed: {e}", exc_info=True)
            return False

    def activate(self) -> bool:
        if self._state != State.INACTIVE:
            self.log.warning(f"activate() called in state {self._state}")
            return False
        self.log.info("Activating...")
        try:
            self.on_activate()
            self._state  = State.ACTIVE
            self.log.info(f"State → {self._state}")
            return True
        except Exception as e:
            self.log.error(f"activate() failed: {e}", exc_info=True)
            return False

    def deactivate(self) -> bool:
        if self._state != State.ACTIVE:
            return False
        try:
            self.on_deactivate()
        except Exception as e:
            self.log.error(f"deactivate() failed: {e}")
        self._state = State.INACTIVE
        self.log.info(f"State → {self._state}")
        return True

    def cleanup(self) -> bool:
        if self._state == State.ACTIVE:
            self.deactivate()
        self._running_heartbeat = False
        try:
            self.on_cleanup()
        except Exception as e:
            self.log.error(f"cleanup() failed: {e}")
        self._state = State.UNCONFIGURED
        self.log.info(f"State → {self._state}")
        return True

    def shutdown(self):
        self.log.info("Shutdown requested")
        self._running = False
        self._running_heartbeat = False
        try:
            self.on_shutdown()
        except Exception:
            pass
        self._state = State.FINALIZED
        self._close_all_sockets()

    @property
    def state(self) -> str:
        return self._state

    # ─── Publisher / Subscriber API ──────────────────────────────────────────

    def create_publisher(self, topic: str) -> None:
        """Create a ZMQ PUB socket connected to the broker."""
        sock = self._ctx.socket(zmq.PUB)
        sock.connect(BUS_PUB)
        time.sleep(0.05)
        self._pubs[topic] = sock
        self.log.debug(f"Publisher created for {topic}")

    def publish(self, topic: str, msg) -> None:
        """Serialize and publish a typed message on a topic."""
        if topic not in self._pubs:
            self.log.error(f"No publisher for {topic}")
            return
        from core.messages import encode
        raw = encode(msg)
        self._pubs[topic].send_multipart([topic.encode(), raw])
        self._seq += 1

    def create_subscription(self, topic: str, callback, timeout_ms: int = 100) -> None:
        """Subscribe to a topic with a typed message callback."""
        sock = self._ctx.socket(zmq.SUB)
        sock.connect(BUS_SUB)
        sock.setsockopt(zmq.SUBSCRIBE, topic.encode())
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._subs[topic] = (sock, callback)
        self.log.debug(f"Subscribed to {topic}")

    def spin_once(self) -> None:
        """Process one pending message from all subscriptions."""
        for topic, (sock, cb) in self._subs.items():
            try:
                parts = sock.recv_multipart()
                from core.messages import decode
                cb(decode(parts[1]))
            except zmq.Again:
                pass
            except Exception as e:
                self.log.error(f"Callback error on {topic}: {e}")

    def spin(self) -> None:
        """Block and process messages until shutdown."""
        self.log.info("Spinning...")
        while self._running:
            self.spin_once()

    # ─── Diagnostics ─────────────────────────────────────────────────────────

    def _setup_diag_pub(self):
        self.create_publisher("/diagnostics")

    def publish_diag(self, status: str, message: str, values: dict = None):
        from core.messages import DiagMsg, DiagValue, Header
        msg = DiagMsg(
            header  = Header(node_id=self.node_id),
            node_id = self.node_id,
            status  = status,
            message = message,
            state   = self._state,
            values  = [DiagValue(k, str(v)) for k, v in (values or {}).items()]
        )
        self.publish("/diagnostics", msg)

    def _heartbeat_loop(self):
        self.log.info("Diagnostics heartbeat thread active")
        while self._running_heartbeat:
            try:
                self.publish_diag(
                    status="OK",
                    message="Periodic heartbeat",
                    values={"uptime_s": round(time.time() - self._seq * 1.0, 1)}
                )
            except Exception as e:
                if self._running_heartbeat:
                    self.log.debug(f"Heartbeat send suppressed during shutdown: {e}")
            time.sleep(1.0)
        self.log.info("Diagnostics heartbeat thread inactive")

    # ─── Next sequence number ─────────────────────────────────────────────────

    def next_header(self, frame_id: str = "base"):
        from core.messages import Header
        h = Header(frame_id=frame_id, seq=self._seq, node_id=self.node_id)
        return h

    # ─── Cleanup ─────────────────────────────────────────────────────────────

    def _close_all_sockets(self):
        for sock, _ in self._subs.values():
            sock.close()
        for sock in self._pubs.values():
            sock.close()

    # ─── Abstract hooks ──────────────────────────────────────────────────────

    def on_configure(self):   pass
    def on_activate(self):    pass
    def on_deactivate(self):  pass
    def on_cleanup(self):     pass
    def on_shutdown(self):    pass

    # ─── Boot helper ─────────────────────────────────────────────────────────

    def boot(self) -> None:
        """Full lifecycle: configure → activate → spin."""
        if self.configure() and self.activate():
            self.spin()
        self.shutdown()
