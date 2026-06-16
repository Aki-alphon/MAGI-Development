"""
MAGI OS v2 — Parameter Server
src/core/param_server.py

Runtime key-value store accessible by all nodes.
Nodes can get/set parameters without restarting.
Change events are published to /parameters topic.

Protocol (ZMQ REQ/REP):
    GET  → {"cmd":"get",  "node":"magi1", "key":"confidence_threshold"}
    SET  → {"cmd":"set",  "node":"magi1", "key":"confidence_threshold", "value":0.6}
    LIST → {"cmd":"list", "node":"magi1"}
    ALL  → {"cmd":"all"}
"""

import os, sys, time, signal, threading, yaml, zmq, msgpack
sys.path.insert(0, "/opt/magi/src")

try:
    os.sched_setaffinity(0, {0})
except (AttributeError, OSError):
    pass

from common.logger import get_logger

log = get_logger("param_server")

PARAM_ADDR = "ipc:///tmp/magi/param_server.sock"   # REQ/REP for param get/set
BUS_PUB    = "ipc:///tmp/magi/bus_pub.sock"  # Publish /parameters events


class ParamServer:
    """
    Centralized parameter store with change notifications.
    Loaded from config.yaml on startup, then updated at runtime.
    """

    def __init__(self, config_path: str = "/opt/magi/config/config.yaml"):
        self._params: dict[str, dict] = {}   # {node_id: {key: value}}
        self._lock   = threading.Lock()
        self._running = True
        self._ctx    = zmq.Context()

        # REP socket — answers GET/SET requests
        self.rep = self._ctx.socket(zmq.REP)
        self.rep.bind(PARAM_ADDR)

        # PUB socket — notifies subscribers of changes
        self.pub = self._ctx.socket(zmq.PUB)
        self.pub.connect(BUS_PUB)
        time.sleep(0.1)

        self._load_from_config(config_path)
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT,  self._shutdown)

        log.info(f"Parameter server ready on {PARAM_ADDR}")

    def _load_from_config(self, path: str):
        """Seed parameters from config.yaml."""
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f)

            models = cfg.get("models", {})
            for node_id, mcfg in models.items():
                self._params[f"magi{node_id[-1]}"] = {
                    "confidence_threshold": mcfg.get("confidence_threshold", 0.5),
                    "num_threads":          mcfg.get("num_threads", 1),
                    "use_xnnpack":          mcfg.get("use_xnnpack", True),
                    "inference_timeout_ms": mcfg.get("inference_timeout_ms", 200),
                }

            self._params["system"] = {
                "log_level":    cfg.get("system", {}).get("log_level", "INFO"),
                "poll_rate_hz": cfg.get("sensors", {}).get("poll_rate_hz", 50),
            }

            log.info(f"Loaded params for nodes: {list(self._params.keys())}")
        except Exception as e:
            log.warning(f"Could not load config: {e}")

    def get(self, node_id: str, key: str):
        with self._lock:
            return self._params.get(node_id, {}).get(key)

    def set(self, node_id: str, key: str, value) -> bool:
        with self._lock:
            if node_id not in self._params:
                self._params[node_id] = {}
            old = self._params[node_id].get(key)
            self._params[node_id][key] = value

        # Publish change event to /parameters topic
        event = {
            "topic":    "/parameters",
            "node_id":  node_id,
            "key":      key,
            "old":      old,
            "new":      value,
            "ts":       time.time(),
        }
        self.pub.send_multipart([
            b"/parameters",
            msgpack.packb(event, use_bin_type=True)
        ])
        log.info(f"Param SET {node_id}.{key} = {value!r}  (was {old!r})")
        return True

    def list_node(self, node_id: str) -> dict:
        with self._lock:
            return dict(self._params.get(node_id, {}))

    def all_params(self) -> dict:
        with self._lock:
            return {k: dict(v) for k, v in self._params.items()}

    def run(self):
        log.info("Parameter server running...")
        while self._running:
            try:
                if not self.rep.poll(timeout=500):
                    continue
                raw = self.rep.recv()
                req = msgpack.unpackb(raw, raw=False)
                cmd = req.get("cmd", "")

                if cmd == "get":
                    val = self.get(req["node"], req["key"])
                    self.rep.send(msgpack.packb({"value": val, "found": val is not None}, use_bin_type=True))

                elif cmd == "set":
                    ok = self.set(req["node"], req["key"], req["value"])
                    self.rep.send(msgpack.packb({"ok": ok}, use_bin_type=True))

                elif cmd == "list":
                    self.rep.send(msgpack.packb({"params": self.list_node(req.get("node",""))}, use_bin_type=True))

                elif cmd == "all":
                    self.rep.send(msgpack.packb({"params": self.all_params()}, use_bin_type=True))

                else:
                    self.rep.send(msgpack.packb({"error": f"Unknown: {cmd}"}, use_bin_type=True))

            except Exception as e:
                log.error(f"Param server error: {e}")
                try:
                    self.rep.send(msgpack.packb({"error": str(e)}, use_bin_type=True))
                except Exception:
                    pass

        self.rep.close()
        self.pub.close()
        log.info("Parameter server stopped.")

    def _shutdown(self, sig, frame):
        self._running = False


# ─── Client helper (used by nodes) ──────────────────────────────────────────

class ParamClient:
    """Used by each node to access the parameter server."""

    ADDR = "ipc:///tmp/magi/param_server.sock"

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._ctx  = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.connect(self.ADDR)
        self._sock.setsockopt(zmq.RCVTIMEO, 500)
        self._sock.setsockopt(zmq.SNDTIMEO, 500)
        self._local: dict = {}   # Local cache

    def get(self, key: str, default=None):
        """Get a parameter, with local cache fallback."""
        if key in self._local:
            return self._local[key]
        try:
            self._sock.send(msgpack.packb({"cmd":"get","node":self.node_id,"key":key}, use_bin_type=True))
            resp = msgpack.unpackb(self._sock.recv(), raw=False)
            val  = resp.get("value", default)
            if val is not None:
                self._local[key] = val
            return val
        except Exception:
            return default

    def set(self, key: str, value) -> bool:
        try:
            self._sock.send(msgpack.packb({"cmd":"set","node":self.node_id,"key":key,"value":value}, use_bin_type=True))
            resp = msgpack.unpackb(self._sock.recv(), raw=False)
            self._local[key] = value
            return resp.get("ok", False)
        except Exception:
            return False


if __name__ == "__main__":
    ParamServer().run()
