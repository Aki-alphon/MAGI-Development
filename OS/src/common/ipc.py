"""
MAGI OS — IPC Utilities (ZeroMQ + msgpack)
/opt/magi/src/common/ipc.py

All inter-process messages use msgpack for fast binary serialization.
Large tensors (camera frames) use POSIX shared memory — only the shm
key is passed through ZeroMQ.
"""

import time
import zmq
import msgpack
import numpy as np
from typing import Any, Optional
from common.logger import get_logger

log = get_logger("ipc")


# ─── Serialization ──────────────────────────────────────────────────────────

def pack(data: Any) -> bytes:
    """Serialize data to msgpack bytes."""
    return msgpack.packb(data, use_bin_type=True)


def unpack(raw: bytes) -> Any:
    """Deserialize msgpack bytes to Python object."""
    return msgpack.unpackb(raw, raw=False)


# ─── Publisher ──────────────────────────────────────────────────────────────

class Publisher:
    """
    ZeroMQ PUB socket wrapper.
    Used by: sensor_hub → broadcasts sensor data to all MAGI nodes.
    """

    def __init__(self, address: str):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.bind(address)
        self.address = address
        # Allow subscribers time to connect
        time.sleep(0.1)
        log.info(f"Publisher bound to {address}")

    def publish(self, topic: str, data: Any):
        """Publish data under a topic string."""
        topic_bytes = topic.encode("utf-8")
        payload = pack(data)
        self.sock.send_multipart([topic_bytes, payload])

    def close(self):
        self.sock.close()


# ─── Subscriber ─────────────────────────────────────────────────────────────

class Subscriber:
    """
    ZeroMQ SUB socket wrapper.
    Used by: MAGI nodes ← subscribing to sensor_hub data.
    """

    def __init__(self, address: str, topics: list[str], timeout_ms: int = 500):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.connect(address)
        for topic in topics:
            self.sock.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        log.info(f"Subscriber connected to {address}, topics={topics}")

    def receive(self) -> Optional[tuple[str, Any]]:
        """
        Receive a message. Returns (topic, data) or None on timeout.
        """
        try:
            parts = self.sock.recv_multipart()
            topic = parts[0].decode("utf-8")
            data = unpack(parts[1])
            return topic, data
        except zmq.Again:
            return None

    def close(self):
        self.sock.close()


# ─── Push / Pull ────────────────────────────────────────────────────────────

class Pusher:
    """
    ZeroMQ PUSH socket.
    Used by: MAGI-1, MAGI-2 → send results to MAGI-3 (Lugia).
    """

    def __init__(self, address: str):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUSH)
        self.sock.bind(address)
        log.info(f"Pusher bound to {address}")

    def send(self, data: Any):
        self.sock.send(pack(data), zmq.NOBLOCK)

    def close(self):
        self.sock.close()


class Puller:
    """
    ZeroMQ PULL socket.
    Used by: MAGI-3 ← collects results from MAGI-1 and MAGI-2.
    """

    def __init__(self, addresses: list[str], timeout_ms: int = 100):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PULL)
        for addr in addresses:
            self.sock.connect(addr)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        log.info(f"Puller connected to {addresses}")

    def receive(self) -> Optional[Any]:
        try:
            return unpack(self.sock.recv())
        except zmq.Again:
            return None

    def close(self):
        self.sock.close()


# ─── Shared Memory Helper (large tensors / camera frames) ───────────────────

class SharedFrame:
    """
    POSIX shared memory for zero-copy camera frame passing.
    Writer (camera_capture) writes the frame.
    Readers (MAGI-1) read directly from shared memory.
    """

    def __init__(self, name: str, size: int, create: bool = False):
        import posix_ipc
        import mmap

        self.name = name
        self.size = size

        if create:
            try:
                posix_ipc.unlink_shared_memory(name)
            except posix_ipc.ExistentialError:
                pass
            self.shm = posix_ipc.SharedMemory(name, posix_ipc.O_CREX, size=size)
        else:
            self.shm = posix_ipc.SharedMemory(name)

        self.mmap = mmap.mmap(self.shm.fd, size)
        self.shm.close_fd()

    def write(self, frame: np.ndarray):
        data = frame.tobytes()
        self.mmap.seek(0)
        self.mmap.write(data[: self.size])

    def read(self, shape: tuple, dtype=np.uint8) -> np.ndarray:
        self.mmap.seek(0)
        data = self.mmap.read(self.size)
        return np.frombuffer(data, dtype=dtype).reshape(shape).copy()

    def close(self):
        self.mmap.close()
