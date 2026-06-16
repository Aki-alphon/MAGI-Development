"""
MAGI OS — Mock posix_ipc
docker/mock_hardware/posix_ipc.py

Replaces POSIX shared memory with in-process mmap on platforms
that don't support posix_ipc (Windows host, some Docker setups).
"""

import mmap
import os

O_CREX = os.O_CREAT | os.O_EXCL

ExistentialError = FileNotFoundError

_store: dict[str, bytes] = {}   # In-process shared memory store


def unlink_shared_memory(name: str):
    _store.pop(name, None)


class SharedMemory:
    """
    Mock POSIX shared memory using a plain bytearray.
    Works cross-platform (Windows Docker / macOS).
    """
    def __init__(self, name: str, flags: int = 0, size: int = 0):
        self.name = name
        self._size = size

        if flags & os.O_CREAT:
            _store[name] = bytearray(size)
        elif name not in _store:
            raise ExistentialError(f"Shared memory '{name}' not found")

        self._buf = _store[name]
        # Fake fd for compatibility — uses /dev/null
        self.fd = open(os.devnull, "rb").fileno()

    def close_fd(self):
        pass   # No real fd to close

    @property
    def size(self) -> int:
        return self._size

    def read(self, offset: int = 0, n: int = -1) -> bytes:
        if n < 0:
            return bytes(self._buf[offset:])
        return bytes(self._buf[offset:offset + n])

    def write(self, data: bytes, offset: int = 0):
        end = offset + len(data)
        self._buf[offset:end] = data

    def unlink(self):
        _store.pop(self.name, None)
