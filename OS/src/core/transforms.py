"""
MAGI OS v2 — TF Transform Store
src/core/transforms.py

Lightweight TF2-like sensor frame coordinate transform registry.
Each sensor publishes its frame relationship to a parent frame.
MAGI-3 uses this for proper sensor fusion in 3D space.

Example frames:
    base_link → imu_link      (IMU offset from robot center)
    base_link → camera_link   (camera position + orientation)
    base_link → tof_front     (ToF sensor position)
    base_link → map           (global localization)
"""

from __future__ import annotations
import time
import math
import threading
from dataclasses import dataclass, field
from typing import Optional
from common.logger import get_logger

log = get_logger("tf")


# ─── Math Helpers ────────────────────────────────────────────────────────────

@dataclass
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, o): return Vec3(self.x+o.x, self.y+o.y, self.z+o.z)
    def as_list(self): return [self.x, self.y, self.z]


@dataclass
class Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    @staticmethod
    def from_euler(roll: float, pitch: float, yaw: float) -> "Quaternion":
        """Convert roll/pitch/yaw (radians) to quaternion."""
        cr, sr = math.cos(roll/2),  math.sin(roll/2)
        cp, sp = math.cos(pitch/2), math.sin(pitch/2)
        cy, sy = math.cos(yaw/2),   math.sin(yaw/2)
        return Quaternion(
            x = sr*cp*cy - cr*sp*sy,
            y = cr*sp*cy + sr*cp*sy,
            z = cr*cp*sy - sr*sp*cy,
            w = cr*cp*cy + sr*sp*sy,
        )

    def to_euler(self) -> tuple[float, float, float]:
        """Convert quaternion to (roll, pitch, yaw) radians."""
        sinr = 2*(self.w*self.x + self.y*self.z)
        cosr = 1 - 2*(self.x**2 + self.y**2)
        roll = math.atan2(sinr, cosr)

        sinp = 2*(self.w*self.y - self.z*self.x)
        pitch = math.copysign(math.pi/2, sinp) if abs(sinp) >= 1 else math.asin(sinp)

        siny = 2*(self.w*self.z + self.x*self.y)
        cosy = 1 - 2*(self.y**2 + self.z**2)
        yaw = math.atan2(siny, cosy)
        return roll, pitch, yaw

    def as_list(self): return [self.x, self.y, self.z, self.w]


@dataclass
class Transform:
    translation: Vec3       = field(default_factory=Vec3)
    rotation:    Quaternion = field(default_factory=Quaternion)


@dataclass
class TransformStamped:
    parent_frame: str      = "base_link"
    child_frame:  str      = ""
    transform:    Transform = field(default_factory=Transform)
    ts:           float    = field(default_factory=time.monotonic)


# ─── TF Buffer ────────────────────────────────────────────────────────────────

class TFBuffer:
    """
    Thread-safe transform registry.
    Stores the latest transform for each parent→child frame pair.
    """

    def __init__(self, max_age_s: float = 1.0):
        self._transforms: dict[tuple, TransformStamped] = {}
        self._lock    = threading.RLock()
        self._max_age = max_age_s

    def set_transform(self, ts: TransformStamped) -> None:
        key = (ts.parent_frame, ts.child_frame)
        with self._lock:
            self._transforms[key] = ts
        log.debug(f"TF set: {ts.parent_frame} → {ts.child_frame}")

    def get_transform(self, parent: str, child: str) -> Optional[TransformStamped]:
        key = (parent, child)
        with self._lock:
            ts = self._transforms.get(key)
        if ts is None:
            return None
        age = time.monotonic() - ts.ts
        if age > self._max_age:
            log.warning(f"TF {parent}→{child} is stale ({age:.2f}s)")
            return None
        return ts

    def all_frames(self) -> list[str]:
        with self._lock:
            frames = set()
            for p, c in self._transforms:
                frames.add(p)
                frames.add(c)
        return sorted(frames)

    def tree(self) -> str:
        with self._lock:
            lines = ["TF Tree:"]
            for (p, c), ts in self._transforms.items():
                age = time.monotonic() - ts.ts
                t   = ts.transform.translation
                lines.append(f"  {p} → {c}  xyz=[{t.x:.3f},{t.y:.3f},{t.z:.3f}]  age={age:.2f}s")
        return "\n".join(lines)


# ─── Static Frame Definitions ─────────────────────────────────────────────────

def build_default_tf(buf: TFBuffer) -> None:
    """
    Register default static transforms for common MAGI sensor layouts.
    Edit these to match your physical hardware mounting positions.
    """

    # IMU mounted at center of robot body (no offset)
    buf.set_transform(TransformStamped(
        parent_frame = "base_link",
        child_frame  = "imu_link",
        transform    = Transform(
            translation = Vec3(0.0, 0.0, 0.05),    # 5cm above base
            rotation    = Quaternion.from_euler(0, 0, 0),
        )
    ))

    # Camera mounted 10cm forward, 15cm up, pointing forward
    buf.set_transform(TransformStamped(
        parent_frame = "base_link",
        child_frame  = "camera_link",
        transform    = Transform(
            translation = Vec3(0.10, 0.0, 0.15),
            rotation    = Quaternion.from_euler(0, math.radians(-15), 0),  # 15° downward tilt
        )
    ))

    # ToF front sensor: 12cm forward, 8cm up
    buf.set_transform(TransformStamped(
        parent_frame = "base_link",
        child_frame  = "tof_front",
        transform    = Transform(
            translation = Vec3(0.12, 0.0, 0.08),
            rotation    = Quaternion.from_euler(0, 0, 0),
        )
    ))

    log.info(f"Static TF frames loaded: {buf.all_frames()}")


# ─── Global TF Buffer (singleton) ────────────────────────────────────────────

_tf_buffer = TFBuffer()
build_default_tf(_tf_buffer)


def get_tf() -> TFBuffer:
    """Return the global TF buffer."""
    return _tf_buffer
