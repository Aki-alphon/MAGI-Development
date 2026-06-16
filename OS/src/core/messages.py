"""
MAGI OS v2 — Typed Message System
src/core/messages.py

Industry-standard typed messages — equivalent to ROS2 .msg files.
All inter-node communication uses these dataclasses, serialized
with msgpack for minimal overhead.

Message type string format: "category/name/version"
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import time


# ─── Primitive sub-types ────────────────────────────────────────────────────

@dataclass
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


@dataclass
class BoundingBox:
    ymin: float = 0.0
    xmin: float = 0.0
    ymax: float = 0.0
    xmax: float = 0.0


@dataclass
class Header:
    """Standard message header — every message has one."""
    stamp:    float = field(default_factory=time.monotonic)
    wall_ts:  float = field(default_factory=time.time)
    seq:      int   = 0
    frame_id: str   = "base"
    node_id:  str   = ""


# ─── /sensors topic ─────────────────────────────────────────────────────────

@dataclass
class IMUData:
    MSG_TYPE = "sensor/imu/v1"
    accel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    gyro:  list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    temp:  float       = 0.0


@dataclass
class ToFData:
    MSG_TYPE = "sensor/tof/v1"
    distance_mm: float = -1.0
    status:      int   = 0


@dataclass
class GPSData:
    MSG_TYPE = "sensor/gps/v1"
    lat:      float = 0.0
    lon:      float = 0.0
    alt:      float = 0.0
    fix:      int   = 0
    sats:     int   = 0


@dataclass
class SensorMsg:
    """Published on /sensors — master sensor packet."""
    MSG_TYPE = "sensor/bundle/v1"
    header:  Header             = field(default_factory=Header)
    imu:     IMUData | None     = None
    tof:     ToFData | None     = None
    gps:     GPSData | None     = None
    gpio:    dict[str, bool]    = field(default_factory=dict)
    uart:    dict[str, str]     = field(default_factory=dict)
    adc:     dict[str, float]   = field(default_factory=dict)


# ─── /detections topic (MAGI-1 output) ──────────────────────────────────────

@dataclass
class Detection:
    label:      str          = ""
    class_id:   int          = 0
    confidence: float        = 0.0
    bbox:       BoundingBox  = field(default_factory=BoundingBox)
    track_id:   int          = -1     # -1 = not tracked


@dataclass
class DetectionMsg:
    """Published on /detections by MAGI-1 Melchior."""
    MSG_TYPE = "vision/detections/v1"
    header:     Header          = field(default_factory=Header)
    detections: list[Detection] = field(default_factory=list)
    count:      int             = 0
    fps:        float           = 0.0
    latency_ms: float           = 0.0


# ─── /scene topic (MAGI-2 output) ──────────────────────────────────────────

@dataclass
class MotionState:
    motion_mag: float = 0.0
    jerk:       float = 0.0
    is_moving:  bool  = False
    velocity:   Vec3  = field(default_factory=Vec3)


@dataclass
class SceneMsg:
    """Published on /scene by MAGI-2 Balthasar."""
    MSG_TYPE = "vision/scene/v1"
    header:        Header      = field(default_factory=Header)
    scene:         str         = "unknown"
    scene_id:      int         = -1
    confidence:    float       = 0.0
    top3:          list[dict]  = field(default_factory=list)
    anomaly_score: float       = 0.0
    motion:        MotionState = field(default_factory=MotionState)
    latency_ms:    float       = 0.0


# ─── /decision topic (MAGI-3 output) ────────────────────────────────────────

@dataclass
class DecisionMsg:
    """Published on /decision by MAGI-3 Caspar. TRANSIENT_LOCAL QoS."""
    MSG_TYPE = "fusion/decision/v1"
    header:        Header          = field(default_factory=Header)
    action:        str             = "IDLE"
    priority:      int             = 0
    targets:       list[Detection] = field(default_factory=list)
    reason:        str             = ""
    scene:         str             = "unknown"
    anomaly_score: float           = 0.0
    tof_mm:        float           = -1.0
    is_moving:     bool            = False
    confidence:    float           = 0.0


# ─── /diagnostics topic ──────────────────────────────────────────────────────

class DiagStatus:
    OK    = "OK"
    WARN  = "WARN"
    ERROR = "ERROR"
    STALE = "STALE"


@dataclass
class DiagValue:
    key:   str = ""
    value: str = ""


@dataclass
class DiagMsg:
    """Published on /diagnostics by every node every second."""
    MSG_TYPE = "system/diag/v1"
    header:    Header          = field(default_factory=Header)
    node_id:   str             = ""
    status:    str             = DiagStatus.OK
    message:   str             = ""
    values:    list[DiagValue] = field(default_factory=list)
    state:     str             = "UNKNOWN"     # Lifecycle state


# ─── /tf topic ───────────────────────────────────────────────────────────────

@dataclass
class Transform:
    translation: Vec3       = field(default_factory=Vec3)
    rotation:    Quaternion = field(default_factory=Quaternion)


@dataclass
class TransformStamped:
    MSG_TYPE = "geometry/transform_stamped/v1"
    header:        Header    = field(default_factory=Header)
    child_frame:   str       = ""
    transform:     Transform = field(default_factory=Transform)


@dataclass
class TFMsg:
    """Published on /tf — batch of transforms."""
    MSG_TYPE = "geometry/tf/v1"
    transforms: list[TransformStamped] = field(default_factory=list)


# ─── /parameters topic ───────────────────────────────────────────────────────

@dataclass
class ParamEvent:
    MSG_TYPE = "system/param_event/v1"
    header:    Header = field(default_factory=Header)
    node_id:   str    = ""
    key:       str    = ""
    old_value: Any    = None
    new_value: Any    = None


# ─── Serialization helpers ───────────────────────────────────────────────────

import msgpack


def encode(msg) -> bytes:
    """Serialize a dataclass message to msgpack bytes."""
    d = asdict(msg) if hasattr(msg, "__dataclass_fields__") else msg
    d["__type__"] = getattr(msg, "MSG_TYPE", "unknown")
    return msgpack.packb(d, use_bin_type=True)


def decode(raw: bytes) -> dict:
    """Deserialize msgpack bytes to a dict (with __type__ key)."""
    return msgpack.unpackb(raw, raw=False)
