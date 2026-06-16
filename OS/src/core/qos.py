"""
MAGI OS v2 — QoS Profiles
src/core/qos.py

Per-topic Quality of Service — equivalent to ROS2 QoS policies.
Defines delivery guarantees for every topic in the system.
"""

from dataclasses import dataclass


# ─── QoS Constants ──────────────────────────────────────────────────────────

class Reliability:
    RELIABLE     = "RELIABLE"      # Every message delivered, retry on fail
    BEST_EFFORT  = "BEST_EFFORT"   # Drop if slow — lowest latency


class History:
    KEEP_LAST = "KEEP_LAST"   # Keep only last N messages
    KEEP_ALL  = "KEEP_ALL"    # Keep all messages (bounded by memory)


class Durability:
    VOLATILE         = "VOLATILE"          # Messages lost if no subscriber yet
    TRANSIENT_LOCAL  = "TRANSIENT_LOCAL"   # Latch — new subscribers get last msg


# ─── QoS Profile ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class QoSProfile:
    reliability: str = Reliability.BEST_EFFORT
    history:     str = History.KEEP_LAST
    depth:       int = 1
    durability:  str = Durability.VOLATILE


# ─── Standard Profiles (mirrors ROS2 built-in profiles) ─────────────────────

class QoS:
    """Pre-defined QoS profiles for common use cases."""

    # Real-time sensor data — drop stale, never queue
    SENSOR_DATA = QoSProfile(
        reliability = Reliability.BEST_EFFORT,
        history     = History.KEEP_LAST,
        depth       = 1,
        durability  = Durability.VOLATILE,
    )

    # Camera frames — absolute latest only
    CAMERA = QoSProfile(
        reliability = Reliability.BEST_EFFORT,
        history     = History.KEEP_LAST,
        depth       = 1,
        durability  = Durability.VOLATILE,
    )

    # Detection events — reliable, small buffer
    DETECTIONS = QoSProfile(
        reliability = Reliability.RELIABLE,
        history     = History.KEEP_LAST,
        depth       = 10,
        durability  = Durability.VOLATILE,
    )

    # Scene analysis — reliable, small buffer
    SCENE = QoSProfile(
        reliability = Reliability.RELIABLE,
        history     = History.KEEP_LAST,
        depth       = 5,
        durability  = Durability.VOLATILE,
    )

    # Decision output — TRANSIENT_LOCAL (new subscribers get last decision)
    DECISION = QoSProfile(
        reliability = Reliability.RELIABLE,
        history     = History.KEEP_LAST,
        depth       = 1,
        durability  = Durability.TRANSIENT_LOCAL,
    )

    # Diagnostics — best effort, small buffer
    DIAGNOSTICS = QoSProfile(
        reliability = Reliability.BEST_EFFORT,
        history     = History.KEEP_LAST,
        depth       = 5,
        durability  = Durability.VOLATILE,
    )

    # Parameters — always reliable, larger history
    PARAMETERS = QoSProfile(
        reliability = Reliability.RELIABLE,
        history     = History.KEEP_ALL,
        depth       = 100,
        durability  = Durability.TRANSIENT_LOCAL,
    )

    # TF transforms — reliable, keep last per frame pair
    TF = QoSProfile(
        reliability = Reliability.RELIABLE,
        history     = History.KEEP_LAST,
        depth       = 10,
        durability  = Durability.VOLATILE,
    )


# ─── Topic Registry ──────────────────────────────────────────────────────────

TOPIC_QOS: dict[str, QoSProfile] = {
    "/sensors":     QoS.SENSOR_DATA,
    "/camera":      QoS.CAMERA,
    "/detections":  QoS.DETECTIONS,
    "/scene":       QoS.SCENE,
    "/decision":    QoS.DECISION,
    "/diagnostics": QoS.DIAGNOSTICS,
    "/parameters":  QoS.PARAMETERS,
    "/tf":          QoS.TF,
}


def get_qos(topic: str) -> QoSProfile:
    """Return the QoS profile for a given topic."""
    return TOPIC_QOS.get(topic, QoS.SENSOR_DATA)
