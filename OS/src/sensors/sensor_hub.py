"""
MAGI OS — Sensor Hub
/opt/magi/src/sensors/sensor_hub.py

Aggregates all connected sensors (I2C, SPI, UART, GPIO) and
publishes unified data packets to MAGI nodes via ZeroMQ PUB socket.

CPU Core 0 only (OS core — non-isolated).
"""

import os
import time
import signal
import threading
import yaml
import numpy as np

# Pin this process to core 0 (no-op in Docker/non-Linux)
try:
    os.sched_setaffinity(0, {0})
except (AttributeError, OSError):
    pass

from common.logger import get_logger
from common.ipc import Publisher, pack

log = get_logger("sensor_hub")

# ─── Load Config ────────────────────────────────────────────────────────────

with open("/opt/magi/config/config.yaml") as f:
    CFG = yaml.safe_load(f)

SENSOR_CFG = CFG["sensors"]
IPC_CFG    = CFG["ipc"]
POLL_RATE  = SENSOR_CFG["poll_rate_hz"]
POLL_DT    = 1.0 / POLL_RATE

_running = True


def _shutdown(sig, frame):
    global _running
    log.info("Sensor hub shutting down...")
    _running = False


signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT,  _shutdown)


# ─── I2C Sensors ────────────────────────────────────────────────────────────

class IMU_MPU6050:
    """MPU-6050 / MPU-9250 IMU over I2C."""

    REG_PWR_MGMT = 0x6B
    REG_ACCEL_XOUT = 0x3B
    SCALE_ACCEL = 16384.0   # ±2g range
    SCALE_GYRO  = 131.0     # ±250 deg/s range

    def __init__(self, bus: int, address: int):
        import smbus2
        self.bus  = smbus2.SMBus(bus)
        self.addr = address
        # Wake up sensor
        self.bus.write_byte_data(self.addr, self.REG_PWR_MGMT, 0)
        log.info(f"IMU MPU-6050 initialized at I2C 0x{address:02X}")

    def read(self) -> dict:
        """Read accel (g) and gyro (deg/s) values."""
        raw = self.bus.read_i2c_block_data(self.addr, self.REG_ACCEL_XOUT, 14)

        def to_int16(hi, lo):
            val = (hi << 8) | lo
            return val - 65536 if val > 32767 else val

        ax = to_int16(raw[0],  raw[1])  / self.SCALE_ACCEL
        ay = to_int16(raw[2],  raw[3])  / self.SCALE_ACCEL
        az = to_int16(raw[4],  raw[5])  / self.SCALE_ACCEL
        gx = to_int16(raw[8],  raw[9])  / self.SCALE_GYRO
        gy = to_int16(raw[10], raw[11]) / self.SCALE_GYRO
        gz = to_int16(raw[12], raw[13]) / self.SCALE_GYRO

        return {
            "accel": [round(ax, 4), round(ay, 4), round(az, 4)],
            "gyro":  [round(gx, 4), round(gy, 4), round(gz, 4)],
        }


class ToF_VL53L0X:
    """VL53L0X Time-of-Flight distance sensor over I2C."""

    def __init__(self, bus: int, address: int):
        try:
            import VL53L0X
            self.tof = VL53L0X.VL53L0X(i2c_bus=bus, i2c_address=address)
            self.tof.open()
            self.tof.start_ranging(VL53L0X.Vl53l0xAccuracyMode.BETTER)
            log.info(f"VL53L0X ToF initialized at I2C 0x{address:02X}")
        except Exception as e:
            log.warning(f"VL53L0X init failed: {e}")
            self.tof = None

    def read(self) -> dict:
        if self.tof is None:
            return {"distance_mm": -1}
        try:
            d = self.tof.get_distance()
            return {"distance_mm": d}
        except Exception:
            return {"distance_mm": -1}


# ─── GPIO Handler ───────────────────────────────────────────────────────────

class GPIOHandler:
    """Manages GPIO inputs with interrupt support and outputs."""

    def __init__(self, gpio_cfg: list):
        import pigpio
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpio daemon not running! Start with: sudo pigpiod")

        self.pins   = {}
        self._events = {}

        for pin_cfg in gpio_cfg:
            if not pin_cfg.get("enabled", False):
                continue

            pin  = pin_cfg["pin"]
            mode = pin_cfg["mode"]
            name = pin_cfg["name"]

            if mode == "input":
                pull = {"up": pigpio.PUD_UP, "down": pigpio.PUD_DOWN, "none": pigpio.PUD_OFF}
                self.pi.set_mode(pin, pigpio.INPUT)
                self.pi.set_pull_up_down(pin, pull.get(pin_cfg.get("pull", "none"), pigpio.PUD_OFF))

                if pin_cfg.get("interrupt"):
                    edge = {"rising": pigpio.RISING_EDGE,
                            "falling": pigpio.FALLING_EDGE,
                            "both": pigpio.EITHER_EDGE}
                    self._events[name] = False
                    self.pi.callback(pin,
                                     edge.get(pin_cfg["interrupt"], pigpio.EITHER_EDGE),
                                     lambda gpio, level, tick, n=name: self._set_event(n))

            elif mode == "output":
                self.pi.set_mode(pin, pigpio.OUTPUT)
                self.pi.write(pin, 0)

            self.pins[name] = pin
            log.info(f"GPIO pin {pin} ({name}) configured as {mode}")

    def _set_event(self, name: str):
        self._events[name] = True

    def read_events(self) -> dict:
        """Return and clear all pending interrupt events."""
        events = dict(self._events)
        self._events = {k: False for k in self._events}
        return events

    def write(self, name: str, value: int):
        if name in self.pins:
            self.pi.write(self.pins[name], value)

    def pulse(self, name: str, duration_ms: int = 100):
        """Pulse an output pin high then low."""
        self.write(name, 1)
        threading.Timer(duration_ms / 1000.0, self.write, args=[name, 0]).start()

    def cleanup(self):
        self.pi.stop()


# ─── UART Sensor ─────────────────────────────────────────────────────────────

class UARTSensor:
    """Generic UART sensor (GPS, telemetry MCU, etc.)."""

    def __init__(self, port: str, baud: int, name: str):
        import serial
        self.name = name
        try:
            self.ser = serial.Serial(port, baud, timeout=0.05)
            log.info(f"UART '{name}' opened on {port} @ {baud} baud")
        except Exception as e:
            log.warning(f"UART '{name}' failed: {e}")
            self.ser = None

    def read_line(self) -> str:
        if self.ser and self.ser.in_waiting:
            try:
                return self.ser.readline().decode("utf-8", errors="ignore").strip()
            except Exception:
                return ""
        return ""


# ─── Sensor Hub Main ─────────────────────────────────────────────────────────

# Broker frontend socket (same path used by LifecycleNode.create_publisher)
_BUS_PUB = "ipc:///tmp/magi/bus_pub.sock"


class SensorHub:
    def __init__(self):
        # Connect to the broker (not bind our own socket) so all LifecycleNodes
        # receive /sensors through the standard XPUB/XSUB message bus.
        import zmq, msgpack as _mp
        self._ctx  = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.connect(_BUS_PUB)
        time.sleep(0.2)   # Slow-joiner guard
        self._init_sensors()
        log.info(f"SensorHub ready — publishing /sensors through broker {_BUS_PUB}")

    def _init_sensors(self):
        """Initialize all enabled sensors from config."""
        self.sensors = {}
        bus = SENSOR_CFG["i2c"]["bus"]

        for dev in SENSOR_CFG["i2c"]["devices"]:
            if not dev.get("enabled", False):
                continue
            name, addr = dev["name"], dev["address"]
            try:
                if name == "imu":
                    self.sensors["imu"] = IMU_MPU6050(bus, addr)
                elif name == "tof_front":
                    self.sensors["tof"] = ToF_VL53L0X(bus, addr)
            except Exception as e:
                log.error(f"Failed to init sensor '{name}': {e}")

        # GPIO
        try:
            self.gpio = GPIOHandler(SENSOR_CFG["gpio"])
        except Exception as e:
            log.error(f"GPIO init failed: {e}")
            self.gpio = None

        # UART sensors
        self.uart_sensors = []
        for ucfg in SENSOR_CFG.get("uart", []):
            if ucfg.get("enabled", False):
                self.uart_sensors.append(UARTSensor(ucfg["port"], ucfg["baud"], ucfg["name"]))

    def _collect(self) -> dict:
        """Read all sensors and return unified data packet."""
        packet = {"ts": time.time(), "imu": None, "tof": None, "gpio": {}, "uart": {}}

        # IMU
        if "imu" in self.sensors:
            try:
                packet["imu"] = self.sensors["imu"].read()
            except Exception as e:
                log.warning(f"IMU read error: {e}")

        # ToF
        if "tof" in self.sensors:
            try:
                packet["tof"] = self.sensors["tof"].read()
            except Exception as e:
                log.warning(f"ToF read error: {e}")

        # GPIO events
        if self.gpio:
            packet["gpio"] = self.gpio.read_events()

        # UART
        for us in self.uart_sensors:
            line = us.read_line()
            if line:
                packet["uart"][us.name] = line

        return packet

    def run(self):
        import msgpack as _mp
        log.info(f"Sensor hub running @ {POLL_RATE} Hz")
        seq = 0
        while _running:
            t0 = time.monotonic()

            packet = self._collect()
            packet["seq"] = seq
            seq += 1

            # Publish through broker with /sensors topic (matching LifecycleNode format)
            topic_b   = b"/sensors"
            payload_b = _mp.packb({"__type__": "sensor/bundle/v1", **packet}, use_bin_type=True)
            self._sock.send_multipart([topic_b, payload_b])

            elapsed = time.monotonic() - t0
            sleep_t = POLL_DT - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                log.debug(f"Sensor loop overrun by {-sleep_t*1000:.1f} ms")

        # Cleanup
        if self.gpio:
            self.gpio.cleanup()
        self._sock.close()
        log.info("Sensor hub stopped.")


if __name__ == "__main__":
    hub = SensorHub()
    hub.run()
