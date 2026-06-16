"""
MAGI OS — Mock smbus2
docker/mock_hardware/smbus2.py

Replaces smbus2 for I2C simulation. Returns realistic sensor values:
 - MPU-6050 (0x68): sinusoidal accel + gyro data
 - BMP280   (0x76): stable pressure/temp
 - VL53L0X  (0x29): simulated via VL53L0X mock
"""

import time
import math
import random
import struct


class SMBus:
    """Mock I2C bus — returns realistic sensor readings."""

    def __init__(self, bus: int = 1, *args, **kwargs):
        self.bus = bus
        self._t0 = time.time()
        print(f"[mock_smbus2] SMBus({bus}) stub initialized")

    def _elapsed(self) -> float:
        return time.time() - self._t0

    def write_byte_data(self, addr: int, reg: int, val: int):
        pass  # Ignore writes (power-on, config, etc.)

    def read_byte_data(self, addr: int, reg: int) -> int:
        return 0

    def read_i2c_block_data(self, addr: int, reg: int, length: int) -> list:
        """Return simulated register block based on device address."""
        t = self._elapsed()

        # ── MPU-6050 / MPU-9250 (0x68) — Accel + Gyro ──────────────────
        if addr == 0x68 and reg == 0x3B and length == 14:
            # Accel: gentle sine wave (simulates motion)
            ax = int(math.sin(t * 0.5) * 800)           # ~±0.05 g
            ay = int(math.cos(t * 0.3) * 600)
            az = int(16384 + math.sin(t * 0.1) * 200)   # ~1g + noise
            # Gyro: small random drift
            gx = int(math.sin(t * 1.2) * 130)
            gy = int(math.cos(t * 0.8) * 100)
            gz = int(math.sin(t * 0.4) * 80)
            temp_raw = 20000  # raw temp register (unused in hub)

            def to_bytes(v):
                v = int(v) & 0xFFFF
                return [(v >> 8) & 0xFF, v & 0xFF]

            return (to_bytes(ax) + to_bytes(ay) + to_bytes(az) +
                    to_bytes(temp_raw) +
                    to_bytes(gx) + to_bytes(gy) + to_bytes(gz))

        # ── BMP280 (0x76) — Pressure/Temp ───────────────────────────────
        if addr == 0x76:
            return [0x00] * length

        # Default: zeros
        return [0x00] * length

    def close(self):
        pass
