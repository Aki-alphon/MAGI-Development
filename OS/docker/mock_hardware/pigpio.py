"""
MAGI OS — Mock pigpio
docker/mock_hardware/pigpio.py

Replaces the real pigpio library in Docker/dev environments.
Simulates all GPIO reads/writes with realistic fake data.
"""

import time
import threading
import random

# Constants matching real pigpio API
INPUT  = 0
OUTPUT = 1
PUD_UP   = 2
PUD_DOWN = 1
PUD_OFF  = 0
RISING_EDGE  = 1
FALLING_EDGE = 2
EITHER_EDGE  = 3


class _CallbackRef:
    def __init__(self, pin, edge, func):
        self._pin  = pin
        self._edge = edge
        self._func = func
        self._active = True
        # Simulate random GPIO events every 15–30 seconds
        self._thread = threading.Thread(target=self._simulate, daemon=True)
        self._thread.start()

    def _simulate(self):
        while self._active:
            delay = random.uniform(15, 30)
            time.sleep(delay)
            if self._active:
                # Simulate level change
                self._func(self._pin, 0, int(time.time() * 1e6))

    def cancel(self):
        self._active = False


class pi:
    """Mock pigpio.pi() — simulates GPIO hardware."""

    def __init__(self, *args, **kwargs):
        self.connected = True
        self._pins = {}
        print("[mock_pigpio] pigpio stub initialized")

    def set_mode(self, pin: int, mode: int):
        self._pins[pin] = {"mode": mode, "val": 0}

    def set_pull_up_down(self, pin: int, pud: int):
        pass

    def read(self, pin: int) -> int:
        return self._pins.get(pin, {}).get("val", 0)

    def write(self, pin: int, val: int):
        if pin in self._pins:
            self._pins[pin]["val"] = val

    def callback(self, pin: int, edge: int, func) -> _CallbackRef:
        return _CallbackRef(pin, edge, func)

    def set_PWM_dutycycle(self, pin: int, duty: int):
        pass

    def hardware_PWM(self, pin: int, freq: int, duty: int):
        pass

    def stop(self):
        self.connected = False
        print("[mock_pigpio] pigpio stub stopped")
