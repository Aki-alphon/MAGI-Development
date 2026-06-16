"""
MAGI OS — Mock spidev
docker/mock_hardware/spidev.py
"""


class SpiDev:
    def __init__(self):
        self.max_speed_hz = 1000000
        self.mode = 0
        print("[mock_spidev] SpiDev stub initialized")

    def open(self, bus: int, device: int):
        pass

    def xfer2(self, data: list) -> list:
        return [0x00] * len(data)

    def close(self):
        pass
