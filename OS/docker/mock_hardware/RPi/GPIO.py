"""
MAGI OS — Mock RPi.GPIO
docker/mock_hardware/RPi/GPIO.py
"""

BCM    = 11
BOARD  = 10
IN     = 0
OUT    = 1
PUD_UP   = 22
PUD_DOWN = 21
PUD_OFF  = 20
RISING   = 31
FALLING  = 32
BOTH     = 33
HIGH = 1
LOW  = 0

_pins = {}


def setmode(mode): pass
def setup(pin, mode, pull_up_down=PUD_OFF, initial=0):
    _pins[pin] = initial

def output(pin, val):
    _pins[pin] = val

def input(pin) -> int:
    return _pins.get(pin, 0)

def add_event_detect(pin, edge, callback=None, bouncetime=0): pass
def remove_event_detect(pin): pass
def cleanup(): pass
def setwarnings(flag): pass
