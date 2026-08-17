#!/usr/bin/env python3
"""
TEST 05 — Pi side of UART loopback test
Run this on Raspberry Pi while ESP32 runs tests/05_uart_loopback/main.cpp

Usage:
    python3 uart_test.py /dev/ttyS0 115200
    # or
    python3 uart_test.py /dev/serial0 115200

On Pi: GPIO 14 (pin 8) = TX → ESP32 GPIO 16 (RX)
       GPIO 15 (pin 10) = RX ← ESP32 GPIO 17 (TX)
       GND (pin 6) → ESP32 GND
"""
import serial
import sys
import time

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/serial0"
BAUD = int(sys.argv[2]) if len(sys.argv) > 2 else 115200

print(f"=== MAGI UART Loopback Test — Pi side ===")
print(f"Port: {PORT}  Baud: {BAUD}")
print("Waiting for PING from ESP32... (Ctrl+C to stop)\n")

ping_count = 0
pong_sent = 0

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(0.5)

    while True:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if not line:
            continue

        if line.startswith("PING"):
            ping_count += 1
            pong_sent += 1
            response = f"PONG {ping_count}\n"
            ser.write(response.encode())
            print(f"[RX] {line:<20} → [TX] PONG {ping_count}")
        else:
            print(f"[RX] {line}")

except serial.SerialException as e:
    print(f"ERROR: {e}")
    print("\nTips:")
    print("  - Enable UART: sudo raspi-config → Interfaces → Serial → disable login, enable hardware")
    print("  - Reboot after enabling")
    print("  - Check: ls -la /dev/serial*")
except KeyboardInterrupt:
    print(f"\nDone. Received {ping_count} PINGs, sent {pong_sent} PONGs.")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
