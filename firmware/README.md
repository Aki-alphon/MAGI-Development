# MAGI Hardware & Firmware Interface

This directory contains low-level microcontroller code and test scripts for the hardware components interfacing with the MAGI OS.

## Components
- **`magi_esp32/`**: Firmware for the ESP32 coprocessor handling real-time locomotion, kinematics, and direct PWM generation.
- **`pca9685_test/`**: Diagnostic scripts for testing the I2C PWM servo drivers (PCA9685) controlling the robotic legs.

These modules communicate with the main Raspberry Pi OS via serial or SPI.
