# MAGI Firmware — ESP32 Hardware Implementation

**MAGI** (Multispectral Autonomous Ground Intelligence) — ESP32 quadruped firmware.

## Hardware Stack

| Component | Specification |
|---|---|
| **MCU** | ESP32 (WROOM-32 or DevKit V1) |
| **Servos** | 12× MG996R (3 per leg × 4 legs) |
| **PWM driver** | PCA9685 16-channel, I²C @ 0x40, 400 kHz |
| **Power — Servo domain** | XL4016 DC-DC buck → 6.0 V / 20 A |
| **Power — Logic domain** | LiPo → 5V buck → ESP32 VIN |
| **Battery** | 3S LiPo 5200 mAh 35C (XT60) |
| **Comms to Pi** | UART (115200 baud) on GPIO16 (RX) / GPIO17 (TX) |
| **IMU** | MPU-6050 on I²C @ 0x68 (shared bus) |

## Folder Structure

```
firmware/
├── README.md                    ← This file
│
├── magi_esp32/                  ← MAIN firmware (PlatformIO project)
│   ├── platformio.ini           PlatformIO build config
│   ├── src/
│   │   ├── main.cpp             Entry point — setup() + loop()
│   │   ├── servo_controller.h/cpp   PCA9685 servo abstraction
│   │   ├── kinematics.h/cpp     3-DOF IK solver (coxa/femur/tibia)
│   │   ├── gait_engine.h/cpp    Crawl / trot / stand gait patterns
│   │   ├── imu_driver.h/cpp     MPU-6050 read + complementary filter
│   │   ├── uart_protocol.h/cpp  Pi ↔ ESP32 command protocol
│   │   └── config.h             All tunable constants
│   └── lib/                     (empty — libs via PlatformIO registry)
│
└── tests/                       ← Hardware testing sketches (standalone)
    ├── 01_pca9685_scan/         I²C device scan + PCA9685 init test
    ├── 02_servo_sweep/          Single servo full range sweep
    ├── 03_all_servos/           All 12 servos neutral → wave test
    ├── 04_imu_read/             MPU-6050 raw accel/gyro output
    ├── 05_uart_loopback/        ESP32 ↔ Pi UART echo test
    ├── 06_ik_solver/            IK solver unit test (no hardware)
    └── 07_gait_crawl/           Full crawl gait on real hardware
```

## Quick Start

### 1 — Install PlatformIO

```bash
pip install platformio
# or install the VS Code PlatformIO extension
```

### 2 — Build and upload main firmware

```bash
cd firmware/magi_esp32
pio run --target upload --upload-port /dev/ttyUSB0
pio device monitor --port /dev/ttyUSB0 --baud 115200
```

### 3 — Run a hardware test

```bash
cd firmware/tests/02_servo_sweep
pio run --target upload --upload-port /dev/ttyUSB0
```

## Wiring Summary

```
ESP32              PCA9685
  GPIO21 (SDA) ─── SDA
  GPIO22 (SCL) ─── SCL
  3.3V         ─── VCC
  GND          ─── GND (star ground)

PCA9685            Servo domain
  V+  ──────────── Buck converter output (6.0 V)
  GND ──────────── Star ground

PCA9685 Channel Map (leg-major order):
  Ch  0 = Leg1 FR Coxa    Ch  3 = Leg2 FL Coxa
  Ch  1 = Leg1 FR Femur   Ch  4 = Leg2 FL Femur
  Ch  2 = Leg1 FR Tibia   Ch  5 = Leg2 FL Tibia
  Ch  6 = Leg3 BR Coxa    Ch  9 = Leg4 BL Coxa
  Ch  7 = Leg3 BR Femur   Ch 10 = Leg4 BL Femur
  Ch  8 = Leg3 BR Tibia   Ch 11 = Leg4 BL Tibia

ESP32              MPU-6050
  GPIO21 (SDA) ─── SDA  (shared I²C bus)
  GPIO22 (SCL) ─── SCL
  3.3V         ─── VCC
  GND          ─── GND

ESP32              Raspberry Pi 4B
  GPIO16 (RX2) ─── TX (GPIO 14 / pin 8)
  GPIO17 (TX2) ─── RX (GPIO 15 / pin 10)
  GND          ─── GND (shared star ground)
```

## UART Command Protocol (Pi → ESP32)

All commands are newline-terminated ASCII for easy debugging.

| Command | Format | Example |
|---|---|---|
| Move all joints | `MOVE a0,a1,...,a11\n` | `MOVE 90,45,90,90,45,90,...\n` |
| Set gait mode | `GAIT crawl\|trot\|stand\n` | `GAIT trot\n` |
| Set speed | `SPEED 0-100\n` | `SPEED 60\n` |
| Emergency stop | `STOP\n` | `STOP\n` |
| Neutral pose | `HOME\n` | `HOME\n` |
| Request IMU | `IMU\n` | responds with `IMU ax,ay,az,gx,gy,gz\n` |
| Request status | `STATUS\n` | responds with `OK gen=N\n` |

## MG996R PWM Mapping

| Angle | Pulse width | PCA9685 OFF_TIME |
|---|---|---|
| 0°   | 500 µs  | 102 |
| 90°  | 1500 µs | 307 |
| 180° | 2500 µs | 512 |

Formula: `OFF_TIME = pulse_us × 50 × 4096 / 1_000_000`

## Servo Channel → Leg Mapping

Leg order as viewed from above (top-down):
```
    [FR=L1] [FL=L2]
       |||     |||
    [=BODY======]
       |||     |||
    [BR=L3] [BL=L4]
```

Each leg: Coxa (horizontal yaw), Femur (vertical swing), Tibia (knee)
