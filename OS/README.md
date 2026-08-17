# MAGI OS — Raspberry Pi 4B Autonomous Intelligence Platform

> **M**ulti-purpose **A**utonomous **G**eneral **I**ntelligence Operating System  
> A lightweight, production-grade embedded OS for simultaneous AI inference + sensor fusion on Raspberry Pi 4B.

---

## Table of Contents
1. [What Is MAGI OS](#1-what-is-magi-os)
2. [Why This Architecture](#2-why-this-architecture)
3. [Hardware Requirements](#3-hardware-requirements)
4. [System Architecture](#4-system-architecture)
5. [File Structure](#5-file-structure)
6. [Core Components — What, Why, How](#6-core-components)
7. [Sensor Layer](#7-sensor-layer)
8. [AI Inference Nodes](#8-ai-inference-nodes)
9. [ROS2-Grade Middleware](#9-ros2-grade-middleware)
10. [Docker Testing Environment](#10-docker-testing-environment)
11. [Quick Setup — Pi Deployment](#11-quick-setup)
12. [MAGI CLI Reference](#12-magi-cli-reference)
13. [RAM Budget](#13-ram-budget)
14. [Thermal Guidelines](#14-thermal-guidelines)
15. [Adding Sensors](#15-adding-sensors)
16. [Adding / Replacing Models](#16-adding--replacing-models)
17. [Troubleshooting](#17-troubleshooting)

---

## 1. What Is MAGI OS

MAGI OS is a **purpose-built, minimal Linux-based operating system** designed to run on a Raspberry Pi 4B (4 GB RAM). It was built to solve one specific problem:

> *Run 3 AI inference models simultaneously while talking to all connected sensors — with maximum reliability, minimum RAM, and industry-grade software structure.*

It is **not** a general-purpose OS. Every decision — from the init system to the IPC mechanism to the message format — was made to serve this single mission.

**Named after the MAGI supercomputer** from Neon Genesis Evangelion — three specialized systems (Celebi, Gengar, Lugia) each making independent decisions that are fused into one final output.

| MAGI Node | Role | CPU Core |
|---|---|---|
| **Celebi** (MAGI-1) | Object / Target Detection | Core 1 |
| **Gengar** (MAGI-2) | Scene Analysis + Motion | Core 2 |
| **Lugia** (MAGI-3) | Sensor Fusion + Decisions | Core 3 |
| **Sensor Hub** | All hardware I/O | Core 0 |

---

## 2. Why This Architecture

### Why not full Ubuntu + ROS2?
| Concern | ROS2 Full Stack | MAGI OS |
|---|---|---|
| RAM at boot | ~600–800 MB | ~180 MB |
| Boot time | ~30–60 s | ~8 s |
| Complexity | Very high | Moderate |
| Inference support | None native | TFLite + XNNPACK |
| Customization | Limited | Complete |

### Why stripped RPi OS Lite as base?
- Kernel already has all RPi 4B drivers (VideoCore VI, I2C, SPI, UART, GPIO)
- Removes 300+ MB of desktop/bluetooth/audio packages not needed
- Retains `apt` for easy sensor driver installation
- 64-bit Bookworm = full ARM64 SIMD support for TFLite XNNPACK

### Why ZeroMQ instead of DDS?
- ZeroMQ uses **8 MB RAM** for our XPUB/XSUB broker vs **80–150 MB** for Fast-DDS
- No XML config, no daemon discovery protocol
- Microsecond latency on localhost IPC sockets
- POSIX shared memory for camera frames = **zero-copy** frame passing

### Why separate processes (not threads)?
- One model crash cannot kill the others
- Each process pinned to dedicated CPU core = no OS scheduling interference
- Independent restart via watchdog without full system restart
- Memory isolation = no GIL contention between models

### Why bake ROS2 features natively instead of installing ROS2?
MAGI OS extracts the **concepts** from ROS2 (typed messages, lifecycle nodes, QoS, diagnostics, parameter server, transforms, recorder) and implements them in pure Python/ZeroMQ — gaining all the software-engineering benefits at a fraction of the cost:
- **+22 MB RAM** overhead for all middleware vs **+200 MB** for ROS2 Humble

---

## 3. Hardware Requirements

| Component | Specification |
|---|---|
| **Board** | Raspberry Pi 4B |
| **RAM** | 4 GB LPDDR4 |
| **Storage** | ≥32 GB microSD, A2 rated (or USB 3.0 SSD for reliability) |
| **Power** | 5V / 3A USB-C (official Pi PSU recommended) |
| **Cooling** | **Heatsink + 5V fan — MANDATORY for continuous inference** |
| **OS Base** | Raspberry Pi OS Lite, Bookworm, 64-bit |

**Optional hardware (configured in `config.yaml`):**

| Interface | Example Devices |
|---|---|
| I2C | MPU-6050 IMU, BMP280 barometer, VL53L0X ToF |
| SPI | MCP3208 ADC, high-speed displays |
| UART | GPS (NMEA), telemetry MCU, external microcontroller |
| GPIO | Trigger inputs, relay outputs, status LEDs |
| USB | USB webcam, Pi Camera (via V4L2) |

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      MAGI OS v2                                 │
├─────────────────────────────────────────────────────────────────┤
│  APPLICATION LAYER (one process per CPU core)                   │
│  Core 1: MAGI-1 Celebi    → Object Detection (TFLite)         │
│  Core 2: MAGI-2 Gengar   → Scene Analysis   (TFLite)         │
│  Core 3: MAGI-3 Lugia      → Fusion + Decision Engine          │
│  Core 0: Sensor Hub         → All hardware I/O                  │
│  Core 0: Camera Capture     → Camera → shared memory            │
├─────────────────────────────────────────────────────────────────┤
│  MAGI MESSAGE BUS (ZMQ XPUB/XSUB proxy — port 5555/5556)        │
│  Topics: /sensors  /camera  /detections  /scene  /decision      │
│          /diagnostics  /parameters  /tf                         │
├─────────────────────────────────────────────────────────────────┤
│  MIDDLEWARE SERVICES (all on Core 0)                            │
│  Parameter Server (port 5558) — live runtime config             │
│  Diagnostics Monitor  — node health aggregator                  │
│  TF Transform Store   — sensor coordinate frames                │
│  SQLite Recorder      — optional message recording              │
│  Watchdog             — process supervisor + auto-restart        │
├─────────────────────────────────────────────────────────────────┤
│  HARDWARE ABSTRACTION                                           │
│  pigpio · smbus2 · spidev · pyserial · V4L2 · RPi.GPIO          │
├─────────────────────────────────────────────────────────────────┤
│  KERNEL: Linux 6.x · Raspberry Pi OS Lite Bookworm 64-bit       │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Sensors (I2C/SPI/UART/GPIO)
        │
        ▼
  sensor_hub.py ──────────────────────────────→ /sensors topic
        │
camera_capture.py → POSIX shared memory
        │                   │
        │                   ▼
        │         celebi.py (Core 1) → /detections topic ──┐
        │                                                     │
        └──────→ gengar.py (Core 2) → /scene topic ────────┤
                                                              │
                                              lugia.py (Core 3) → /decision topic
                                                              │
                                              GPIO alert · UART · log file
```

---

## 5. File Structure

```
MAGI/OS/
│
├── README.md                        ← This file
├── .dockerignore
│
├── setup/                           ← Run these IN ORDER on RPi
│   ├── 01_strip_os.sh               Strip bloat, mask services
│   ├── 02_install_deps.sh           Install Python venv + libs
│   ├── 03_configure_boot.sh         CPU isolation, freq, interfaces
│   └── 04_install_services.sh       Deploy to /opt/magi + systemd
│
├── src/
│   │
│   ├── core/                        ← ROS2-grade middleware (native)
│   │   ├── message_bus.py           XPUB/XSUB broker (replaces roscore)
│   │   ├── messages.py              Typed message dataclasses
│   │   ├── lifecycle.py             Node lifecycle state machine
│   │   ├── qos.py                   Per-topic QoS profiles
│   │   ├── param_server.py          Runtime parameter store
│   │   ├── diagnostics.py           Health monitor
│   │   ├── transforms.py            TF2 sensor frame registry
│   │   └── recorder.py              SQLite message recorder
│   │
│   ├── cli/
│   │   └── magi_cli.py              magi-cli command line tool
│   │
│   ├── common/                      ← Shared utilities
│   │   ├── config.yaml              Master system configuration
│   │   ├── logger.py                Rotating file + console logger
│   │   ├── ipc.py                   ZMQ + POSIX shared memory helpers
│   │   └── base_node.py             Legacy base (superseded by lifecycle.py)
│   │
│   ├── sensors/                     ← Hardware I/O layer (Core 0)
│   │   ├── sensor_hub.py            I2C/SPI/UART/GPIO aggregator
│   │   └── camera_capture.py        Camera → shared memory writer
│   │
│   ├── magi1/
│   │   └── celebi.py              Detection node (Core 1)
│   │
│   ├── magi2/
│   │   └── gengar.py             Scene analysis node (Core 2)
│   │
│   ├── magi3/
│   │   └── lugia.py                Fusion + decision node (Core 3)
│   │
│   ├── watchdog.py                  Process supervisor
│   └── status_monitor.py            Terminal dashboard
│
├── services/
│   └── magi-watchdog.service        Systemd unit (single entry point)
│
└── docker/                          ← Local testing environment
    ├── Dockerfile
    ├── docker-compose.yml
    ├── entrypoint.sh
    ├── requirements-docker.txt
    ├── test_config.yaml
    ├── start_magi.bat               Windows launcher
    ├── DOCKER_TESTING.md
    ├── mock_hardware/               Hardware stubs for Docker
    │   ├── pigpio.py
    │   ├── smbus2.py
    │   ├── spidev.py
    │   ├── posix_ipc.py
    │   ├── cpu_affinity.py
    │   ├── RPi/GPIO.py
    │   └── tflite_runtime/interpreter.py
    └── dummy_models/
        ├── celebi.tflite
        ├── gengar.tflite
        └── lugia.tflite
```

---

## 6. Core Components

### `core/message_bus.py` — The Broker
**What:** ZeroMQ XPUB/XSUB proxy that routes messages between all nodes.  
**Why:** Equivalent to `roscore` but uses 8 MB RAM vs 150 MB for DDS. Enables any node to publish/subscribe to any topic dynamically.  
**How:** Publishers connect to port 5555, subscribers to port 5556. The broker forwards all traffic and maintains a last-value cache for TRANSIENT_LOCAL topics.  
**Where:** Started first by watchdog. Listens on TCP ports 5555 (frontend) and 5556 (backend).

### `core/messages.py` — Typed Messages
**What:** Python dataclass definitions for every message type in the system.  
**Why:** Equivalent to ROS2 `.msg` files. Enforces structure, enables serialization, makes the system self-documenting.  
**How:** Serialized with `msgpack` (10× faster than JSON, typed, binary). Each message carries a `Header` with timestamp, sequence number, frame ID, and node ID.  
**Where:** Imported by every node that publishes or subscribes.

### `core/lifecycle.py` — Lifecycle State Machine
**What:** Base class implementing the ROS2 Managed Node lifecycle.  
**Why:** Prevents nodes from crashing during startup. Resources allocated in `on_configure()`, processing starts in `on_activate()`. Watchdog can call `deactivate()` cleanly before restart.  
**How:** States: `UNCONFIGURED → INACTIVE → ACTIVE → FINALIZED`. Call `node.boot()` to run full cycle automatically.  
**Where:** All three MAGI nodes inherit from `LifecycleNode`.

### `core/qos.py` — Quality of Service
**What:** Per-topic delivery guarantee profiles.  
**Why:** Camera frames should be dropped if stale (BEST_EFFORT). Decisions must never be lost (RELIABLE). New nodes that subscribe late must get the last decision (TRANSIENT_LOCAL).  
**How:** Profiles defined as frozen dataclasses, registered in `TOPIC_QOS` dict. Broker reads QoS to decide whether to cache messages.

### `core/param_server.py` — Parameter Server
**What:** Runtime key-value configuration store.  
**Why:** Change `confidence_threshold` on MAGI-1 without restarting the system. Critical for field tuning.  
**How:** ZMQ REQ/REP on port 5558. Seeded from `config.yaml` on startup. All nodes use `ParamClient` to get/set values. Changes published to `/parameters` topic.  
**Where:** `magi-cli param set magi1 confidence_threshold 0.6`

### `core/diagnostics.py` — Health Monitor
**What:** Aggregates health pings from all nodes.  
**Why:** Know instantly if a node is STALE (crashed), ERROR (inference failing), or OK. Equivalent to `ros_diagnostics`.  
**How:** Every node publishes to `/diagnostics` each second with status + key metrics. Monitor detects nodes that haven't reported in 5 seconds.  
**Where:** `magi-cli diag`

### `core/transforms.py` — TF Frame Registry
**What:** Coordinate transform registry for sensor frames.  
**Why:** IMU is mounted 5cm above robot center. Camera is 10cm forward and tilted 15°. ToF is at the front. For accurate sensor fusion, MAGI-3 needs to know where each sensor is in 3D space.  
**How:** Static transforms defined in `build_default_tf()`. Uses quaternion math for 3D rotations. Gengar uses the IMU frame transform when processing motion data.  
**Where:** Edit `src/core/transforms.py` to match your physical hardware layout.

### `core/recorder.py` — Message Recorder
**What:** SQLite-based topic recorder and player.  
**Why:** Record a real sensor session on the Pi, replay it on your laptop for debugging without hardware. Equivalent to `rosbag`.  
**How:** Subscribes to specified topics, batches messages to SQLite every 500ms. Player reads the DB and re-publishes at variable speed.  
**Where:** `magi-cli record --topics /sensors /detections --output run.db`

### `cli/magi_cli.py` — Command Line Interface
**What:** Full CLI tool for system inspection and control.  
**Why:** Replaces `ros2 topic echo`, `ros2 param set`, `ros2 node info`.  
**How:** Connects to the message bus and parameter server via ZMQ.  
**Where:** `python3 /opt/magi/src/cli/magi_cli.py` or symlinked as `magi-cli`.

---

## 7. Sensor Layer

### `sensors/sensor_hub.py`
**What:** Reads all connected hardware sensors and publishes a `SensorMsg` to `/sensors` at 50 Hz.  
**Why:** Centralizes all I/O on Core 0, keeping cores 1–3 free for inference.  
**How:** Initializes each enabled sensor from `config.yaml`. Reads I2C (IMU, ToF), SPI (ADC), UART (GPS), GPIO (interrupts) in a timed loop.

**Supported devices:**

| Sensor | Interface | Address | Config key |
|---|---|---|---|
| MPU-6050/9250 IMU | I2C | 0x68 | `sensors.i2c[name=imu]` |
| BMP280 Barometer | I2C | 0x76 | `sensors.i2c[name=barometer]` |
| VL53L0X ToF | I2C | 0x29 | `sensors.i2c[name=tof_front]` |
| MCP3208 ADC | SPI | CS0 | `sensors.spi[name=adc]` |
| GPS (NMEA) | UART | /dev/ttyAMA0 | `sensors.uart[name=gps]` |
| Trigger Input | GPIO | BCM 17 | `sensors.gpio[name=trigger_in]` |
| Alert Output | GPIO | BCM 22 | `sensors.gpio[name=alert_out]` |

### `sensors/camera_capture.py`
**What:** Captures video frames and writes to POSIX shared memory.  
**Why:** Shared memory = zero-copy. MAGI-1 and MAGI-2 read the same frame without any data being copied or serialized.  
**How:** OpenCV reads from `/dev/video0`, writes raw BGR bytes to named shared memory block. Set `camera.enabled: true` in config.

---

## 8. AI Inference Nodes

### MAGI-1 Celebi — `magi1/celebi.py`
**What:** Object detection using YOLOv8-nano or MobileNet-SSD TFLite.  
**Why:** Detects objects/targets in the camera frame at ~10 Hz.  
**How:** Reads camera frame from shared memory → resize → normalize → TFLite inference → parse boxes/classes/scores → publish `DetectionMsg` to `/detections`.  
**Core:** 1 | **Input:** 320×320 RGB | **Output:** list of `Detection(label, confidence, bbox)`

### MAGI-2 Gengar — `magi2/gengar.py`
**What:** Scene classification + IMU motion analysis using EfficientNet-Lite TFLite.  
**Why:** Understands the context of the environment (restricted zone, emergency, crowded, etc.) and whether the platform is moving.  
**How:** Camera frame → ImageNet normalization → TFLite inference → softmax scene class. IMU accel data → motion magnitude + jerk calculation. Anomaly score = uncertainty × motion.  
**Core:** 2 | **Input:** 224×224 RGB | **Output:** `SceneMsg(scene, anomaly_score, motion)`

### MAGI-3 Lugia — `magi3/lugia.py`
**What:** Sensor fusion and decision engine.  
**Why:** Combines detections + scene + raw sensor data into one prioritized action decision.  
**How:** Subscribes to `/detections`, `/scene`, `/sensors`. Rule-based engine evaluates 6 priority rules. Publishes `DecisionMsg` to `/decision` with TRANSIENT_LOCAL QoS (new subscribers always get the last decision).  
**Core:** 3 | **Output:** `action` (IDLE / TRACK / ALERT / ANALYZE / EMERGENCY)

**Decision Rules (priority order):**

| Priority | Condition | Action |
|---|---|---|
| 10 | ToF < 200 mm | EMERGENCY |
| 9 | GPIO trigger_in fired | EMERGENCY |
| 8 | Person in restricted zone | EMERGENCY |
| 6 | Anomaly score > 0.7 | ALERT |
| 5 | Scene = emergency/obstacle_close | ANALYZE |
| 4 | Detections > 0 | TRACK |
| 0 | Nothing | IDLE |

---

## 9. ROS2-Grade Middleware

MAGI OS implements **8 ROS2 features natively** — no ROS2 installed:

| ROS2 Feature | MAGI Implementation | RAM Cost |
|---|---|---|
| roscore / DDS | `core/message_bus.py` (ZMQ XPUB/XSUB) | 8 MB |
| `.msg` typed messages | `core/messages.py` (dataclasses + msgpack) | 0 MB |
| Managed Node lifecycle | `core/lifecycle.py` | 0 MB |
| QoS profiles | `core/qos.py` | 0 MB |
| Parameter server | `core/param_server.py` | 4 MB |
| ros_diagnostics | `core/diagnostics.py` | 3 MB |
| TF2 transforms | `core/transforms.py` | 2 MB |
| rosbag | `core/recorder.py` (SQLite) | 5 MB |
| ros2 CLI | `cli/magi_cli.py` | 0 MB |
| **Total overhead** | | **+22 MB** |

---

## 10. Docker Testing Environment

Test the full MAGI OS on your Windows/Mac/Linux machine without any hardware.

**How it works:** Every hardware library (`pigpio`, `smbus2`, `tflite_runtime`, `posix_ipc`) is intercepted by mock stubs in `docker/mock_hardware/`. The actual IPC, lifecycle, decision logic, and message routing run 100% real.

**Mock behaviors:**
- `smbus2` → MPU-6050 returns sine-wave IMU data (realistic motion simulation)
- `pigpio` → Random GPIO interrupt fires every 15–30 seconds
- `tflite_runtime` → Returns 3 realistic fake detections (person, cat, car)
- `posix_ipc` → In-process bytearray (cross-platform shared memory)

### Start on Windows
```
double-click: docker\start_magi.bat
```
Then choose **[1] Start ALL services**.

### Manual commands
```bash
# Build image (first time, ~3–5 min)
docker compose -f docker/docker-compose.yml build

# Start all services
docker compose -f docker/docker-compose.yml up -d

# Watch MAGI-3 decisions
docker compose -f docker/docker-compose.yml logs -f magi3

# Stop
docker compose -f docker/docker-compose.yml down
```

---

## 11. Quick Setup

Flash **Raspberry Pi OS Lite, Bookworm, 64-bit** to microSD. Boot and SSH in, then:

```bash
# Clone or copy MAGI OS to the Pi
scp -r MAGI/OS pi@192.168.1.100:~/magi-os
ssh pi@192.168.1.100

# Run setup scripts IN ORDER:
sudo bash ~/magi-os/setup/01_strip_os.sh
sudo reboot

sudo bash ~/magi-os/setup/02_install_deps.sh
sudo bash ~/magi-os/setup/03_configure_boot.sh
sudo reboot

# Place your TFLite model files
sudo cp celebi.tflite  /opt/magi/models/
sudo cp gengar.tflite /opt/magi/models/
sudo cp lugia.tflite    /opt/magi/models/

# Edit config: enable your sensors and camera
sudo nano /opt/magi/config/config.yaml

# Deploy services
sudo bash ~/magi-os/setup/04_install_services.sh

# Start MAGI OS
sudo systemctl start magi-watchdog
sudo systemctl status magi-watchdog
```

**Service management:**
```bash
sudo systemctl start   magi-watchdog
sudo systemctl stop    magi-watchdog
sudo systemctl restart magi-watchdog
sudo journalctl -fu    magi-watchdog
```

---

## 12. MAGI CLI Reference

```bash
# ── Topics ─────────────────────────────────────────────────────────
magi-cli topic list                      # List all active topics
magi-cli topic echo /detections          # Print messages live
magi-cli topic echo /decision --count 5  # Print 5 then stop
magi-cli topic hz   /sensors             # Measure publish rate

# ── Parameters ─────────────────────────────────────────────────────
magi-cli param list magi1                # List all magi1 params
magi-cli param get  magi1 confidence_threshold
magi-cli param set  magi1 confidence_threshold 0.6
magi-cli param set  system poll_rate_hz 100

# ── Diagnostics ────────────────────────────────────────────────────
magi-cli diag                            # Show all node health

# ── Recording ──────────────────────────────────────────────────────
magi-cli record --topics /sensors /detections /decision --output run1.db
magi-cli play   --file run1.db --speed 2.0
magi-cli info   --file run1.db

# ── TF Frames ──────────────────────────────────────────────────────
magi-cli tf tree                         # Show sensor frame tree
magi-cli tf frames                       # List all frame names
```

---

## 13. RAM Budget

| Component | RAM Usage |
|---|---|
| Stripped Linux kernel + initramfs | ~40 MB |
| System daemons (udev, sshd, journald) | ~60 MB |
| Python runtime + sensor middleware | ~80 MB |
| **OS Baseline** | **~180 MB** |
| Message bus | ~8 MB |
| Parameter server | ~4 MB |
| Diagnostics monitor | ~3 MB |
| TF store + recorder | ~7 MB |
| **Middleware Total** | **~22 MB** |
| MAGI-1 Celebi (YOLOv8-nano TFLite) | ~300 MB |
| MAGI-2 Gengar (EfficientNet-Lite) | ~200 MB |
| MAGI-3 Lugia (LSTM / rule engine) | ~150 MB |
| Inference buffers + preprocessing | ~400 MB |
| **Models Total** | **~1050 MB** |
| **TOTAL USED** | **~1252 MB** |
| **FREE HEADROOM** | **~2844 MB ✅** |

---

## 14. Thermal Guidelines

| Load | Expected Temperature | Action |
|---|---|---|
| Idle (OS only) | 45–50°C | Normal |
| 3 models running | 65–72°C | OK with fan |
| 3 models + camera | 68–75°C | OK with fan |
| > 80°C | CPU throttles to 600 MHz | Reduce `arm_freq` |

**Required cooling:** Heatsink on SoC + 5V PWM fan. Without this, the Pi will thermal throttle within 2–3 minutes of full inference load.

---

## 15. Adding Sensors

1. **Edit `config.yaml`** — enable the sensor and set its address/port
2. **Add a driver class** in `sensor_hub.py` (follow the `IMU_MPU6050` pattern)
3. **Instantiate it** in `SensorHub._init_sensors()` based on the config `name` field
4. **Add its data** to the `SensorMsg` dataclass in `core/messages.py` if needed

**Example — adding a new I2C sensor:**
```yaml
# config.yaml
sensors:
  i2c:
    devices:
      - name: "my_sensor"
        address: 0x48
        enabled: true
```
```python
# sensor_hub.py — inside _init_sensors()
if name == "my_sensor":
    self.sensors["my_sensor"] = MySensorDriver(bus, addr)
```

---

## 16. Adding / Replacing Models

Replace placeholder `.tflite` files in `/opt/magi/models/`:

| File | Replace With |
|---|---|
| `celebi.tflite` | YOLOv8-nano / MobileNet-SSD TFLite detection model |
| `gengar.tflite` | EfficientNet-Lite / custom scene classifier |
| `lugia.tflite` | LSTM sequence model (optional — rule engine works without it) |

**Convert from PyTorch:**
```bash
pip install onnx onnx-tf
# Export PyTorch → ONNX → TFLite via TF converter
```

**Convert from Keras/TF:**
```python
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model("my_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open("celebi.tflite", "wb").write(tflite_model)
```

---

## 17. Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `pigpio daemon not running` | pigpiod not started | `sudo systemctl start pigpiod` |
| `Model load failed` | Wrong model path or corrupt file | Check `/opt/magi/models/`, replace placeholder |
| `zmq.error.ZMQError: Address in use` | Previous session didn't clean up | `rm /tmp/magi/*.sock` then restart |
| Decision always IDLE | MAGI-1/2 not publishing | `magi-cli topic hz /detections` |
| High RAM usage | Model too large | Use quantized INT8 TFLite models |
| CPU thermal throttle | Insufficient cooling | Add heatsink + fan, reduce `arm_freq` to 1500 |
| `STALE` in diagnostics | Node crashed | `magi-cli diag`, check `journalctl -fu magi-watchdog` |
| Docker: IPC error | Bus not ready | Increase `sleep` in `entrypoint.sh` |

---

## License

MAGI OS — Custom embedded platform built for autonomous AI inference on Raspberry Pi 4B.
