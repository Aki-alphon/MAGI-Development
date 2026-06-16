# MAGI — 12-DOF Quadruped Robot Platform

MAGI (Multispectral Autonomous Ground Intelligence) is a 12-DOF quadruped robot platform. The system is split into three main parts: an embedded OS running on a Raspberry Pi 4 for high-level decision making and AI inference, ESP32 firmware for joint servo control and kinematics, and a browser-based 3D simulation sandbox for gait training and validation.

---

## System Architecture

The high-level software stack is based on three independent modules running concurrently on the Raspberry Pi:

* **Melchior (MAGI-1)**: Object and target detection (using TFLite inference).
* **Balthasar (MAGI-2)**: Scene classification and IMU motion anomaly detection.
* **Caspar (MAGI-3)**: Sensor fusion and priority-based decision engine.

```
                  ┌────────────────────────────────────────┐
                  │              MAGI OS                   │
                  ├────────────────────────────────────────┤
                  │  Core 1: Melchior (Target Detection)   │
                  │  Core 2: Balthasar (Scene Analysis)    │
                  │  Core 3: Caspar (Decision Engine)      │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼ [UART Protocol]
                  ┌────────────────────────────────────────┐
                  │            ESP32 Firmware              │
                  ├────────────────────────────────────────┤
                  │  - 12-DOF Inverse Kinematics Solver    │
                  │  - PCA9685 Servo Controller (12 joints) │
                  │  - Gait Engine (Trot, Crawl, Bound)    │
                  └────────────────────────────────────────┘
```

---

## Repository Layout

* **[OS](file:///home/aki/Downloads/MAGI/OS/)**: Stripped-down Raspberry Pi OS Lite configuration. Includes ZeroMQ and POSIX shared-memory middleware (replacing standard ROS2/DDS to save RAM), process affinity pins, watchdog scripts, and diagnostic tools.
* **[firmware](file:///home/aki/Downloads/MAGI/firmware/)**: ESP32 joint controller code. Handles 3-DOF Inverse Kinematics (IK), complementary filtering for the MPU-6050 IMU, PCA9685 I2C servo drivers, and UART packet parsing for Pi communication.
* **[simulation](file:///home/aki/Downloads/MAGI/simulation/)**: Three.js WebGL visualizer and PyBullet physics scripts. Includes browser-based genetic algorithm training for gait parameters, visual phase diagrams, and coordinate solvers.
* **[model & miscellenious](file:///home/aki/Downloads/MAGI/model%20&miscellenious/)**: Chassis CAD project (`MAGI.3mf`) detailing the physical body, legs, and electronic mounts.
* **[HeatMAP](file:///home/aki/Downloads/MAGI/HeatMAP/)**: Python/Streamlit dashboard for image preprocessing pipelines (downsampling, CLAHE, scaling) and confusion matrix evaluation.

---

## Quick Start

### 3D Web Simulator
Run a local HTTP server in the simulation directory:
```bash
cd simulation
python3 -m http.server 8765
```
Open `http://localhost:8765` in a web browser.

### Preprocessing & ML Dashboard
Start the Streamlit dashboard:
```bash
cd HeatMAP
streamlit run visauls.py
```

### Uploading ESP32 Firmware
Build and flash using PlatformIO CLI:
```bash
cd firmware/magi_esp32
pio run --target upload
```
