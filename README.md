<div align="center">

# MAGI

### Multispectral Autonomous Ground Intelligence

**A 12-DOF Quadruped Robot with Edge AI, Custom OS, and Evolutionary Gait Training**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#license)
[![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi%204B-c51a4a.svg)](#hardware-overview)
[![MCU](https://img.shields.io/badge/MCU-ESP32-orange.svg)](#firmware--esp32-joint-controller)
[![Simulation](https://img.shields.io/badge/simulator-Three.js%20%2B%20PyBullet-brightgreen.svg)](#simulation--gait-training)
[![Paper](https://img.shields.io/badge/paper-IEEE%20format-yellow.svg)](#research-paper)

---

*Named after the [MAGI supercomputer](https://evangelion.fandom.com/wiki/MAGI) from Neon Genesis Evangelion — three independent systems (Celebi, Gengar, Lugia) that each reason autonomously and fuse their outputs into a single decision.*

</div>

---

## What Is MAGI?

MAGI is a complete, integrated autonomous robotics platform built from scratch. It combines a **custom quadruped robot**, a **purpose-built embedded operating system**, **real-time edge AI inference**, and a **browser-based 3D simulation environment** — all designed to operate within the constraints of a Raspberry Pi 4B with only 4 GB of RAM.

The project was originally developed for **precision agriculture** — a walking robot that can navigate uneven crop rows, identify diseased plants in real-time using onboard AI, and make autonomous decisions without any cloud connectivity.

### Core Problem Solved

> Run 3 AI models simultaneously on a $35 single-board computer while controlling a walking robot — with no cloud, no GPU, and no ROS2.

### What Makes MAGI Different

| Aspect | Traditional Approach | MAGI Approach |
|---|---|---|
| Middleware | ROS2 + DDS (~200 MB RAM) | Custom ZeroMQ middleware (22 MB RAM) |
| Inference | Single model, GPU-dependent | 3 concurrent TFLite models, CPU-only |
| Mobility | Wheeled UGV (flat terrain only) | 12-DOF quadruped (uneven terrain) |
| Gait Design | Hand-tuned parameters | Genetically evolved + neural network policies |
| Simulation | Gazebo (heavy install) | Zero-install browser-based 3D simulator |

---

## System Architecture

MAGI is composed of three tightly coupled subsystems that communicate through well-defined interfaces:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MAGI SYSTEM                                    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    MAGI OS  (Raspberry Pi 4B)                       │    │
│  │                                                                     │    │
│  │   Core 0: Sensor Hub ──────┐                                        │    │
│  │   Core 0: Camera Capture ──┤── POSIX Shared Memory (Zero-Copy) ──┐  │    │
│  │                            │                                      │  │    │
│  │   Core 1: CELEBI ──────────┤── Object/Disease Detection (TFLite)  │  │    │
│  │   Core 2: GENGAR ──────────┤── Scene Analysis + IMU Anomaly       │  │    │
│  │   Core 3: LUGIA ───────────┘── Sensor Fusion + Decision Engine    │  │    │
│  │                            │                                      │  │    │
│  │        ZeroMQ XPUB/XSUB Message Bus (8 MB RAM)                   │  │    │
│  │        Parameter Server · Diagnostics · TF Frames · Recorder      │  │    │
│  └────────────────────────────┬──────────────────────────────────────┘  │    │
│                               │ UART (115200 baud)                      │    │
│  ┌────────────────────────────▼──────────────────────────────────────┐  │    │
│  │                    ESP32 Firmware                                  │  │    │
│  │                                                                   │  │    │
│  │   3-DOF Inverse Kinematics Solver ──► PCA9685 ──► 12× MG996R     │  │    │
│  │   Gait Engine (Crawl / Trot / Bound) ──► Cubic Bézier Trajectories│  │    │
│  │   MPU-6050 IMU ──► Complementary Filter                           │  │    │
│  └───────────────────────────────────────────────────────────────────┘  │    │
│                                                                         │    │
│  ┌───────────────────────────────────────────────────────────────────┐  │    │
│  │              Browser Simulation & Training                        │  │    │
│  │                                                                   │  │    │
│  │   Three.js WebGL 3D Visualizer ──► Real-time IK + Gait Playback  │  │    │
│  │   Genetic Algorithm Trainer ──► Evolve Gait Params / NN Weights   │  │    │
│  │   PyBullet Physics ──► URDF Rigid-Body Validation                 │  │    │
│  └───────────────────────────────────────────────────────────────────┘  │    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### How the Three Subsystems Connect

```
                    ┌──────────────────────────┐
                    │    3D SIMULATION         │
                    │    (Browser / PyBullet)  │
                    │                          │
                    │  Evolve gait parameters  │
                    │  Validate IK equations   │
                    │  Test decision logic     │
                    └──────────┬───────────────┘
                               │
                    Export optimized params
                    (JSON genome / NN weights)
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                        MAGI OS                                   │
│                   (Raspberry Pi 4B)                               │
│                                                                  │
│  Camera ──► Shared Memory ──► Celebi (Detection)                 │
│                            ──► Gengar (Scene)                    │
│  Sensors ──► Sensor Hub ──────────────────────► Lugia (Decision) │
│                                                     │            │
│  Decision: IDLE / TRACK / ALERT / EMERGENCY         │            │
└─────────────────────────────────────────────────────┬────────────┘
                                                      │
                                             UART Commands
                                           (GAIT/MOVE/STOP)
                                                      │
                                                      ▼
                                    ┌─────────────────────────────┐
                                    │       ESP32 FIRMWARE        │
                                    │                             │
                                    │  Parse command ──► IK solve │
                                    │  ──► Gait engine ──► Servos │
                                    │  ──► IMU feedback ──► Pi    │
                                    └─────────────────────────────┘
```

1. **Simulation** evolves optimal gait parameters offline using a genetic algorithm. The best genome (12 parameters or ~300 neural network weights) is exported as JSON.
2. **MAGI OS** runs the AI inference stack. Three models process sensor data in parallel and the decision engine (Lugia) outputs high-level commands.
3. **ESP32 Firmware** receives commands over UART, solves inverse kinematics in real-time, and drives the 12 servo motors to walk.

---

## Repository Layout

```
MAGI/
│
├── OS/                           Embedded operating system (Raspberry Pi 4B)
│   ├── setup/                    4-stage deployment scripts
│   ├── src/
│   │   ├── core/                 ROS2-grade middleware (native implementation)
│   │   │   ├── message_bus.py    ZeroMQ XPUB/XSUB broker
│   │   │   ├── messages.py       Typed message dataclasses (msgpack serialized)
│   │   │   ├── lifecycle.py      Managed node lifecycle state machine
│   │   │   ├── qos.py            Per-topic Quality of Service profiles
│   │   │   ├── param_server.py   Runtime parameter store (ZMQ REQ/REP)
│   │   │   ├── diagnostics.py    Node health monitor
│   │   │   ├── transforms.py     TF2 coordinate frame registry
│   │   │   └── recorder.py       SQLite message recorder (rosbag equivalent)
│   │   ├── sensors/              Hardware I/O layer
│   │   ├── magi1/celebi.py       Object/disease detection (Core 1)
│   │   ├── magi2/gengar.py       Scene analysis + anomaly (Core 2)
│   │   ├── magi3/lugia.py        Fusion + decision engine (Core 3)
│   │   └── watchdog.py           Process supervisor + auto-restart
│   ├── services/                 Systemd unit files
│   └── docker/                   Hardware-mocked testing environment
│
├── firmware/                     ESP32 joint controller
│   ├── magi_esp32/               PlatformIO project
│   │   └── src/
│   │       ├── main.cpp          Entry point
│   │       ├── kinematics.h/cpp  3-DOF IK solver
│   │       ├── gait_engine.h/cpp Crawl / Trot / Bound gait patterns
│   │       ├── servo_controller  PCA9685 I²C servo driver
│   │       ├── imu_driver        MPU-6050 complementary filter
│   │       └── uart_protocol     Pi ↔ ESP32 command parser
│   └── tests/                    7 standalone hardware test sketches
│
├── simulation/                   Browser-based 3D simulator + training
│   ├── index.html                Main UI (5 tabbed interface)
│   ├── robot.js                  Three.js WebGL viewport + IK solver
│   ├── gait.js                   Cubic Bézier foot trajectory generator
│   ├── rl_train.js               Genetic algorithm + NNPolicy trainer
│   ├── magi_fusion.js            Lugia decision engine (browser mirror)
│   ├── dashboard.js              60fps coordination loop
│   ├── SpiderQ.urdf / .glb       Robot geometry and 3D meshes
│   ├── MAGI_gait_learning.py     Standalone PyBullet RL trainer
│   └── ik.py / fk.py             Reference Python kinematics solvers
│
├── model & miscellenious/        3D print files (chassis, legs, joints)
│   └── MAGI.3mf                  Full assembly CAD project
│
├── HeatMAP/                      ML preprocessing evaluation dashboard
│   ├── visauls.py                Streamlit app (CLAHE, confusion matrix)
│   └── Heatmap_algo.ipynb        Baseline spectral heatmap algorithm
│
└── Research paper/               IEEE-format research paper (LaTeX)
    └── main.tex                  Full paper source
```

---

## MAGI OS — The Brain

### What It Is

MAGI OS is a **purpose-built, minimal Linux-based operating system** designed to run on a Raspberry Pi 4B. It is not a general-purpose OS — every decision from the init system to the IPC mechanism was made to serve one mission: run 3 AI inference models simultaneously while talking to all connected sensors.

### Why Not Just Use ROS2?

| Concern | ROS2 + DDS | MAGI OS |
|---|---|---|
| RAM at boot | ~600–800 MB | **~180 MB** |
| Middleware overhead | ~200 MB (Fast-DDS) | **~22 MB** (ZeroMQ) |
| Boot time | ~30–60 sec | **~8 sec** |
| Frame passing | Serialized via DDS | **Zero-copy** shared memory |
| Fault isolation | Threads share memory | **Process isolation** per core |

ROS2's DDS middleware alone consumes 80–150 MB of RAM — nearly the entire headroom budget on a 4 GB Pi running 3 inference models. MAGI OS extracts the 8 most useful **concepts** from ROS2 and reimplements them natively in Python + ZeroMQ.

### The Three MAGI Nodes

Named after the MAGI supercomputer from Evangelion, each node runs as an **isolated process pinned to a dedicated CPU core**:

| Node | CPU Core | Role | Model | RAM |
|---|---|---|---|---|
| **Celebi** (MAGI-1) | Core 1 | Object / Disease Detection | YOLOv8-nano / MobileNetV2 TFLite | ~300 MB |
| **Gengar** (MAGI-2) | Core 2 | Scene Classification + Motion Anomaly | EfficientNet-Lite TFLite | ~200 MB |
| **Lugia** (MAGI-3) | Core 3 | Sensor Fusion + Decision Engine | Rule engine (+ optional LSTM) | ~150 MB |
| Sensor Hub | Core 0 | All hardware I/O (I2C, SPI, UART, GPIO) | — | ~80 MB |

**Total system RAM: ~1252 MB out of 4096 MB (30.6% utilization)**

### Data Flow Through the OS

```
Sensors (I2C/SPI/UART/GPIO)       USB Camera
        │                              │
        ▼                              ▼
  sensor_hub.py ──► /sensors    camera_capture.py ──► POSIX shared memory
       (Core 0)                    (Core 0)                 │
                                                     ┌──────┴──────┐
                                                     │             │
                                                     ▼             ▼
                                              celebi.py      gengar.py
                                               (Core 1)       (Core 2)
                                                     │             │
                                              /detections      /scene
                                                     │             │
                                                     └──────┬──────┘
                                                            ▼
                                                      lugia.py
                                                       (Core 3)
                                                            │
                                                       /decision
                                                            │
                                               ┌────────────┼────────────┐
                                               ▼            ▼            ▼
                                          UART to ESP32  GPIO alert   Log file
```

### ROS2-Equivalent Middleware (Native Implementation)

MAGI OS implements **8 ROS2 features** without installing ROS2:

| ROS2 Feature | MAGI Implementation | How It Works | RAM |
|---|---|---|---|
| `roscore` / DDS | `message_bus.py` | ZeroMQ XPUB/XSUB proxy on ports 5555/5556 | 8 MB |
| `.msg` files | `messages.py` | Python dataclasses + msgpack serialization (10× faster than JSON) | 0 MB |
| Managed Nodes | `lifecycle.py` | 4-state FSM: UNCONFIGURED → INACTIVE → ACTIVE → FINALIZED | 0 MB |
| QoS Profiles | `qos.py` | RELIABLE/BEST_EFFORT, TRANSIENT_LOCAL durability per topic | 0 MB |
| Parameter Server | `param_server.py` | ZMQ REQ/REP on port 5558, seeded from `config.yaml` | 4 MB |
| `ros_diagnostics` | `diagnostics.py` | Node health pings to `/diagnostics` every 1s | 3 MB |
| `tf2` transforms | `transforms.py` | Quaternion-based sensor coordinate frame registry | 2 MB |
| `rosbag` | `recorder.py` | SQLite message recording + variable-speed playback | 5 MB |
| **Total** | | | **22 MB** |

### Lifecycle-Gated Inference

A key innovation: **Celebi doesn't run all the time.** The disease classifier (the most expensive model) stays in `INACTIVE` state until Gengar detects a stable plant canopy:

```
                                    ┌─────────────┐
          configure()               │             │  activate()
  UNCONFIGURED ──────► INACTIVE ────►   ACTIVE    ◄──── Gengar says
   (boot)          (model loaded,   │ (inferring) │     "stable_plant"
                    waiting)        │             │
                                    └──────┬──────┘
                                           │ deactivate()
                                           │ (no plant for 5s)
                                           ▼
                                       INACTIVE
                                   (model stays loaded,
                                    no CPU used)
```

This reduces average CPU utilization by **34%** and thermal dissipation by **8°C** compared to always-on inference.

### Decision Engine (Lugia)

Lugia fuses outputs from Celebi, Gengar, and raw sensors into priority-ordered actions:

| Priority | Condition | Action | Effect |
|---|---|---|---|
| 10 | ToF sensor < 200mm | **EMERGENCY** | Halt all servos, drop body |
| 9 | GPIO hardware trigger | **EMERGENCY** | Immediate stop |
| 8 | Person in restricted zone | **EMERGENCY** | Safety halt |
| 6 | Anomaly score > 0.7 | **ALERT** | Slow cautious crawl |
| 5 | Scene = emergency/obstacle | **ANALYZE** | Evaluate environment |
| 4 | Detections present | **TRACK** | Lean forward, inspect |
| 0 | Nothing | **IDLE** | Continue normal walk |

The anomaly score combines scene classification entropy with IMU motion:

```
anomaly = min(1.0, (H(softmax) / 3.0) × (1 + ‖acceleration‖))
```

High uncertainty + platform movement = high anomaly = cautious behavior.

---

## Firmware — ESP32 Joint Controller

### What It Does

The ESP32 handles everything that requires **hard real-time guarantees**: inverse kinematics solving at 50 Hz, servo PWM generation, IMU reading, and communication with the Raspberry Pi.

### Hardware Stack

| Component | Specification |
|---|---|
| **MCU** | ESP32 WROOM-32 |
| **Servos** | 12× MG996R metal-gear (3 per leg × 4 legs) |
| **PWM Driver** | PCA9685 16-channel I²C @ 400 kHz |
| **IMU** | MPU-6050 6-axis (accel + gyro) on I²C @ 0x68 |
| **Power (Servos)** | XL4016 DC-DC buck → 6.0V / 20A |
| **Power (Logic)** | 3S LiPo → 5V buck → ESP32 + Pi |
| **Battery** | 3S LiPo 5200 mAh 35C (XT60 connector) |
| **Communication** | UART 115200 baud (GPIO16 RX / GPIO17 TX) |

### The 12-DOF Leg Architecture

```
    [FR = Leg 1]  [FL = Leg 2]         Each leg has 3 joints:
       |||           |||
    [====BODY==========]                 Coxa (θc)  ← horizontal yaw
       |||           |||                 Femur (θf) ← vertical swing (hip)
    [BR = Leg 3]  [BL = Leg 4]          Tibia (θt) ← knee bend

                                        Link lengths:
                                         Coxa  = 30mm
     Shoulder pivot                      Femur = 90mm
          │                              Tibia = 90mm
       [Coxa]   ← θc
          │
       [Femur]  ← θf
          │
       [Tibia]  ← θt
          │
       [Foot]
```

### Inverse Kinematics Equations

Given a target foot position (x, y, z) relative to the shoulder joint:

```
θc = atan2(y, x)                                    ← coxa angle

r  = √(x² + y²)                                     ← horizontal reach
Ld = √((r − L_coxa)² + z²)                          ← distance in sagittal plane

θt = π − arccos[(L_femur² + L_tibia² − Ld²) / (2 × L_femur × L_tibia)]    ← law of cosines

α  = atan2(z, r − L_coxa)
β  = arccos[(L_femur² + Ld² − L_tibia²) / (2 × L_femur × Ld)]
θf = α + β                                          ← femur angle
```

These equations are implemented identically in:
- **C++** on the ESP32 (`kinematics.cpp`) for real-time control
- **JavaScript** in the browser (`robot.js`) for simulation
- **Python** reference solvers (`ik.py`, `fk.py`)

### UART Command Protocol

All commands are newline-terminated ASCII (debuggable with any serial monitor):

| Command | Format | Example | Direction |
|---|---|---|---|
| Move joints | `MOVE a0,a1,...,a11\n` | `MOVE 90,45,90,90,45,90,...\n` | Pi → ESP32 |
| Set gait | `GAIT crawl\|trot\|stand\n` | `GAIT trot\n` | Pi → ESP32 |
| Set speed | `SPEED 0-100\n` | `SPEED 60\n` | Pi → ESP32 |
| Emergency stop | `STOP\n` | `STOP\n` | Pi → ESP32 |
| Home position | `HOME\n` | `HOME\n` | Pi → ESP32 |
| Request IMU | `IMU\n` | Response: `IMU ax,ay,az,gx,gy,gz\n` | Both ways |

### Gait Patterns

| Gait | Phase Pattern | Duty Cycle | Flight Phase | Use Case |
|---|---|---|---|---|
| **Crawl** | One leg at a time (FR→FL→BL→BR) | 75% | No | Maximum stability, rough terrain |
| **Trot** | Diagonal pairs (FR+BL, FL+BR) | 60% | No | Normal locomotion, moderate speed |
| **Gallop** | Staggered fore + rear burst | 40% | Yes | Maximum speed |
| **Bound** | Front pair → Rear pair | 35% | Yes | High energy forward propulsion |

Foot trajectories use **cubic Bézier curves** for biomechanically natural swing arcs:

```
Foot_X(t) = Bézier(t, startX, startX+0.3×stride, startX+0.7×stride, endX)
Foot_Y(t) = Bézier(t, 0, 0.1×height, 1.1×height, 0)

where t ∈ [0,1] is normalized swing progress
```

This produces slow-lift → fast-peak → rapid-plant motion (mimicking biological limbs) with guaranteed non-negative ground clearance.

---

## Simulation & Gait Training

### Overview

The simulation environment runs **entirely in the browser** — no installation required. It provides 5 integrated tools:

| Tab | Purpose |
|---|---|
| **🤖 Locomotion Sim** | Real-time 3D robot walking with adjustable gait parameters |
| **🧠 RL Gait Training** | Genetic algorithm evolving locomotion policies |
| **⚡ Decision Fusion** | MAGI-3 Lugia decision engine simulation |
| **📊 Gait Analysis** | Phase timing diagrams, foot path traces, energy metrics |
| **📐 Kinematics Math** | Interactive IK equation debugger with 2D visualizer |

### Running the Simulator

```bash
cd simulation
python3 -m http.server 8765
# Open http://localhost:8765 in your browser
```

### The Training Pipeline

MAGI uses a **Genetic Algorithm (GA)** to evolve optimal gait parameters. Training runs entirely client-side in JavaScript:

```
┌──────────────────────────────────────────────────────────────────┐
│  1. Create population of N agents with random genes/weights      │
│  2. Evaluate each agent for 5 simulated seconds                  │
│  3. Score with multi-objective fitness function                   │
│  4. Select top 20% (elitism) → crossover → mutate               │
│  5. Repeat from step 2 until convergence                         │
│  6. Export best genome as JSON → deploy to robot                 │
└──────────────────────────────────────────────────────────────────┘
```

### Two Policy Modes

**Direct Gene Mode** — 12 genes map directly to gait parameters:

| Genes 0–3 | Genes 4–7 | Genes 8–11 |
|---|---|---|
| Phase offsets per leg (0.0–1.0) | Stride length multipliers (×0.5–×1.5) | Body pitch sway, roll sway, step duration, height bias |

**NNPolicy Mode** — An 8→16→12 feedforward neural network where all ~300 weights are evolved:

```
Inputs (8):  sin/cos phase (×2), body pitch, roll, height, friction
Hidden (16): ReLU neurons
Outputs (12): Same 12 locomotion parameters as gene mode
```

### Multi-Objective Fitness Function

```
Fitness = Σ over time of:
  + velocity_score × forward_velocity          ← go fast
  − 3.5 × (|pitch| + |roll|)                  ← stay stable
  − 0.08 × Σ(Δjoint²)                         ← minimize energy
  + 5.0 × symmetry_score                       ← balanced L/R gait
```

The trainer also implements **curriculum learning** — automatically increasing difficulty (stride range, terrain variation) as the population fitness improves through 5 levels.

### PyBullet Physics Validation

For rigid-body physics validation beyond the browser, a standalone Python trainer (`MAGI_gait_learning.py`) uses PyBullet with the URDF model:

```bash
pip install pybullet numpy
python3 simulation/MAGI_gait_learning.py
```

### Sim-to-Real Pipeline

```
Browser GA Training ──► Export best genome (JSON)
                              │
                              ▼
                    MAGI OS loads parameters
                              │
                              ▼
                    Lugia selects gait mode
                              │
                              ▼
                    UART: GAIT trot + SPEED 60
                              │
                              ▼
                    ESP32 applies evolved params
                    to Bézier gait engine
                              │
                              ▼
                    Robot walks with optimized gait
```

---

## The Connection Between Everything

### Full System Data Flow

Here's how data flows through the complete system during operation:

```
 PHYSICAL WORLD                   RASPBERRY PI 4B                    ESP32
 ═══════════════        ═══════════════════════════════       ═══════════════
                         
 USB Camera ──────────► camera_capture.py (Core 0)
                              │
                         POSIX Shared Memory
                         (zero-copy, ~900KB/frame)
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              celebi.py           gengar.py
               (Core 1)            (Core 2)
                    │                   │
            Spectral Masking    Scene Classification
            → CLAHE → Resize   → EfficientNet-Lite
            → MobileNetV2      → IMU Anomaly Score
                    │                   │
                    ▼                   ▼
              /detections            /scene
              (Detection[]         (SceneMsg)
               with bbox,               │
               label,              ┌────┘
               confidence)         │  Lifecycle Gate:
                    │              │  "stable_plant"
                    │              │  activates Celebi
                    ▼              ▼
              ┌─────────────────────┐
              │    lugia.py         │
              │     (Core 3)       │
              │                    │
              │  + /sensors data   │
              │  + ToF distance    │
              │  + GPIO triggers   │
              │                    │
              │  Priority Rules    │
              │  ──────────────    │
              │  EMERGENCY (P10)   │
              │  ALERT     (P6)    │
              │  TRACK     (P4)    │
              │  IDLE      (P0)    │
              └────────┬───────────┘
                       │
                   /decision
                       │
                       ▼
                  UART Command ──────────────────────► uart_protocol.cpp
                  "GAIT crawl\n"                          │
                  "SPEED 40\n"                            ▼
                  "STOP\n"                          gait_engine.cpp
                                                         │
                                                    kinematics.cpp
                                                   (IK solver @ 50Hz)
                                                         │
                                                   servo_controller.cpp
                                                         │
                                                    PCA9685 I²C bus
                                                         │
                                                  12× MG996R Servos
                                                         │
                                                    ROBOT WALKS
```

### Communication Interfaces

| Interface | Between | Protocol | Data | Rate |
|---|---|---|---|---|
| POSIX Shared Memory | Camera → Celebi/Gengar | Zero-copy mmap | Raw BGR frames (640×480) | 30 fps |
| ZeroMQ XPUB/XSUB | All MAGI nodes | msgpack over TCP | Typed messages (200–500 bytes) | 10–50 Hz |
| ZeroMQ REQ/REP | CLI/Nodes → Param Server | Key-value protocol | Config parameters | On-demand |
| UART | Pi → ESP32 | ASCII newline-delimited | Commands (MOVE/GAIT/STOP) | 115200 baud |
| I²C | ESP32 → PCA9685 | PWM register writes | 12 servo angles | 50 Hz |
| I²C | ESP32 → MPU-6050 | Sensor register reads | 6-axis accel/gyro | 100 Hz |

### How Simulation Feeds Into the Real Robot

The simulation doesn't just visualize — it produces **deployable artifacts**:

| What Simulation Produces | How the Robot Uses It |
|---|---|
| Evolved gait genome (12 params) | Loaded by ESP32 gait engine as initial parameters |
| Trained NN weights (~300 floats) | Optional: Lugia sends real-time params via UART |
| Validated IK equations | Same math runs in C++ on ESP32 (`kinematics.cpp`) |
| Decision engine rules | Identical priority chain runs in Python on Pi (`lugia.py`) |
| Phase offset sequences | Define which gait pattern to use for each terrain type |

---

## Hardware Overview

### Electronics Architecture

```
                           ┌───────────────────────┐
                           │   3S LiPo Battery     │
                           │   5200mAh 35C (XT60)  │
                           └─────────┬─────────────┘
                                     │
                          ┌──────────┼──────────┐
                          │                     │
                   XL4016 Buck              5V Buck
                   6.0V / 20A              Regulator
                          │                     │
                          ▼                     ▼
                   ┌─────────────┐    ┌──────────────────┐
                   │  PCA9685    │    │  Raspberry Pi 4B  │
                   │  (12 Servo  │    │  4GB LPDDR4       │
                   │   Channels) │    │                   │
                   └──────┬──────┘    │  USB Camera       │
                          │           │  WiFi/SSH         │
                   12× MG996R         └────────┬──────────┘
                   Servos                      │ UART
                          ▲                    │ (GPIO 14/15)
                          │ I²C                │
                   ┌──────┴──────┐    ┌────────▼──────────┐
                   │  MPU-6050   │────│     ESP32         │
                   │  IMU        │I²C │  WROOM-32         │
                   └─────────────┘    └───────────────────┘
```

### 3D Printed Chassis

The chassis is fully 3D-printable. Manufacturing files are in `model & miscellenious/`:
- **MAGI.3mf** — Full assembly with print tray layouts
- Central electronics enclosure (Pi 4B + ESP32 mounts)
- 4× leg assemblies: Coxa → Femur → Tibia linkages
- Servo joint brackets (MG996R compatible)
- Battery compartment and power distribution plate

---

## Edge AI & Preprocessing

### Spectral Masking Pipeline

MAGI uses a 4-stage fixed-function preprocessing chain called **spectral masking** that improves classification accuracy by **7.5 percentage points** on field-realistic images:

```
Raw BGR Frame (640×480, ~900KB)
        │
        ▼
  Stage 1: Spatial Downsampling
  cv2.resize → 224×224 (8× data reduction)
        │
        ▼
  Stage 2: CLAHE Spectral Normalization
  BGR → LAB → CLAHE on L channel (clip=2.0, grid=8×8)
  → LAB → RGB
  (suppresses shadows, specular reflections, illumination gradients)
        │
        ▼
  Stage 3: Tensor Scaling
  uint8 [0,255] → float32 [0.0, 1.0]
        │
        ▼
  Stage 4: ImageNet Channel Normalization
  T = (I − μ) / σ
  μ = [0.485, 0.456, 0.406]
  σ = [0.229, 0.224, 0.225]
        │
        ▼
  Ready for TFLite Inference (1×224×224×3 float32)
```

| Configuration | Accuracy | F1 Score | Latency |
|---|---|---|---|
| No preprocessing (raw uint8) | 89.3% | 0.882 | 187.2 ms |
| + Resize | 90.1% | 0.891 | 183.5 ms |
| + CLAHE | 95.4% | 0.948 | 186.1 ms |
| + float32 scaling | 95.7% | 0.951 | 186.8 ms |
| + ImageNet normalization (full pipeline) | **96.8%** | **0.964** | 187.4 ms |

### ML Evaluation Dashboard

The `HeatMAP/` Streamlit dashboard provides interactive tools for:
- **Preprocessing pipeline simulation** — toggle individual stages and observe effects
- **Threshold tuning** — adjust classification confidence thresholds with live confusion matrix
- **Memory budget analysis** — warnings for running 12MP images on Pi hardware

```bash
cd HeatMAP
pip install streamlit pandas numpy plotly
streamlit run visauls.py
```

---

## Quick Start

### 1. Run the 3D Simulator (No hardware needed)

```bash
cd simulation
python3 -m http.server 8765
# Open http://localhost:8765
```

### 2. Test MAGI OS in Docker (No Pi needed)

```bash
cd OS/docker
docker compose build
docker compose up -d
docker compose logs -f magi3    # Watch decision engine
```

All hardware is mocked — real IPC, lifecycle, and decision logic execute on your machine.

### 3. Deploy to Raspberry Pi

```bash
# Flash Raspberry Pi OS Lite (Bookworm, 64-bit) to SD card
scp -r MAGI/OS pi@192.168.1.100:~/magi-os
ssh pi@192.168.1.100

sudo bash ~/magi-os/setup/01_strip_os.sh && sudo reboot
sudo bash ~/magi-os/setup/02_install_deps.sh
sudo bash ~/magi-os/setup/03_configure_boot.sh && sudo reboot
sudo bash ~/magi-os/setup/04_install_services.sh

# Place your TFLite models
sudo cp *.tflite /opt/magi/models/

# Start MAGI
sudo systemctl start magi-watchdog
```

### 4. Flash ESP32 Firmware

```bash
cd firmware/magi_esp32
pip install platformio
pio run --target upload --upload-port /dev/ttyUSB0
pio device monitor --baud 115200
```

---

## RAM Budget

| Component | RAM |
|---|---|
| Stripped Linux kernel + initramfs | ~40 MB |
| System daemons (udev, sshd, journald) | ~60 MB |
| Python runtime + sensor middleware | ~80 MB |
| **OS Baseline** | **~180 MB** |
| ZeroMQ message bus | ~8 MB |
| Parameter server + diagnostics + TF + recorder | ~14 MB |
| **Middleware Total** | **~22 MB** |
| MAGI-1 Celebi (detection model) | ~300 MB |
| MAGI-2 Gengar (scene model) | ~200 MB |
| MAGI-3 Lugia (fusion engine) | ~150 MB |
| Inference buffers + preprocessing | ~400 MB |
| **AI Total** | **~1050 MB** |
| **TOTAL USED** | **~1252 MB** |
| **FREE HEADROOM** | **~2844 MB ✅** |

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **OS** | Raspberry Pi OS Lite (Bookworm, ARM64), Python 3.11, ZeroMQ, POSIX IPC |
| **AI** | TensorFlow Lite + XNNPACK, MobileNetV2, EfficientNet-Lite, YOLOv8-nano |
| **Firmware** | ESP32 (C++, PlatformIO), PCA9685 I²C, MPU-6050 IMU |
| **Simulation** | Three.js (WebGL), vanilla JavaScript, HTML/CSS |
| **Physics** | PyBullet, URDF, NumPy |
| **ML Tools** | Streamlit, Plotly, OpenCV (CLAHE) |
| **Hardware** | 3D printed PLA chassis, MG996R servos, 3S LiPo |
| **DevOps** | Docker, systemd, PlatformIO CLI |

---

## License

MAGI — Multispectral Autonomous Ground Intelligence  
Custom-built autonomous robotics platform for edge AI inference and quadruped locomotion.
