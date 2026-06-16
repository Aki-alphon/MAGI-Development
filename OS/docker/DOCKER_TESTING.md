# MAGI OS — Docker Testing Guide

## Prerequisites

| Tool | Version | Link |
|------|---------|------|
| Docker Desktop | ≥ 4.x (Linux containers mode) | [docker.com](https://www.docker.com/products/docker-desktop/) |
| Windows 10/11 | WSL2 backend recommended | |

---

## Quick Start (Windows)

```
double-click:  MAGI\OS\docker\start_magi.bat
```

Select **[1] Start ALL services** from the menu. That's it.

---

## Manual Commands

All commands run from `MAGI\OS\` directory:

### 1. Build the image (first time only, ~3–5 min)
```powershell
docker compose -f docker/docker-compose.yml build
```

### 2. Start all MAGI services
```powershell
docker compose -f docker/docker-compose.yml up -d
```

### 3. Watch live logs from all services
```powershell
docker compose -f docker/docker-compose.yml logs -f
```

### 4. Watch logs from one specific service
```powershell
docker compose -f docker/docker-compose.yml logs -f magi3
```

### 5. Open live dashboard (monitor)
```powershell
docker compose -f docker/docker-compose.yml --profile monitor up monitor
```

### 6. Stop everything
```powershell
docker compose -f docker/docker-compose.yml down
```

---

## What Each Service Does in Docker

| Container | Process | Mock Hardware Used |
|-----------|---------|-------------------|
| `magi_sensor_hub` | Reads sensors, publishes ZMQ | `smbus2` → sine-wave IMU, `pigpio` → random GPIO events |
| `magi_camera` | Writes camera frames to shared mem | Disabled (camera=false in test_config) |
| `magi1_melchior` | Object detection inference | `tflite_runtime` → returns 3 fake detections |
| `magi2_balthasar` | Scene analysis inference | `tflite_runtime` → returns random scene class |
| `magi3_caspar` | Fusion + decision engine | Rule engine (no mock needed) |

---

## Expected Log Output

After `docker compose up`, you should see:

```
magi_sensor_hub  | [sensor_hub] INFO — SensorHub ready — publishing to ipc:///tmp/magi/sensor_pub.sock
magi_sensor_hub  | [sensor_hub] INFO — IMU MPU-6050 initialized at I2C 0x68
magi1_melchior   | [magi1] INFO — Loading model: /opt/magi/models/melchior.tflite
magi1_melchior   | [mock_tflite] Interpreter stub for 'melchior'
magi1_melchior   | [magi1] INFO — Melchior ready — input 320x320, conf≥0.4
magi2_balthasar  | [magi2] INFO — Balthasar ready — input 224x224
magi3_caspar     | [magi3] INFO — Caspar ready — fusion engine active
magi3_caspar     | [magi3] INFO — Decision → TRACK (priority=4) | Tracking 3 object(s)
```

---

## Mock Hardware Behaviour

| Mock | Simulates |
|------|-----------|
| `smbus2` | MPU-6050 IMU: sine-wave accel (±0.05g), small gyro drift |
| `pigpio` | GPIO interrupts fire randomly every 15–30 seconds |
| `tflite_runtime` | Melchior: 3 random detections (person, cat, car). Balthasar: random scene. |
| `posix_ipc` | In-process bytearray (no real POSIX shm needed) |
| `spidev` | Returns zeros |
| `RPi.GPIO` | Silent no-op |

---

## Verifying the Pipeline

### Check MAGI-3 is making decisions:
```powershell
docker compose -f docker/docker-compose.yml logs magi3 --tail=20
```
Expected output:
```
[magi3] INFO — Decision → TRACK | Tracking 3 object(s)
[magi3] INFO — Decision → IDLE  | No significant events
[magi3] INFO — Decision → ALERT | High anomaly score: 0.72
```

### Check inference is running:
```powershell
docker compose -f docker/docker-compose.yml logs magi1 --tail=10
```
Expected output:
```
[magi1] INFO — Detected 3 object(s): ['person', 'cat', 'car']
[magi1] DEBUG — Inference: 12.4 ms
```

---

## File Structure (Docker-specific)

```
docker/
├── Dockerfile                   Base image definition
├── docker-compose.yml           All 5 service definitions
├── requirements-docker.txt      Python packages for test env
├── entrypoint.sh                Routes container command → process
├── start_magi.bat               Windows quick-start menu
├── test_config.yaml             Config with all sensors enabled (mocked)
├── .dockerignore                Keeps build context small
│
├── mock_hardware/               Hardware stub layer (on PYTHONPATH)
│   ├── pigpio.py                GPIO + interrupt simulation
│   ├── smbus2.py                I2C with realistic sine-wave IMU data
│   ├── spidev.py                SPI stub (returns zeros)
│   ├── posix_ipc.py             Shared memory via in-process bytearray
│   ├── cpu_affinity.py          Patches os.sched_setaffinity → no-op
│   ├── RPi/
│   │   ├── __init__.py
│   │   └── GPIO.py              RPi.GPIO stub
│   └── tflite_runtime/
│       ├── __init__.py
│       └── interpreter.py       Returns realistic fake model outputs
│
└── dummy_models/
    ├── melchior.tflite           16-byte placeholder (mock ignores content)
    ├── balthasar.tflite
    └── caspar.tflite
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `docker: command not found` | Install Docker Desktop, ensure it's running |
| `Error: No such service: magi1` | Run from `MAGI\OS\` directory, not `docker\` |
| IPC socket errors at startup | Normal — services wait with `sleep N` in entrypoint |
| `posix_ipc` import error on Windows | Already mocked — check PYTHONPATH includes `mock_hardware/` |
| `tflite_runtime` not found | Docker uses mock — not installed in `requirements-docker.txt` by design |

---

## Moving to Real RPi

When you're ready to deploy:
1. Replace `docker/dummy_models/*.tflite` with real model files
2. Copy entire `MAGI/OS/` to the Pi via `scp` or USB
3. Run `setup/01_strip_os.sh` through `setup/04_install_services.sh` in order
4. Real hardware replaces all mocks automatically (mock layer not in PYTHONPATH on Pi)
