#!/bin/bash
# =============================================================================
# MAGI OS Docker — Entrypoint
# Routes to the correct process based on the command argument
# =============================================================================

set -e

export PYTHONPATH="/opt/magi/src:/opt/magi/mock_hardware"

# Ensure IPC directory exists
mkdir -p /tmp/magi

echo "======================================"
echo " MAGI OS Docker — Starting: $1"
echo " ENV: $MAGI_ENV"
echo " PYTHONPATH: $PYTHONPATH"
echo "======================================"

case "$1" in
    message_bus)
        echo "[MAGI] Launching Message Bus..."
        exec python3 /opt/magi/src/core/message_bus.py
        ;;
    param_server)
        echo "[MAGI] Launching Parameter Server..."
        sleep 1   # Wait for message_bus
        exec python3 /opt/magi/src/core/param_server.py
        ;;
    batch_manager)
        echo "[MAGI] Launching Batch Manager..."
        sleep 5
        exec python3 /opt/magi/src/core/batch_manager.py
        ;;
    dashboard)
        echo "[MAGI] Launching Web Telemetry Dashboard..."
        sleep 6
        exec python3 /opt/magi/src/core/dashboard.py
        ;;
    sensor_hub)
        echo "[MAGI] Launching Sensor Hub..."
        sleep 2
        exec python3 /opt/magi/src/sensors/sensor_hub.py
        ;;
    camera)
        echo "[MAGI] Launching Camera Capture (simulated)..."
        sleep 3
        exec python3 /opt/magi/src/sensors/camera_capture.py
        ;;
    magi1)
        echo "[MAGI] Launching MAGI-1 Melchior (Detection)..."
        sleep 4   # Wait for sensor_hub to be ready
        exec python3 /opt/magi/src/magi1/melchior.py
        ;;
    magi2)
        echo "[MAGI] Launching MAGI-2 Balthasar (Analysis)..."
        sleep 4
        exec python3 /opt/magi/src/magi2/balthasar.py
        ;;
    magi3)
        echo "[MAGI] Launching MAGI-3 Caspar (Fusion)..."
        sleep 7   # Wait for magi1 + magi2
        exec python3 /opt/magi/src/magi3/caspar.py
        ;;
    monitor)
        echo "[MAGI] Launching Status Monitor..."
        sleep 8
        exec python3 /opt/magi/src/status_monitor.py
        ;;
    test)
        echo "[MAGI] Running test suite..."
        exec python3 -m pytest /opt/magi/src/tests/ -v --timeout=30
        ;;
    bash)
        exec /bin/bash
        ;;
    *)
        echo "Unknown command: $1"
        echo "Usage: sensor_hub | camera | magi1 | magi2 | magi3 | monitor | test | bash"
        exit 1
        ;;
esac
