#!/bin/bash
# =============================================================================
# MAGI OS — Phase 4: Deploy Services to System
# Run this after copying /opt/magi onto the Pi
# =============================================================================

set -euo pipefail

MAGI_DIR="/opt/magi"
SERVICE_DIR="/etc/systemd/system"
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "======================================================"
echo " MAGI OS — Step 4: Deploy & Enable Services"
echo " Source: $SRC_DIR"
echo "======================================================"

# ── Copy project files ────────────────────────────────────
echo "[MAGI] Copying project files to $MAGI_DIR..."
cp -r "$SRC_DIR/src/"*    "$MAGI_DIR/src/"
cp -r "$SRC_DIR/src/common/config.yaml" "$MAGI_DIR/config/config.yaml"
chmod -R 755 "$MAGI_DIR/src"

# ── Create __init__.py for packages ──────────────────────
touch "$MAGI_DIR/src/common/__init__.py"
touch "$MAGI_DIR/src/sensors/__init__.py"
touch "$MAGI_DIR/src/magi1/__init__.py"
touch "$MAGI_DIR/src/magi2/__init__.py"
touch "$MAGI_DIR/src/magi3/__init__.py"

# ── Create placeholder model files (replace with real .tflite) ──
echo "[MAGI] Checking model directory..."
for model in celebi gengar lugia; do
    path="$MAGI_DIR/models/${model}.tflite"
    if [ ! -f "$path" ]; then
        echo "  WARNING: $path not found — place your TFLite model there"
    else
        echo "  OK: $path"
    fi
done

# ── Install systemd service ───────────────────────────────
echo "[MAGI] Installing systemd service..."
cp "$SRC_DIR/services/magi-watchdog.service" "$SERVICE_DIR/"
systemctl daemon-reload
systemctl enable magi-watchdog.service

# ── Create tmpfs mount ────────────────────────────────────
grep -q '/tmp/magi' /etc/fstab || \
    echo "tmpfs /tmp/magi tmpfs defaults,size=512M,mode=1777 0 0" >> /etc/fstab
mount /tmp/magi 2>/dev/null || true

# ── Set permissions ──────────────────────────────────────
chown -R root:root "$MAGI_DIR"
chmod -R 755 "$MAGI_DIR/src"
chmod 644 "$MAGI_DIR/config/config.yaml"
mkdir -p "$MAGI_DIR/logs"

echo ""
echo "======================================================"
echo " MAGI OS Deployment COMPLETE"
echo ""
echo " To start:  sudo systemctl start magi-watchdog"
echo " To stop:   sudo systemctl stop  magi-watchdog"
echo " Status:    sudo systemctl status magi-watchdog"
echo " Live logs: sudo journalctl -fu magi-watchdog"
echo " Monitor:   python3 /opt/magi/src/status_monitor.py"
echo "======================================================"
