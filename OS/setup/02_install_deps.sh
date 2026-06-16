#!/bin/bash
# =============================================================================
# MAGI OS — Phase 2: Install Dependencies
# Run after reboot from 01_strip_os.sh
# =============================================================================

set -euo pipefail
LOG="/var/log/magi_setup.log"
exec > >(tee -a "$LOG") 2>&1

echo "======================================================"
echo " MAGI OS Setup — Step 2: Install Dependencies"
echo " $(date)"
echo "======================================================"

# --- System packages ---
echo "[MAGI] Installing system dependencies..."
apt-get update -y
apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-numpy \
    python3-smbus2 \
    python3-serial \
    python3-yaml \
    pigpio \
    python3-pigpio \
    i2c-tools \
    spidev \
    git \
    htop \
    libopenblas-dev \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev \
    libusb-1.0-0-dev \
    libzmq3-dev \
    cmake \
    build-essential

# --- Create Python venv for MAGI ---
echo "[MAGI] Creating Python virtual environment..."
python3 -m venv /opt/magi/venv --system-site-packages
source /opt/magi/venv/bin/activate

# --- Python packages ---
echo "[MAGI] Installing Python packages..."
pip install --upgrade pip wheel

# ZeroMQ + msgpack for IPC
pip install pyzmq msgpack

# Sensor libraries
pip install \
    smbus2 \
    spidev \
    pyserial \
    RPi.GPIO \
    posix-ipc \
    psutil \
    pyyaml \
    numpy \
    pandas \
    pyarrow \
    zarr \
    zstandard

# TFLite runtime (ARM64 optimized — NOT full TensorFlow)
echo "[MAGI] Installing TFLite runtime..."
pip install tflite-runtime

# ONNX Runtime (ARM64 build)
echo "[MAGI] Installing ONNX Runtime..."
pip install onnxruntime

# OpenCV (headless — no GUI deps)
pip install opencv-python-headless

# Watchdog
pip install watchdog

deactivate

# --- Configure pigpio to start at boot ---
echo "[MAGI] Enabling pigpio daemon..."
systemctl enable pigpiod
systemctl start pigpiod

# --- Enable I2C and SPI via raspi-config-like approach ---
echo "[MAGI] Enabling I2C and SPI kernel modules..."
# Add to /etc/modules if not already present
grep -q '^i2c-dev' /etc/modules || echo "i2c-dev" >> /etc/modules
grep -q '^spi-dev' /etc/modules || echo "spi-dev" >> /etc/modules

# /boot/firmware/config.txt tweaks
CONFIG="/boot/firmware/config.txt"
grep -q 'dtparam=i2c_arm=on' "$CONFIG" || echo "dtparam=i2c_arm=on" >> "$CONFIG"
grep -q 'dtparam=spi=on' "$CONFIG" || echo "dtparam=spi=on" >> "$CONFIG"
grep -q 'enable_uart=1' "$CONFIG" || echo "enable_uart=1" >> "$CONFIG"

# GPU memory reduction (headless mode)
grep -q '^gpu_mem=' "$CONFIG" && \
    sed -i 's/^gpu_mem=.*/gpu_mem=64/' "$CONFIG" || \
    echo "gpu_mem=64" >> "$CONFIG"

# Disable Wi-Fi and Bluetooth overlays
grep -q 'dtoverlay=disable-wifi' "$CONFIG" || echo "dtoverlay=disable-wifi" >> "$CONFIG"
grep -q 'dtoverlay=disable-bt' "$CONFIG" || echo "dtoverlay=disable-bt" >> "$CONFIG"

echo ""
echo "======================================================"
echo " Step 2 COMPLETE — Run 03_configure_boot.sh next"
echo "======================================================"
