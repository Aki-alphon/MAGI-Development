#!/bin/bash
# =============================================================================
# MAGI OS — Phase 1: OS Strip & Harden Script
# Run on Raspberry Pi 4B after fresh RPi OS Lite (Bookworm 64-bit) install
# =============================================================================

set -euo pipefail
LOG="/var/log/magi_setup.log"
exec > >(tee -a "$LOG") 2>&1

echo "======================================================"
echo " MAGI OS Setup — Step 1: Strip & Harden"
echo " $(date)"
echo "======================================================"

# --- Update first ---
apt-get update -y
apt-get upgrade -y

# --- Remove unnecessary packages ---
echo "[MAGI] Removing unused packages..."
apt-get purge -y \
    triggerhappy \
    dphys-swapfile \
    avahi-daemon \
    bluetooth \
    bluez \
    bluez-firmware \
    pi-bluetooth \
    modemmanager \
    fake-hwclock \
    nfs-common \
    rfkill \
    v4l-utils \
    alsa-utils \
    alsa-base \
    pulseaudio \
    wpasupplicant \
    dhcpcd5 \
    pigpiod \
    libraspberrypi-doc \
    raspi-config \
    || true

apt-get autoremove -y
apt-get clean

# --- Disable / mask unused systemd services ---
echo "[MAGI] Masking unused systemd units..."
systemctl mask \
    bluetooth.service \
    avahi-daemon.service \
    ModemManager.service \
    wpa_supplicant.service \
    apt-daily.service \
    apt-daily-upgrade.service \
    apt-daily.timer \
    apt-daily-upgrade.timer \
    man-db.timer \
    logrotate.timer \
    motd-news.service \
    motd-news.timer \
    e2scrub_reap.service \
    phpsessionclean.timer \
    || true

systemctl disable \
    hciuart.service \
    rpi-eeprom-update.service \
    || true

# --- Disable IPv6 if not needed ---
echo "[MAGI] Disabling IPv6..."
cat >> /etc/sysctl.d/99-magi.conf << 'EOF'
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1

# VM / memory tuning for inference workloads
vm.swappiness = 10
vm.dirty_ratio = 5
vm.dirty_background_ratio = 2

# Network tuning
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
EOF
sysctl -p /etc/sysctl.d/99-magi.conf

# --- Set CPU governor to performance ---
echo "[MAGI] Setting CPU governor to performance..."
cat > /etc/systemd/system/cpu-governor.service << 'EOF'
[Unit]
Description=Set CPU governor to performance
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
systemctl enable cpu-governor.service

# --- Create MAGI directory structure ---
echo "[MAGI] Creating directory structure..."
mkdir -p /opt/magi/{models,src/{magi1,magi2,magi3,common},sensors,config,logs}
mkdir -p /etc/magi
chmod 755 /opt/magi

# --- tmpfs for IPC (RAM-backed, zero disk I/O) ---
echo "[MAGI] Configuring tmpfs for IPC..."
grep -q '/tmp/magi' /etc/fstab || \
    echo "tmpfs /tmp/magi tmpfs defaults,size=512M,mode=1777 0 0" >> /etc/fstab
mkdir -p /tmp/magi

# --- Rotate logs to prevent SD card fill ---
cat > /etc/logrotate.d/magi << 'EOF'
/opt/magi/logs/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
    size 10M
    maxsize 100M
}
EOF

echo ""
echo "======================================================"
echo " Step 1 COMPLETE — Reboot then run 02_install_deps.sh"
echo "======================================================"
