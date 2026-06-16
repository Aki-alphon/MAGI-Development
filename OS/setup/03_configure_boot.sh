#!/bin/bash
# =============================================================================
# MAGI OS — Phase 3: Boot Configuration & CPU Core Isolation
# Configures /boot/firmware/config.txt and cmdline.txt for optimal inference
# =============================================================================

set -euo pipefail

CONFIG="/boot/firmware/config.txt"
CMDLINE="/boot/firmware/cmdline.txt"

echo "======================================================"
echo " MAGI OS Setup — Step 3: Boot Configuration"
echo " $(date)"
echo "======================================================"

# ─── config.txt ──────────────────────────────────────────
echo "[MAGI] Writing /boot/firmware/config.txt..."
cp "$CONFIG" "${CONFIG}.bak.$(date +%s)"

cat >> "$CONFIG" << 'EOF'

# ── MAGI Performance Settings ──────────────────────────
# Stable 1.8 GHz with slight over-voltage (requires heatsink + fan)
arm_freq=1800
over_voltage=2
force_turbo=0

# Disable display (headless operation)
# Comment out if HDMI display is needed
hdmi_blanking=2

# Camera — set to 1 if using Pi Camera
camera_auto_detect=0

# Disable audio (saves ~8 MB RAM)
dtparam=audio=off
EOF

echo "[MAGI] config.txt updated."

# ─── cmdline.txt ─────────────────────────────────────────
echo "[MAGI] Updating /boot/firmware/cmdline.txt..."
cp "$CMDLINE" "${CMDLINE}.bak.$(date +%s)"

# Read current cmdline (single line)
CURRENT=$(cat "$CMDLINE")

# Add CPU isolation and quiet boot params if not already present
NEW_PARAMS="quiet loglevel=3 isolcpus=1,2,3 rcu_nocbs=1,2,3 nohz_full=1,2,3 elevator=mq-deadline"
UPDATED="$CURRENT"

for PARAM in $NEW_PARAMS; do
    KEY=$(echo "$PARAM" | cut -d= -f1)
    if ! echo "$CURRENT" | grep -qw "$KEY"; then
        UPDATED="$UPDATED $PARAM"
    fi
done

echo "$UPDATED" > "$CMDLINE"
echo "[MAGI] cmdline.txt updated with CPU isolation (cores 1,2,3)."

# ─── Verify /etc/fstab tmpfs ─────────────────────────────
echo "[MAGI] Ensuring tmpfs mount for IPC..."
if ! grep -q '/tmp/magi' /etc/fstab; then
    echo "tmpfs /tmp/magi tmpfs defaults,size=512M,mode=1777 0 0" >> /etc/fstab
fi
mkdir -p /tmp/magi

# ─── Set hostname ────────────────────────────────────────
echo "[MAGI] Setting hostname to 'magi-node'..."
hostnamectl set-hostname magi-node
sed -i 's/raspberrypi/magi-node/g' /etc/hosts

# ─── Static IP (edit as needed) ──────────────────────────
echo "[MAGI] Configuring static IP..."
cat > /etc/dhcpcd.conf << 'EOF'
# MAGI Static IP Configuration
# Modify to match your network

interface eth0
static ip_address=192.168.1.100/24
static routers=192.168.1.1
static domain_name_servers=8.8.8.8 1.1.1.1

# Disable WLAN (we removed Wi-Fi overlay)
denyinterfaces wlan0
EOF

echo ""
echo "======================================================"
echo " Step 3 COMPLETE"
echo " Summary of changes:"
echo "  - CPU cores 1,2,3 isolated for MAGI inference"
echo "  - GPU mem = 64 MB (headless)"
echo "  - ARM freq locked @ 1800 MHz"
echo "  - Hostname = magi-node"
echo "  - Reboot to apply ALL changes"
echo "======================================================"
