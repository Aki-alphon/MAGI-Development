"""
MAGI OS — Live Status Monitor
/opt/magi/src/status_monitor.py

Terminal dashboard: shows RAM, CPU per core, inference stats,
sensor data, and latest MAGI-3 decisions in real time.
Run: python3 /opt/magi/src/status_monitor.py
"""

import os, sys, time, yaml
sys.path.insert(0, "/opt/magi/src")

import psutil
import zmq
import msgpack

with open("/opt/magi/config/config.yaml") as f:
    CFG = yaml.safe_load(f)

IPC = CFG["ipc"]

def clear():
    os.system("clear")

def bar(val: float, width: int = 20, max_val: float = 100.0) -> str:
    filled = int((val / max_val) * width)
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)

def ram_info():
    m = psutil.virtual_memory()
    used_mb  = m.used  / 1024 / 1024
    total_mb = m.total / 1024 / 1024
    pct      = m.percent
    return used_mb, total_mb, pct

def cpu_info():
    return psutil.cpu_percent(percpu=True)

def temp_info():
    try:
        t = psutil.sensors_temperatures()
        cpu_temps = t.get("cpu_thermal", t.get("coretemp", []))
        if cpu_temps:
            return cpu_temps[0].current
    except Exception:
        pass
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return 0.0

def main():
    ctx  = zmq.Context()
    sub  = ctx.socket(zmq.SUB)
    sub.connect(IPC["magi3_decision"])
    sub.setsockopt(zmq.SUBSCRIBE, b"decision")
    sub.setsockopt(zmq.RCVTIMEO, 200)

    last_decision = {"action": "WAITING...", "reason": "-", "scene": "-",
                     "anomaly_score": 0.0, "tof_mm": -1}
    last_update   = 0.0

    print("MAGI OS Status Monitor — Press Ctrl+C to exit\n")
    time.sleep(1)

    while True:
        # Try to get latest decision
        try:
            parts = sub.recv_multipart()
            data  = msgpack.unpackb(parts[1], raw=False)
            last_decision = data
            last_update   = time.time()
        except zmq.Again:
            pass

        # Collect system stats
        used_mb, total_mb, ram_pct = ram_info()
        cores   = cpu_info()
        temp    = temp_info()
        age     = time.time() - last_update if last_update else 999

        clear()
        print("╔══════════════════════════════════════════════════════╗")
        print("║            MAGI OS — Live System Monitor             ║")
        print(f"║  {time.strftime('%Y-%m-%d %H:%M:%S')}                              ║")
        print("╠══════════════════════════════════════════════════════╣")

        # RAM
        print(f"║  RAM  [{bar(ram_pct)}] {ram_pct:5.1f}%  {used_mb:.0f}/{total_mb:.0f} MB")

        # CPU per core
        core_labels = ["OS+Sensors", "MAGI-1 Celebi", "MAGI-2 Gengar", "MAGI-3 Lugia"]
        for i, (pct, label) in enumerate(zip(cores[:4], core_labels)):
            print(f"║  C{i}  [{bar(pct)}] {pct:5.1f}%  {label}")

        # Temperature
        warn = " ⚠ HOT!" if temp > 70 else ""
        print(f"║  TEMP  {temp:.1f}°C{warn}")

        print("╠══════════════════════════════════════════════════════╣")
        print("║  MAGI-3 DECISION ENGINE")
        d = last_decision
        age_s = f"{age:.1f}s ago" if last_update else "no data yet"
        action = d.get("action", "?")
        color  = "🔴" if action == "EMERGENCY" else "🟡" if action in ("ALERT","TRACK") else "🟢"
        print(f"║  {color} Action : {action:<20} ({age_s})")
        print(f"║  📍 Scene  : {str(d.get('scene','-')):<30}")
        print(f"║  ⚠  Anomaly: {d.get('anomaly_score',0.0):.3f}")
        print(f"║  📏 ToF    : {d.get('tof_mm',-1)} mm")
        print(f"║  💬 Reason : {str(d.get('reason','-'))[:48]}")
        print("╚══════════════════════════════════════════════════════╝")
        print("  Ctrl+C to exit")

        time.sleep(1.0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
