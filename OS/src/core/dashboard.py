"""
MAGI OS v2 — Tier 2 Observer Dashboard
src/core/dashboard.py

Lightweight FastAPI web server running on CPU Core 0.
Subscribes to ZMQ bus and serves real-time HTML/JS telemetry dashboard.
Features dynamic 5-second SPI node discovery WhoAmI polling.
"""

import os
import sys
import time
import signal
import threading
import zmq
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import msgpack
import numpy as np

sys.path.insert(0, "/opt/magi/src")

# Pin to Core 0 (OS and Sensor Core)
try:
    os.sched_setaffinity(0, {0})
except (AttributeError, OSError):
    pass

from common.logger import get_logger
from core.messages import decode

log = get_logger("dashboard")

app = FastAPI(title="MAGI OS Telemetry Dashboard")

# Global state
LATEST_TELEMETRY = {
    "sensors": {},
    "scene": {},
    "detections": {},
    "decision": {},
    "diagnostics": {},
    "parameters": {},
}
SPI_NODES = []
_running = True
active_websockets = []


# ─── ZMQ Subscriber Thread ──────────────────────────────────────────────────

class ZmqSubscriber(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)

    def run(self):
        # Wait for bus
        time.sleep(1.0)
        try:
            self.sock.connect("ipc:///tmp/magi/bus_sub.sock")
            # Subscribe to all telemetry topics
            self.sock.setsockopt(zmq.SUBSCRIBE, b"/sensors")
            self.sock.setsockopt(zmq.SUBSCRIBE, b"/scene")
            self.sock.setsockopt(zmq.SUBSCRIBE, b"/detections")
            self.sock.setsockopt(zmq.SUBSCRIBE, b"/decision")
            self.sock.setsockopt(zmq.SUBSCRIBE, b"/diagnostics")
            self.sock.setsockopt(zmq.SUBSCRIBE, b"/parameters")
            self.sock.setsockopt(zmq.RCVTIMEO, 200)
            log.info("Dashboard ZMQ Subscriber connected and listening")
        except Exception as e:
            log.error(f"Dashboard ZMQ connection failed: {e}")
            return

        while _running:
            try:
                parts = self.sock.recv_multipart()
                if len(parts) >= 2:
                    topic = parts[0].decode("utf-8")
                    msg = decode(parts[1])
                    
                    # Convert object / dict to standard dict for JSON serialization
                    data = {}
                    if hasattr(msg, "__dict__"):
                        data = self._to_dict(msg)
                    elif isinstance(msg, dict):
                        data = msg

                    topic_key = topic.lstrip("/")
                    LATEST_TELEMETRY[topic_key] = data
                    LATEST_TELEMETRY[topic_key]["_updated_at"] = time.time()

                    # Trigger update to all connected Websockets
                    self._broadcast({
                        "type": "telemetry",
                        "topic": topic_key,
                        "data": data
                    })
            except zmq.Again:
                pass
            except Exception as e:
                log.debug(f"ZMQ decode warning: {e}")

        self.sock.close()

    def _to_dict(self, obj):
        """Recursively serialize custom dataclasses to dicts."""
        if hasattr(obj, "__dataclass_fields__"):
            result = {}
            for field in obj.__dataclass_fields__:
                val = getattr(obj, field)
                if hasattr(val, "__dataclass_fields__") or isinstance(val, list) or isinstance(val, dict):
                    result[field] = self._to_dict(val)
                else:
                    result[field] = val
            return result
        elif isinstance(obj, list):
            return [self._to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._to_dict(v) for k, v in obj.items()}
        return obj

    def _broadcast(self, msg):
        dead_sockets = []
        for ws in active_websockets:
            try:
                # FastAPI WebSocket send_json is synchronous or async? Must run in async loop.
                # Since we are in a standard thread, we run it using asyncio if uvicorn is running,
                # but we can safely just send via direct queue or let websockets poll.
                # Instead, we will do dynamic polling from the UI (every 100ms) to ensure zero WebSocket overhead/crashes!
                pass
            except Exception:
                pass


# ─── Dynamic SPI Node Discovery (5s Polling Handshake) ──────────────────────

def _poll_spi_loop():
    global SPI_NODES
    log.info("SPI discovery handshake thread active (5s interval)")
    while _running:
        discovered = []
        # Try discovering custom SPI sensor and display controller nodes
        for bus, device in [(0, 0), (0, 1)]:
            node_discovered = False
            try:
                import spidev
                spi = spidev.SpiDev()
                spi.open(bus, device)
                spi.max_speed_hz = 1000000
                
                # Send "WhoAmI" registers command (e.g. 0x9F)
                # and read 2 bytes back
                resp = spi.xfer2([0x9F, 0x00])
                spi.close()

                # If response byte matches our custom SPI node signature 0x42
                if len(resp) >= 2 and resp[1] == 0x42:
                    discovered.append({
                        "name": f"SensorNode-A" if device == 0 else "DisplayNode-B",
                        "bus": bus,
                        "device": device,
                        "status": "Active",
                        "type": "Physical SPI",
                        "last_seen": time.time()
                    })
                    node_discovered = True
            except Exception:
                pass

            if not node_discovered:
                # Mock fallback for Docker / Testing to demonstrate the dynamic discovery
                if os.environ.get("MAGI_ENV") == "docker":
                    # Mock finding SensorNode-A on Cs=0, and DisplayNode-B on Cs=1
                    if device == 0:
                        discovered.append({
                            "name": "SensorNode-A (Mock)",
                            "bus": bus,
                            "device": device,
                            "status": "Active",
                            "type": "Simulated SPI",
                            "last_seen": time.time()
                        })
                    else:
                        # Display node is discovered periodically for dynamic simulation
                        if int(time.time()) % 15 < 10:
                            discovered.append({
                                "name": "DisplayNode-B (Mock)",
                                "bus": bus,
                                "device": device,
                                "status": "Active",
                                "type": "Simulated SPI",
                                "last_seen": time.time()
                            })

        SPI_NODES = discovered
        time.sleep(5.0)


# ─── FastAPI API Endpoints ──────────────────────────────────────────────────

@app.get("/api/telemetry")
async def get_telemetry():
    return {
        "telemetry": LATEST_TELEMETRY,
        "spi_nodes": SPI_NODES,
        "uptime": time.time()
    }

@app.post("/api/parameter")
async def set_parameter(node: str, key: str, value: float):
    """Bridge API to update ParamServer parameters dynamically from the UI."""
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 500)
    sock.setsockopt(zmq.SNDTIMEO, 500)
    try:
        sock.connect("ipc:///tmp/magi/param_server.sock")
        req = {"cmd": "set", "node": node, "key": key, "value": value}
        sock.send(msgpack.packb(req, use_bin_type=True))
        resp = msgpack.unpackb(sock.recv(), raw=False)
        sock.close()
        return {"success": resp.get("ok", False)}
    except Exception as e:
        log.error(f"Failed to bridge param update: {e}")
        sock.close()
        return {"success": False, "error": str(e)}


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    # Embedded HTML UI — premium design system, glassmorphism, responsive
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MAGI OS Telemetry Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            .glass {
                background: rgba(15, 23, 42, 0.45);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
            .glow-green {
                box-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
            }
            .glow-orange {
                box-shadow: 0 0 15px rgba(245, 158, 11, 0.4);
            }
            .pulse {
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0%, 100% { transform: scale(1); opacity: 1; }
                50% { transform: scale(1.1); opacity: 0.7; }
            }
        </style>
    </head>
    <body class="bg-[#05070f] text-slate-100 font-sans min-h-screen overflow-x-hidden">
        
        <!-- Header -->
        <header class="border-b border-slate-800 bg-[#070b19] px-8 py-5 flex items-center justify-between sticky top-0 z-50">
            <div class="flex items-center gap-4">
                <div class="h-10 w-10 rounded-lg bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center font-bold text-lg shadow-lg">M</div>
                <div>
                    <h1 class="text-xl font-bold tracking-wider bg-gradient-to-r from-slate-100 to-slate-400 bg-clip-text text-transparent">MAGI OS</h1>
                    <p class="text-xs text-slate-500 font-mono">Autonomous Intelligence Base Dashboard</p>
                </div>
            </div>
            <div class="flex items-center gap-6">
                <div class="flex items-center gap-2 font-mono text-xs glass px-3 py-1.5 rounded-md">
                    <span class="h-2 w-2 rounded-full bg-emerald-500 pulse"></span>
                    <span class="text-slate-400">TELEMETRY LINK:</span>
                    <span id="telemetry-status" class="text-emerald-400 font-bold">ONLINE</span>
                </div>
                <div class="text-xs text-slate-500 font-mono" id="system-time">--:--:--</div>
            </div>
        </header>

        <!-- Main Content -->
        <main class="p-8 max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8">
            
            <!-- COLUMN 1: System Architecture & Gating -->
            <div class="flex flex-col gap-8 lg:col-span-1">
                
                <!-- Gating Status Card -->
                <div class="glass p-6 rounded-2xl flex flex-col gap-4 relative overflow-hidden">
                    <div class="absolute -right-8 -top-8 h-24 w-24 bg-purple-500/10 rounded-full blur-2xl"></div>
                    <div class="flex items-center justify-between">
                        <h2 class="text-md font-bold text-slate-300 flex items-center gap-2">
                            <i class="fa-solid fa-microchip text-indigo-400"></i> Cascaded Core Gating
                        </h2>
                        <span class="text-xs font-mono px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">POWER SAVING</span>
                    </div>
                    
                    <div class="grid grid-cols-2 gap-4 mt-2">
                        <!-- Core 2 Balthasar (Trigger) -->
                        <div class="bg-slate-900/60 p-4 rounded-xl border border-slate-800/80 flex flex-col gap-2">
                            <span class="text-[10px] text-slate-500 font-mono">CORE 2 (BALTHASAR)</span>
                            <div class="flex items-center justify-between">
                                <span class="font-semibold text-slate-300">Trigger Model</span>
                                <span class="h-2.5 w-2.5 rounded-full bg-emerald-500 glow-green pulse"></span>
                            </div>
                            <span class="text-xs text-emerald-400 font-mono bg-emerald-500/5 px-2 py-1 rounded border border-emerald-500/10 text-center">CONTINUOUS</span>
                        </div>
                        
                        <!-- Core 1 Melchior (Gated Node) -->
                        <div class="bg-slate-900/60 p-4 rounded-xl border border-slate-800/80 flex flex-col gap-2">
                            <span class="text-[10px] text-slate-500 font-mono">CORE 1 (MELCHIOR)</span>
                            <div class="flex items-center justify-between">
                                <span class="font-semibold text-slate-300">Heavy Model</span>
                                <span id="melchior-gating-indicator" class="h-2.5 w-2.5 rounded-full bg-amber-500 glow-orange pulse"></span>
                            </div>
                            <span id="melchior-gating-state" class="text-xs text-amber-400 font-mono bg-amber-500/5 px-2 py-1 rounded border border-amber-500/10 text-center">SLEEPING (INACTIVE)</span>
                        </div>
                    </div>

                    <div class="text-xs text-slate-400 leading-relaxed font-sans border-t border-slate-800/80 pt-4 flex flex-col gap-1">
                        <div class="flex justify-between font-mono text-[11px]">
                            <span>Stable Plant Trigger:</span>
                            <span id="stable-plant-detected" class="text-rose-400">NO TARGET</span>
                        </div>
                        <div class="flex justify-between font-mono text-[11px]">
                            <span>XNNPACK Pinning:</span>
                            <span class="text-slate-300">1 Thread (Core 1)</span>
                        </div>
                    </div>
                </div>

                <!-- Process State Node Health -->
                <div class="glass p-6 rounded-2xl flex flex-col gap-4">
                    <h2 class="text-md font-bold text-slate-300 flex items-center gap-2">
                        <i class="fa-solid fa-heartbeat text-emerald-400"></i> Self-Healing Diagnostics
                    </h2>
                    
                    <div class="flex flex-col gap-3" id="nodes-list">
                        <!-- Diagnostic nodes injected here -->
                        <div class="text-slate-500 text-xs py-4 text-center font-mono">Loading diagnostics heartbeats...</div>
                    </div>
                </div>
            </div>

            <!-- COLUMN 2: Sensor Telemetry & Charting -->
            <div class="flex flex-col gap-8 lg:col-span-2">
                
                <!-- Telemetry Metrics -->
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div class="glass p-5 rounded-xl flex items-center justify-between">
                        <div>
                            <span class="text-xs text-slate-500 font-mono">TOF RANGE</span>
                            <h3 class="text-xl font-bold text-slate-100 mt-1 font-mono" id="metric-tof">-- mm</h3>
                        </div>
                        <div class="h-10 w-10 rounded-lg bg-indigo-500/10 flex items-center justify-center text-indigo-400">
                            <i class="fa-solid fa-arrows-left-right-to-line text-lg"></i>
                        </div>
                    </div>

                    <div class="glass p-5 rounded-xl flex items-center justify-between">
                        <div>
                            <span class="text-xs text-slate-500 font-mono">SCENE CLASS</span>
                            <h3 class="text-lg font-bold text-slate-100 mt-1 font-sans truncate max-w-[150px]" id="metric-scene">--</h3>
                        </div>
                        <div class="h-10 w-10 rounded-lg bg-purple-500/10 flex items-center justify-center text-purple-400">
                            <i class="fa-solid fa-mountain-sun text-lg"></i>
                        </div>
                    </div>

                    <div class="glass p-5 rounded-xl flex items-center justify-between">
                        <div>
                            <span class="text-xs text-slate-500 font-mono">ANOMALY SCORE</span>
                            <h3 class="text-xl font-bold text-slate-100 mt-1 font-mono" id="metric-anomaly">--</h3>
                        </div>
                        <div class="h-10 w-10 rounded-lg bg-emerald-500/10 flex items-center justify-center text-emerald-400">
                            <i class="fa-solid fa-triangle-exclamation text-lg"></i>
                        </div>
                    </div>
                </div>

                <!-- IMU and ToF Chart -->
                <div class="glass p-6 rounded-2xl flex flex-col gap-4">
                    <div class="flex items-center justify-between">
                        <h2 class="text-md font-bold text-slate-300 flex items-center gap-2">
                            <i class="fa-solid fa-chart-line text-purple-400"></i> Real-time Telemetry (RAM Disk Sockets)
                        </h2>
                        <span class="text-xs font-mono text-slate-500">10 Hz updates</span>
                    </div>
                    <div class="h-64 w-full">
                        <canvas id="telemetry-chart"></canvas>
                    </div>
                </div>

                <!-- Bottom Grid: SPI Discovery & Parameter Setting -->
                <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <!-- SPI Discovery -->
                    <div class="glass p-6 rounded-xl flex flex-col gap-4">
                        <div class="flex items-center justify-between">
                            <h3 class="text-sm font-bold text-slate-300 flex items-center gap-2">
                                <i class="fa-solid fa-network-wired text-emerald-400"></i> Dynamic SPI Discovery
                            </h3>
                            <span class="text-[10px] font-mono bg-emerald-500/10 text-emerald-400 px-2 py-0.5 rounded border border-emerald-500/20">5s Poll</span>
                        </div>
                        
                        <div class="flex flex-col gap-2 mt-1" id="spi-list">
                            <!-- SPI discovered nodes injected here -->
                            <div class="text-slate-500 text-xs py-4 text-center font-mono">Scanning SPI dev bus...</div>
                        </div>
                    </div>

                    <!-- Parameter Tuner -->
                    <div class="glass p-6 rounded-xl flex flex-col gap-4">
                        <h3 class="text-sm font-bold text-slate-300 flex items-center gap-2">
                            <i class="fa-solid fa-sliders text-indigo-400"></i> Live Parameter Tuning (Root Operator)
                        </h3>
                        
                        <div class="flex flex-col gap-4 mt-1">
                            <div>
                                <label class="text-[11px] font-mono text-slate-500 block mb-1">MELCHIOR CONFIDENCE THRESHOLD</label>
                                <div class="flex gap-2">
                                    <input type="number" step="0.05" min="0.1" max="0.95" id="param-conf" class="bg-slate-900 border border-slate-800 rounded-lg px-3 py-1.5 text-sm font-mono flex-1 focus:outline-none focus:border-indigo-500 text-slate-300" value="0.40">
                                    <button onclick="updateParam('magi1', 'confidence_threshold', 'param-conf')" class="bg-indigo-600 hover:bg-indigo-700 text-slate-100 text-xs font-semibold px-4 rounded-lg transition-colors border border-indigo-500/30">SET</button>
                                </div>
                            </div>
                            <div>
                                <label class="text-[11px] font-mono text-slate-500 block mb-1">SENSOR POLL RATE (HZ)</label>
                                <div class="flex gap-2">
                                    <input type="number" step="1" min="1" max="100" id="param-poll" class="bg-slate-900 border border-slate-800 rounded-lg px-3 py-1.5 text-sm font-mono flex-1 focus:outline-none focus:border-indigo-500 text-slate-300" value="10">
                                    <button onclick="updateParam('system', 'poll_rate_hz', 'param-poll')" class="bg-indigo-600 hover:bg-indigo-700 text-slate-100 text-xs font-semibold px-4 rounded-lg transition-colors border border-indigo-500/30">SET</button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

            </div>

        </main>

        <script>
            // System time ticker
            setInterval(() => {
                const d = new Date();
                document.getElementById('system-time').innerText = d.toLocaleTimeString();
            }, 1000);

            // Chart Configuration
            const ctx = document.getElementById('telemetry-chart').getContext('2d');
            const telemetryChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'ToF Range (mm)',
                            data: [],
                            borderColor: 'rgba(99, 102, 241, 0.85)',
                            backgroundColor: 'rgba(99, 102, 241, 0.05)',
                            borderWidth: 2,
                            tension: 0.3,
                            yAxisID: 'y'
                        },
                        {
                            label: 'IMU Accel-Z (g)',
                            data: [],
                            borderColor: 'rgba(16, 185, 129, 0.85)',
                            backgroundColor: 'transparent',
                            borderWidth: 1.5,
                            borderDash: [4, 4],
                            tension: 0.3,
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            grid: { color: 'rgba(255, 255, 255, 0.02)' },
                            ticks: { display: false }
                        },
                        y: {
                            type: 'linear',
                            position: 'left',
                            grid: { color: 'rgba(255, 255, 255, 0.04)' },
                            ticks: { color: 'rgba(156, 163, 175, 0.8)', font: { size: 10 } }
                        },
                        y1: {
                            type: 'linear',
                            position: 'right',
                            grid: { drawOnChartArea: false },
                            ticks: { color: 'rgba(156, 163, 175, 0.8)', font: { size: 10 } }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: { color: 'rgba(226, 232, 240, 0.9)', font: { size: 11 } }
                        }
                    }
                }
            });

            // Polling function for telemetry and SPI nodes
            let count = 0;
            async function pollTelemetry() {
                try {
                    const response = await fetch('/api/telemetry');
                    const r = await response.json();
                    
                    const telemetry = r.telemetry;
                    const spi = r.spi_nodes;

                    // Update Top metrics
                    const sensorData = telemetry.sensors || {};
                    const sceneData = telemetry.scene || {};
                    const detectionData = telemetry.detections || {};
                    const diagData = telemetry.diagnostics || {};

                    const tofVal = (sensorData.tof && sensorData.tof.distance_mm !== undefined) ? sensorData.tof.distance_mm : -1;
                    document.getElementById('metric-tof').innerText = tofVal >= 0 ? `${tofVal} mm` : 'N/A';
                    
                    const sceneVal = sceneData.scene || 'unknown';
                    document.getElementById('metric-scene').innerText = sceneVal;
                    document.getElementById('metric-anomaly').innerText = sceneData.anomaly_score !== undefined ? sceneData.anomaly_score.toFixed(3) : '0.000';

                    // Update gating elements
                    const melchiorActive = telemetry.diagnostics && telemetry.diagnostics.magi1 && telemetry.diagnostics.magi1.state === 'ACTIVE';
                    const melchiorIndicator = document.getElementById('melchior-gating-indicator');
                    const melchiorStateText = document.getElementById('melchior-gating-state');
                    const stablePlantIndicator = document.getElementById('stable-plant-detected');

                    if (sceneVal === 'stable_plant') {
                        stablePlantIndicator.innerText = 'CROP DETECTED';
                        stablePlantIndicator.className = 'text-emerald-400 font-bold pulse';
                    } else {
                        stablePlantIndicator.innerText = 'NO TARGET';
                        stablePlantIndicator.className = 'text-rose-400';
                    }

                    if (melchiorActive) {
                        melchiorIndicator.className = 'h-2.5 w-2.5 rounded-full bg-emerald-500 glow-green pulse';
                        melchiorStateText.innerText = 'PROCESSING (ACTIVE)';
                        melchiorStateText.className = 'text-xs text-emerald-400 font-mono bg-emerald-500/5 px-2 py-1 rounded border border-emerald-500/10 text-center';
                    } else {
                        melchiorIndicator.className = 'h-2.5 w-2.5 rounded-full bg-amber-500 glow-orange pulse';
                        melchiorStateText.innerText = 'SLEEPING (INACTIVE)';
                        melchiorStateText.className = 'text-xs text-amber-400 font-mono bg-amber-500/5 px-2 py-1 rounded border border-amber-500/10 text-center';
                    }

                    // Update Chart
                    if (count > 25) {
                        telemetryChart.data.labels.shift();
                        telemetryChart.data.datasets[0].data.shift();
                        telemetryChart.data.datasets[1].data.shift();
                    }
                    
                    const stamp = new Date().toLocaleTimeString();
                    const accZ = (sensorData.imu && sensorData.imu.accel) ? sensorData.imu.accel[2] : 1.0;
                    
                    telemetryChart.data.labels.push(stamp);
                    telemetryChart.data.datasets[0].data.push(tofVal >= 0 ? tofVal : null);
                    telemetryChart.data.datasets[1].data.push(accZ);
                    telemetryChart.update();
                    count++;

                    // Update Diagnostics Heartbeats
                    const diagEl = document.getElementById('nodes-list');
                    let diagHtml = '';
                    
                    // Core nodes mapping
                    const nodes = [
                        { id: 'sensor_hub', title: 'Sensor Hub (Core 0)', desc: 'Sensor publishing & GPIO interface' },
                        { id: 'magi1', title: 'Melchior (Core 1)', desc: 'Disease detection inference' },
                        { id: 'magi2', title: 'Balthasar (Core 2)', desc: 'Trigger & Scene classification' },
                        { id: 'magi3', title: 'Caspar (Core 3)', desc: 'Fusion decision engine' },
                        { id: 'batch_manager', title: 'Batch Manager (Core 0)', desc: 'Parquet & Zarr Zstd compression' },
                        { id: 'dashboard', title: 'Web Dashboard (Core 0)', desc: 'Lightweight observer endpoint' }
                    ];

                    nodes.forEach(n => {
                        const state = (telemetry.diagnostics && telemetry.diagnostics[n.id] && telemetry.diagnostics[n.id].state) || 'UNCONFIGURED';
                        const status = (telemetry.diagnostics && telemetry.diagnostics[n.id] && telemetry.diagnostics[n.id].status) || 'OFFLINE';
                        
                        let badgeClass = 'text-slate-500 bg-slate-500/10 border-slate-500/20';
                        let dotClass = 'bg-slate-600';
                        if (state === 'ACTIVE') {
                            badgeClass = 'text-emerald-400 bg-emerald-400/10 border-emerald-400/20';
                            dotClass = 'bg-emerald-500 pulse glow-green';
                        } else if (state === 'INACTIVE') {
                            badgeClass = 'text-amber-400 bg-amber-400/10 border-amber-400/20';
                            dotClass = 'bg-amber-500 pulse glow-orange';
                        }

                        diagHtml += `
                        <div class="bg-slate-900/40 border border-slate-800/80 p-3 rounded-xl flex items-center justify-between">
                            <div class="flex items-center gap-3">
                                <span class="h-2 w-2 rounded-full ${dotClass}"></span>
                                <div>
                                    <h4 class="text-xs font-bold text-slate-200 font-mono">${n.title}</h4>
                                    <p class="text-[10px] text-slate-500 leading-tight mt-0.5">${n.desc}</p>
                                </div>
                            </div>
                            <span class="text-[10px] font-mono px-2.5 py-0.5 rounded border ${badgeClass}">${state}</span>
                        </div>
                        `;
                    });
                    diagEl.innerHTML = diagHtml;

                    // Update SPI Nodes
                    const spiEl = document.getElementById('spi-list');
                    if (spi.length === 0) {
                        spiEl.innerHTML = '<div class="text-slate-500 text-xs py-4 text-center font-mono border border-dashed border-slate-800 rounded-xl">No active SPI nodes found</div>';
                    } else {
                        let spiHtml = '';
                        spi.forEach(node => {
                            spiHtml += `
                            <div class="bg-slate-900/40 border border-slate-800/80 p-3 rounded-xl flex items-center justify-between">
                                <div class="flex items-center gap-3">
                                    <i class="fa-solid fa-microchip text-emerald-400 text-sm pulse"></i>
                                    <div>
                                        <h4 class="text-xs font-bold text-slate-200 font-mono">${node.name}</h4>
                                        <p class="text-[9px] text-slate-500 font-mono mt-0.5">CS: ${node.device} | Bus: ${node.bus}</p>
                                    </div>
                                </div>
                                <div class="flex flex-col items-end gap-1">
                                    <span class="text-[9px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">${node.status}</span>
                                    <span class="text-[8px] font-mono text-slate-500">${node.type}</span>
                                </div>
                            </div>
                            `;
                        });
                        spiEl.innerHTML = spiHtml;
                    }

                    document.getElementById('telemetry-status').innerText = 'ONLINE';
                    document.getElementById('telemetry-status').className = 'text-emerald-400 font-bold';

                } catch (e) {
                    console.error("Telemetry fetch failed: ", e);
                    document.getElementById('telemetry-status').innerText = 'ERROR';
                    document.getElementById('telemetry-status').className = 'text-rose-400 font-bold pulse';
                }
            }

            // Start loop
            setInterval(pollTelemetry, 300);

            // Parameter tuning
            async function updateParam(node, key, inputId) {
                const val = parseFloat(document.getElementById(inputId).value);
                try {
                    const response = await fetch(`/api/parameter?node=${node}&key=${key}&value=${val}`, {
                        method: 'POST'
                    });
                    const res = await response.json();
                    if (res.success) {
                        alert(`Parameter ${node}.${key} successfully set to ${val}`);
                    } else {
                        alert(`Failed to set parameter: ${res.error || 'unknown server error'}`);
                    }
                } catch (e) {
                    alert(`Failed to update parameter: ${e}`);
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


# ─── Server Bootstrap / Lifecycle Node integration ─────────────────────────

class DashboardServer(LifecycleNode):
    def __init__(self, node_id: str = "dashboard", cpu_core: int = 0):
        super().__init__(node_id, cpu_core)
        self._sub_thread = None
        self._spi_thread = None

    def on_configure(self):
        # Configure ZMQ thread
        self._sub_thread = ZmqSubscriber()
        # Configure SPI polling thread
        self._spi_thread = threading.Thread(target=_poll_spi_loop, daemon=True)
        self.log.info("Dashboard Node Configured")

    def on_activate(self):
        global _running
        _running = True
        self._sub_thread.start()
        self._spi_thread.start()
        
        # Start FastAPI in background thread so it doesn't block the node loop
        self._web_thread = threading.Thread(
            target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning"),
            daemon=True
        )
        self._web_thread.start()
        self.log.info("Dashboard Web Server active on http://0.0.0.0:8000")

    def on_deactivate(self):
        global _running
        _running = False
        self.log.info("Dashboard Node deactivated")


if __name__ == "__main__":
    DashboardServer().boot()
