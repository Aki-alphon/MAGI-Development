# MAGI Quadruped Locomotion Simulator — Complete Guide
## Version 3.0 · RPi 4B · 12-DOF Quadruped

---

## 🚀 Quick Start

1. Open a terminal inside `/home/aki/Downloads/MAGI/simulation/`
2. Run: `python3 -m http.server 8765`
3. Open browser → **http://localhost:8765**
4. The 3D robot should appear within 1–2 seconds

> **Viewport blank?** See the [Troubleshooting](#-troubleshooting) section at the bottom.

---

## 🗺️ Navigation Overview

The sidebar has **5 tabs**:

| Tab | Icon | Purpose |
|-----|------|---------|
| **Locomotion Sim** | 🤖 | Live 3D robot walking simulation |
| **RL Gait Training** | 🧠 | Genetic algorithm + neural network training |
| **Decision Fusion** | ⚡ | MAGI-3 Lugia sensor decision engine |
| **Gait Analysis** | 📊 | Metrics, foot path, phase diagrams |
| **Kinematics Math** | 📐 | IK equations + interactive solver |

The **Phase Diagram** in the sidebar updates live on every tab, showing which legs are swinging (orange) vs. grounded (dark green).

---

## 🤖 Tab 1 — Locomotion Simulator

This is the main 3D viewport showing the quadruped walking in real time.

### Controls

#### Locomotion Mode (5 Gaits)

| Button | Gait | Description |
|--------|------|-------------|
| **Static Crawl** | Crawl | One leg swings at a time. Most stable. Duty cycle ≈ 75%. Body shifts laterally before each lift for CoM safety. |
| **Trot** | Trot | Diagonal pairs (FR+BL, FL+BR) swing together. 2× faster than crawl. |
| **Gallop** | Gallop | Leading/trailing forelimbs stagger, then rear burst propulsion. Has a **flight phase** (all feet off ground). Fastest. |
| **Bound** | Bound | Front pair then rear pair swing simultaneously. High energy, maximum stride. |
| **Stand/Hold** | Stand | All feet locked in neutral positions. Body pitch/roll still controllable. |

#### Gait Parameters

| Slider | Range | Effect |
|--------|-------|--------|
| **Gait Frequency** | 0.2 – 4.0 Hz | How many complete cycles per second. Higher = faster walking speed. |
| **Step Stride Length** | 10 – 140 mm | How far each foot reaches forward per step. |
| **Step Height (Clearance)** | 5 – 70 mm | How high the foot lifts during swing phase. Increase for rough terrain. |
| **Body Height** | 90 – 210 mm | Chassis height above ground. Higher = better clearance but less stability. |
| **Body Tilt (Pitch)** | ±20° | Lean body forward or backward. |
| **Body Roll** | ±20° | Tilt left or right. |
| **Terrain Roughness** | 0 – 25 mm | Activates the procedural terrain mesh. 0 = flat floor. |

#### Checkboxes

- **Show Support Polygon & CoM** — Draws the green/red quadrilateral connecting all 4 feet and an orange CoM sphere. **Green** = CoM is inside the support polygon (stable). **Red** = CoM outside (risk of tipping).

### Telemetry Panel (Right)

Shows live servo angles for all 12 joints. Leg boxes **glow orange** when that leg is in swing phase.

**IMU readings** simulate a gyroscope: Pitch, Roll, and vertical acceleration (1g nominal, oscillates when walking).

---

## 🧠 Tab 2 — RL Gait Training

Uses a **Genetic Algorithm** to evolve locomotion policies. Optionally switches to a **Neural Network policy** (NNPolicy) mode.

### How Training Works

```
┌──────────────────────────────────────────────────────────┐
│  1. Create population of N agents (random genes/weights) │
│  2. Evaluate each agent for 5 simulated seconds          │
│  3. Score each agent with fitness function               │
│  4. Keep top 20% (elitism) → crossover → mutate         │
│  5. Repeat from step 2                                   │
└──────────────────────────────────────────────────────────┘
```

### Controls

| Control | Description |
|---------|-------------|
| **1× / 5× / 25× / 100× Turbo** | Simulation speed. 100× runs 100 physics steps per visual frame. |
| **Population Size** | Number of agents per generation (10–100). Larger = more diversity but slower. |
| **Mutation Rate** | % chance each gene mutates per generation. 15% is a good default. |
| **Terrain Friction** | Soil (1.0), Sand (0.6), Rock (1.4). Changes how much grip the feet have. |
| **Direct Gene / NNPolicy** | Toggle between 12-gene direct encoding and an 8→16→12 neural network. |
| **Start RL Training** | Begins the evolution loop. Click again to pause. |
| **Reset** | Clears all history and restarts from generation 0. |
| **📋 Export Best Genome** | Copies the best agent's genome as JSON to clipboard. |

### Policy Modes

#### Direct Gene Mode (default)
Each agent has exactly **12 genes** (numbers between −1 and 1):

| Gene Index | Controls |
|-----------|---------|
| 0–3 | Phase offset for each leg (FR, FL, BR, BL) — values 0.0–1.0 |
| 4–7 | Stride length multiplier per leg — ×0.5 to ×1.5 |
| 8 | Body pitch sway amplitude |
| 9 | Body roll sway amplitude |
| 10 | Step duration multiplier |
| 11 | Body height bias |

#### NNPolicy Mode (advanced)
An **8-input, 16-hidden, 12-output** feedforward network:
- **Inputs**: sin/cos of phase (×2), body pitch, body roll, body height, terrain friction
- **Hidden**: 16 ReLU neurons
- **Outputs**: same 12 locomotion parameters as gene mode
- **Evolved**: All ~300 network weights are the "genes" evolved by the GA

### Fitness Function (4 Components)

```
Fitness = ∑ over time of:
  + velocity_score × forward_velocity
  − 3.5 × (|pitch| + |roll|)        ← stability penalty
  − 0.08 × Σ(Δjoint²)               ← energy penalty
  + 5.0 × symmetry_score             ← L/R balance reward
```

- **Velocity score**: Based on how well the phase offsets match the ideal crawl sequence (FR→FL→BL→BR each offset by 0.25)
- **Stability**: Penalises excessive body rocking
- **Energy**: Penalises fast/jerky joint movements
- **Symmetry**: Rewards balanced left/right phase offsets

### Curriculum Learning

The trainer **automatically increases difficulty** as the population improves:

| Level | Condition | Effect |
|-------|-----------|--------|
| LVL 1 | Start | Base stride range |
| LVL 2–5 | Avg fitness > level × 50 | Stride range increases by 10% per level |

Watch the **LVL badge** in the Training Controls header flash orange when difficulty advances.

### Charts & Visualisations

- **Reward Curve**: Max (orange), Average (green), Min (red dashed) fitness per generation
- **Gene Chromosome**: 12-bar chart of the best agent's genome. Positive genes = warm, Negative = red. Updates every generation.
- **Agent Progress Bar**: Shows which agent (1–N) is currently being evaluated
- **Live Fitness Breakdown**: Real-time velocity, stability, energy, symmetry scores for the current agent

---

## ⚡ Tab 3 — Decision Fusion (Lugia / MAGI-3)

Simulates the **high-level decision engine** that overrides walking gait based on sensor inputs.

### Scenarios (preset injectors)

| Scenario | What it sets | Decision |
|----------|-------------|---------|
| 🌾 Clear Path | ToF: 600mm, Anomaly: 0.12 | IDLE — normal walk |
| 🧱 Close Obstacle | ToF: 150mm | **EMERGENCY** — body drops and locks |
| 👤 Intruder | Person detected in restricted zone | **EMERGENCY** — halt |
| ⚠️ High Anomaly | Anomaly score: 0.85 | **ALERT** — slow cautious crawl |
| 🎯 Weed Target | Weed detected at 89% confidence | **TRACK** — lean forward to inspect |

### Manual Sensor Overrides

- **Front ToF Distance**: Drag slider to set the time-of-flight sensor distance. Below 200mm = emergency.
- **Scene Anomaly Score**: 0.0 = normal, 1.0 = extreme. Above 0.70 = cautious crawl.

### Priority Rule Engine (Lugia Logic)

```
Priority 10: ToF < 200mm          → EMERGENCY (halt + body drop)
Priority 9:  GPIO hardware trigger → EMERGENCY
Priority 8:  Person in restricted  → EMERGENCY
Priority 6:  Anomaly > 0.70        → ALERT (slow crawl)
Priority 5:  Scene flag emergency  → ANALYZE
Priority 4:  Detections present    → TRACK (lean forward)
Default 0:   No events             → IDLE / forward walk
```

The active rule **highlights in green** (or red for emergencies) in the Lugia node card.

### Locomotion Loopback

When Lugia fires a decision, it **overrides the gait controller**:

| Decision | Gait effect |
|---------|------------|
| EMERGENCY | Force STAND, body drops to -60mm (locks structure) |
| ALERT | Force CRAWL at 0.6Hz frequency, small steps |
| TRACK | CRAWL at normal speed, pitch +8° (lean camera forward) |
| IDLE | Restore slider values, normal walking |

The **Lugia Terminal Console** logs every state change with timestamp.

---

## 📊 Tab 4 — Gait Analysis

Real-time metrics and visualisation tools for analysing locomotion quality.

### Metric Cards

| Metric | What it means |
|--------|-------------|
| **Gait Type** | Current active gait |
| **Frequency** | Current Hz setting |
| **Stride** | Current stride length |
| **Step Height** | Current clearance |
| **Stride Efficiency** | `frequency × stride` — proxy for forward speed in mm/s |
| **Cumul. Energy** | Accumulated joint torque proxy over session time |

### Stride Efficiency & Energy Chart

Dual-axis live time-series chart:
- **Left axis (green)**: Stride efficiency (mm/s) — higher is more productive
- **Right axis (orange)**: Cumulative energy — lower is more efficient locomotion

### Phase Timing Diagram (Large)

Shows the last 120 frames of swing/stance state per leg:
- **Orange blocks** = leg in swing (foot in air)
- **Dark green blocks** = leg in stance (foot on ground)
- **Vertical dashed line** = current gait phase position
- **Coloured dot (right)** = real-time swing indicator

### Foot Path Plotter (Top View)

A top-down X–Z plane canvas showing the trajectory each foot traces:
- **Healthy gait** = smooth oval/ellipse traces
- **Flat/straight traces** = foot not lifting (step height too low)
- **Asymmetric traces** = potential yaw drift
- **Larger ovals** = bigger stride
- Colours: 🟠 FR · 🟢 FL · 🔵 BR · 🔴 BL

### Gait Comparison Table

| Gait | Duty % | Flight Phase | Cost of Transport |
|------|--------|-------------|-------------------|
| Crawl | 75% | No | High (very stable, slow) |
| Trot | 60% | No | Medium |
| Gallop | 40% | Yes | Low (fast, dynamic) |
| Bound | 35% | Yes | Low (max speed) |

**Duty %** = fraction of cycle each foot is on the ground.

---

## 📐 Tab 5 — Kinematics Math

Interactive explanation and debugger of the **Inverse Kinematics (IK) solver**.

### What is IK?

Given a **target foot position** (x, y, z) relative to the shoulder joint, IK calculates the exact joint angles (coxa, femur, tibia) needed to place the foot there.

### The 3-DOF Leg Architecture

```
 Shoulder pivot
      │
   [Coxa]   ← θc — horizontal rotation
      │
   [Femur]  ← θf — vertical swing (hip)
      │
   [Tibia]  ← θt — knee bend
      │
   [Foot tip]
```

**Link lengths**: Coxa = 30mm · Femur = 90mm · Tibia = 90mm

### IK Equations (Summary)

```
θc = atan2(y, x)

r = √(x² + y²)
Ld = √((r − 30)² + z²)          ← reach in sagittal plane

θt = π − arccos[(90² + 90² − Ld²) / (2×90×90)]   ← law of cosines

α = atan2(z, r−30)
β = arccos[(90² + Ld² − 90²) / (2×90×Ld)]
θf = α + β
```

### Interactive Debugger

Drag the 3 sliders to set a foot target position and see:
- Calculated coxa, femur, tibia angles
- **Sagittal plane diagram** — live 2D drawing of the leg geometry (right panel)
  - Cyan = Coxa link
  - Orange = Femur link
  - White = Tibia link
  - Green dot = Foot tip

---

## 🔄 The Bézier Foot Trajectory Algorithm

This is the core innovation in v3.0. Instead of linear interpolation for foot paths, each swing phase uses a **cubic Bézier arc**:

```
Foot_X(t) = Bézier(t, startX, startX+0.3×stride, startX+0.7×stride, endX)
Foot_Y(t) = Bézier(t, 0, 0.1×H, 1.1×H, 0)   ← height arc
```

Where `t ∈ [0,1]` is the swing phase normalised progress.

**Why Bézier?**
- Slow lift → fast peak → rapid plant (natural biomechanics)
- No foot scraping (Y never goes below ground)
- Smooth servo motion (bounded derivative)
- Control points tunable per terrain type

---

## 🛠️ Troubleshooting

### 3D Viewport is Blank / Black

The Three.js renderer initialises by polling the container size. If the viewport card has zero dimensions at startup, the renderer starts at 0×0 pixels.

**Fix 1 — Hard refresh:**
Press `Ctrl + Shift + R` (force reload, bypasses cache)

**Fix 2 — Resize the window:**
Drag the browser window to resize. The renderer listens to `resize` events and corrects itself.

**Fix 3 — Switch tabs and back:**
Click a different tab then click "Locomotion Sim" again. The tab switch triggers a resize event.

**Fix 4 — Check the browser console:**
Press `F12` → Console tab. Look for red errors. Common issues:
- `THREE is not defined` → CDN blocked, try offline (see below)
- `Cannot read property 'getContext'` → container timing issue (fixed in v3.0)

**Fix 5 — Offline Three.js:**
If CDN is blocked on your network, download Three.js locally:
```bash
cd /home/aki/Downloads/MAGI/simulation
curl -O https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js
curl -O https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js
```
Then change `index.html` to use local paths:
```html
<script src="three.min.js"></script>
<script src="OrbitControls.js"></script>
```

---

### Training Runs But Robot Doesn't Move

Make sure you are on the **RL Gait Training** tab (🧠). The training sandbox only renders when that tab is active.

Also check **Simulation Speed** — at 100× Turbo the visual update rate is throttled to 10% of frames (for performance). Switch to 1× to see full animation.

---

### Port 8765 Can't Be Reached

```bash
# Kill any old servers
pkill -f "http.server"

# Start fresh persistent server
cd /home/aki/Downloads/MAGI/simulation
python3 -m http.server 8765
```

Then open: **http://127.0.0.1:8765** (use `127.0.0.1` instead of `localhost` if DNS resolves differently)

---

### Phase Diagram Is All One Color

The phase diagram draws swing (orange) vs stance (dark green). If all bars are the same color:
- Check you are NOT in **Stand** mode (no movement = all stance)
- Make sure **Gait Frequency > 0.2 Hz**
- The diagram needs ~0.5 seconds of data to fill — wait a moment after switching gaits

---

## 📁 File Structure

```
simulation/
├── index.html          ← Main UI + all 5 tabs
├── style.css           ← All styling (dark theme + v3.0 components)
├── robot.js            ← Three.js 3D visualiser + IK solver
├── gait.js             ← Bézier gait engine (Crawl/Trot/Gallop/Bound)
├── rl_train.js         ← Genetic algorithm + NNPolicy trainer
├── magi_fusion.js      ← Lugia decision engine (priority rules)
├── dashboard.js        ← Main coordinator + all tab event handlers
├── SpiderQ.urdf        ← Physical robot URDF definition
├── SpiderQ.glb         ← 3D robot mesh (GLB format)
├── MAGI_gait_learning.py ← Python standalone RL trainer
└── Guide.md            ← This file
```

---

## 🏗️ Architecture Diagram

```
Browser
├── dashboard.js  ──── runMainLoop() at 60fps ─────────────────────┐
│                                                                   │
├── robot.js      ──── MagiVisualizer                              │
│     ├── Three.js scene (WebGL renderer)                          │
│     ├── solveIK(x,y,z) → {coxa, femur, tibia}                   │
│     └── updateRobotPose() → applies joint angles                 │
│                                                                   │
├── gait.js       ──── GaitGenerator                               │
│     ├── tick() → computes foot targets per frame                 │
│     ├── Bézier arc for swing phase                               │
│     └── Phase diagram ring buffer                                │
│                                                                   │
├── rl_train.js   ──── RLTrainer                                   │
│     ├── RLAgent × N (genes or NNPolicy)                          │
│     ├── evaluateStep() → fitness accumulation                    │
│     ├── nextGeneration() → selection + crossover + mutation      │
│     └── drawGenes() / drawPhaseDiagram()                         │
│                                                                   │
└── magi_fusion.js──── LugiaDecisionEngine                        │
      ├── evaluateRules() → priority-ordered rule chain            │
      └── applyLocomotionLoopback() → overrides GaitGenerator ─────┘
```

---

*MAGI OS v3.0 — CELEBI · GENGAR · LUGIA*
