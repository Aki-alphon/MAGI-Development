/* ============================================================
   MAGI Dashboard Coordinator — dashboard.js  (v3.0)
   Wires all modules · Phase diagram · Gene canvas · Gait Analysis
   ============================================================ */

// Reference to large phase canvas (Analysis tab)
let _largePhaseCtx = null;

// ---- Global instances ----------------------------------------
let simVisualizer, gaitController, decisionEngine;
let trainVisualizer, rlTrainer;
let activeTab = 'simulation';

// ---- Bootstrap -----------------------------------------------
window.addEventListener('DOMContentLoaded', () => {
    // 1. Main simulator
    simVisualizer  = new MagiVisualizer('canvas-container');
    gaitController = new GaitGenerator(simVisualizer);
    decisionEngine = new LugiaDecisionEngine(gaitController);
    decisionEngine.setConsole('lugia-console');

    // 2. RL Training sandbox (minimal — no CoM/poly)
    trainVisualizer = new MagiVisualizer('train-canvas-container', true);
    rlTrainer       = new RLTrainer(trainVisualizer);

    // 3. Math IK tab init
    solveMathIK();

    // 4. Gait analysis tab init
    initGaitAnalysis();

    // 5. Large phase canvas init
    const lpc = document.getElementById('phase-canvas-large');
    if (lpc) _largePhaseCtx = lpc.getContext('2d');

    // 5. Main loop
    runMainLoop();
});

// ---- Tab switching -------------------------------------------
function switchTab(tabId) {
    activeTab = tabId;

    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(btn => {
        if (btn.getAttribute('onclick') && btn.getAttribute('onclick').includes(tabId)) {
            btn.classList.add('active');
        }
    });

    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    const target = document.getElementById(`tab-${tabId}`);
    if (target) target.classList.add('active');

    // Pause training when leaving training tab
    if (tabId !== 'training' && rlTrainer) {
        rlTrainer.isTraining = false;
    }

    setTimeout(() => window.dispatchEvent(new Event('resize')), 100);
}

// ---- Locomotion Simulator handlers ---------------------------
function setGait(gaitType) {
    if (decisionEngine && decisionEngine.action === 'EMERGENCY') return;

    // Clear active state from all buttons
    ['crawl', 'trot', 'gallop', 'bound', 'stand'].forEach(g => {
        const el = document.getElementById(`gait-${g}`);
        if (el) el.classList.remove('active');
    });

    const btn = document.getElementById(`gait-${gaitType}`);
    if (btn) btn.classList.add('active');

    gaitController.setGaitType(gaitType);
}

function updateGaitParam(param, val) {
    const paramToId = { frequency: 'frequency', stride: 'stride', stepHeight: 'step-height', targetVx: 'target-vx', targetVz: 'target-vz', targetYawRate: 'target-yaw-rate' };
    const idKey     = paramToId[param] || param.replace(/([A-Z])/g, '-$1').toLowerCase();
    
    let suffix = ' mm';
    if (param === 'frequency') suffix = ' Hz';
    else if (param.startsWith('target')) suffix = '';

    const el        = document.getElementById(`val-${idKey}`);
    if (el) el.innerText = val + suffix;
    gaitController.updateParam(param, val);
}

function updateBodyParam(param, val) {
    const suffix = (param === 'pitch' || param === 'roll') ? '°' : ' mm';
    const el     = document.getElementById(`val-body-${param}`);
    if (el) el.innerText = val + suffix;
    gaitController.updateBodyParam(param, val);
}

function toggleSupportPolygon(checked) {
    if (!simVisualizer) return;
    if (simVisualizer.supportPolyLine) simVisualizer.supportPolyLine.visible = checked;
    if (simVisualizer.comIndicator)    simVisualizer.comIndicator.visible    = checked;
    if (simVisualizer.comLine)         simVisualizer.comLine.visible         = checked;
}

function setTerrainRoughness(val) {
    const amp = parseFloat(val);
    if (simVisualizer) simVisualizer.setTerrainRoughness(amp);
    const el = document.getElementById('val-terrain-amp');
    if (el) el.innerText = amp.toFixed(0) + ' mm';
}

// ---- RL Training handlers ------------------------------------
function toggleTraining() {
    if (rlTrainer) rlTrainer.toggle();
}

function setTrainingSpeed(speed) {
    if (rlTrainer) rlTrainer.setSpeed(speed);
}

function resetTraining() {
    if (rlTrainer) rlTrainer.reset();
}

function updateTerrainType(val) {
    if (rlTrainer) rlTrainer.updateTerrain(val);
    if (simVisualizer) simVisualizer.setTerrainType(val);
}

function setNNMode(enabled) {
    if (rlTrainer) rlTrainer.setNNMode(enabled);
    const badge = document.getElementById('nn-mode-badge');
    if (badge) {
        badge.innerText   = enabled ? 'NNPolicy' : 'DirectGene';
        badge.style.color = enabled ? '#3de8ff'  : '#e67e22';
    }
}

function exportGenome() {
    if (rlTrainer) rlTrainer.exportBestGenome();
}

// ---- Decision Fusion handlers --------------------------------
function setScenario(name)             { if (decisionEngine) decisionEngine.setScenario(name); }
function overrideSensor(sensor, value) {
    const elVal = document.getElementById(`val-override-${sensor}`);
    if (sensor === 'tof') {
        decisionEngine.tof = parseInt(value);
        if (elVal) elVal.innerText = `${value} mm`;
    } else if (sensor === 'anomaly') {
        decisionEngine.anomalyScore = parseFloat(value);
        if (elVal) elVal.innerText = decisionEngine.anomalyScore.toFixed(2);
    }
    decisionEngine.evaluateRules();
}

// ---- Kinematics Math tab -------------------------------------
function solveMathIK() {
    const x = parseFloat(document.getElementById('math-x').value);
    const y = parseFloat(document.getElementById('math-y').value);
    const z = parseFloat(document.getElementById('math-z').value);

    document.getElementById('math-val-x').innerText = `${x} mm`;
    document.getElementById('math-val-y').innerText = `${y} mm`;
    document.getElementById('math-val-z').innerText = `${z} mm`;

    // Map UI coords (x=Fwd, y=Lateral, z=Up) to new IK frame (ikX=Outward, ikY=Fwd, ikZ=Down)
    const angles = solveIK(y, x, -z);
    document.getElementById('math-res-coxa').innerText  = `${angles.coxa.toFixed(1)}°`;
    document.getElementById('math-res-femur').innerText = `${angles.femur.toFixed(1)}°`;
    document.getElementById('math-res-tibia').innerText = `${angles.tibia.toFixed(1)}°`;

    // Draw 2D leg silhouette on the IK canvas
    drawIKDiagram(x, y, z, angles);
}

function drawIKDiagram(x, y, z, angles) {
    const canvas = document.getElementById('ik-diagram-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W   = canvas.width;
    const H   = canvas.height;
    ctx.clearRect(0, 0, W, H);

    ctx.fillStyle = '#0b0b0f';
    ctx.fillRect(0, 0, W, H);

    // Draw sagittal plane view (x-z projection)
    const cx  = W / 2;
    const cy  = H * 0.3;
    const sc  = 0.6; // pixel-per-mm scale

    const draw = (x1, y1, x2, y2, color, width = 2) => {
        ctx.strokeStyle = color;
        ctx.lineWidth   = width;
        ctx.beginPath();
        ctx.moveTo(cx + x1 * sc, cy - y1 * sc);
        ctx.lineTo(cx + x2 * sc, cy - y2 * sc);
        ctx.stroke();
    };

    const dot = (px, py, r, color) => {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(cx + px * sc, cy - py * sc, r, 0, Math.PI * 2);
        ctx.fill();
    };

    const r  = Math.sqrt(x*x + y*y);

    // Coxa pivot
    const cEnd = { x: LINK_COXA, y: 0 };
    
    // Femur end (angles.femur is elevation below horizontal, e.g. -45°)
    const fRad = angles.femur * Math.PI / 180;
    const fEnd = {
        x: cEnd.x + LINK_FEMUR * Math.cos(fRad),
        y: cEnd.y + LINK_FEMUR * Math.sin(fRad)
    };
    
    // Tibia end (angles.tibia is interior angle, so we subtract it from the extended femur line)
    // When tibia=180, it's straight out. When tibia<180, it bends backwards.
    // However, the standard law of cosines gives us the interior angle (e.g. 90° for a right angle knee)
    const tRad = fRad - Math.PI + (angles.tibia * Math.PI / 180);
    const tEnd = {
        x: fEnd.x + LINK_TIBIA * Math.cos(tRad),
        y: fEnd.y + LINK_TIBIA * Math.sin(tRad)
    };

    // Ground line
    ctx.strokeStyle = '#2b2b33';
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.moveTo(0, cy - z * sc);
    ctx.lineTo(W, cy - z * sc);
    ctx.stroke();

    draw(0, 0, LINK_COXA, 0, '#3de8ff', 3);        // Coxa
    draw(LINK_COXA, 0, fEnd.x, fEnd.y, '#e67e22', 4); // Femur
    draw(fEnd.x, fEnd.y, tEnd.x, tEnd.y, '#e0e0e8', 3); // Tibia

    dot(0, 0, 6, '#e67e22');
    dot(LINK_COXA, 0, 5, '#3de8ff');
    dot(fEnd.x, fEnd.y, 5, '#e67e22');
    dot(tEnd.x, tEnd.y, 7, '#2ecc71');

    // Labels
    ctx.fillStyle = '#8b8b9a';
    ctx.font      = '10px Share Tech Mono';
    ctx.fillText(`θc=${angles.coxa.toFixed(1)}°`, cx + 5, cy + 15);
    ctx.fillText(`θf=${angles.femur.toFixed(1)}°`, cx + LINK_COXA * sc + 5, cy - 10);
}

// ---- Gait Analysis tab init ----------------------------------
let gaStrideHistory = [];
let gaEnergyHistory = [];
let gaTimeHistory   = [];

function initGaitAnalysis() {
    // Stride efficiency mini-chart
    const ctx = document.getElementById('ga-stride-chart');
    if (!ctx) return;
    window._gaChart = new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Stride Efficiency (mm/s)',
                    borderColor: '#2ecc71',
                    backgroundColor: 'rgba(46,204,113,0.1)',
                    data: [],
                    borderWidth: 1.5,
                    tension: 0.4,
                    fill: true,
                    pointRadius: 0
                },
                {
                    label: 'Cumulative Energy (J proxy)',
                    borderColor: '#e67e22',
                    backgroundColor: 'rgba(230,126,34,0.1)',
                    data: [],
                    borderWidth: 1.5,
                    tension: 0.4,
                    fill: false,
                    yAxisID: 'y2',
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            scales: {
                x: { grid: { color: '#1e1e28' }, ticks: { color: '#8b8b9a', maxTicksLimit: 8 } },
                y: { grid: { color: '#1e1e28' }, ticks: { color: '#2ecc71' }, position: 'left' },
                y2:{ grid: { drawOnChartArea: false }, ticks: { color: '#e67e22' }, position: 'right' }
            },
            plugins: {
                legend: { labels: { color: '#e2e2e9', font: { family: 'Outfit', size: 11 }, boxWidth: 12 } }
            }
        }
    });
}

let _gaFrameCount = 0;

function updateGaitAnalysis() {
    _gaFrameCount++;
    if (_gaFrameCount % 10 !== 0) return; // update every 10 frames

    const chart = window._gaChart;
    if (!chart || !gaitController) return;

    const tSec = (Date.now() / 1000).toFixed(1);

    // Approximate stride efficiency from frequency and stride length
    const strideEff = gaitController.frequency * gaitController.stride;
    const energy    = gaitController.cumulativeEnergy;

    gaStrideHistory.push(strideEff.toFixed(1));
    gaEnergyHistory.push(energy.toFixed(2));
    gaTimeHistory.push(tSec);

    if (gaStrideHistory.length > 80) {
        gaStrideHistory.shift();
        gaEnergyHistory.shift();
        gaTimeHistory.shift();
    }

    chart.data.labels                = gaTimeHistory;
    chart.data.datasets[0].data      = gaStrideHistory;
    chart.data.datasets[1].data      = gaEnergyHistory;
    chart.update('none');

    // Foot trajectory dots on 2D canvas
    drawFootTrajectory();

    // Gait statistics panel
    const s = id => document.getElementById(id);
    if (s('ga-gait-type'))   s('ga-gait-type').innerText   = gaitController.gaitType.toUpperCase();
    if (s('ga-frequency'))   s('ga-frequency').innerText   = gaitController.frequency.toFixed(1) + ' Hz';
    if (s('ga-stride'))      s('ga-stride').innerText      = gaitController.stride.toFixed(0) + ' mm';
    if (s('ga-step-height')) s('ga-step-height').innerText = gaitController.stepHeight.toFixed(0) + ' mm';
    if (s('ga-energy'))      s('ga-energy').innerText      = energy.toFixed(2);
    if (s('ga-stride-eff'))  s('ga-stride-eff').innerText  = strideEff.toFixed(1) + ' mm/s';
}

// Foot path canvas — top-down XZ view of all 4 feet
const _footPaths = [[], [], [], []];
let _footPathTimer = 0;

function drawFootTrajectory() {
    const canvas = document.getElementById('foot-path-canvas');
    if (!canvas || !gaitController) return;
    const ctx = canvas.getContext('2d');
    const W   = canvas.width;
    const H   = canvas.height;

    // Record current foot positions every N frames
    if (_footPathTimer++ % 4 === 0) {
        for (let i = 0; i < 4; i++) {
            const f = gaitController.currentFeet[i];
            _footPaths[i].push({ x: f.x, y: f.z }); // top view XZ
            if (_footPaths[i].length > 120) _footPaths[i].shift();
        }
    }

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0b0b0f';
    ctx.fillRect(0, 0, W, H);

    const colors = ['#e67e22', '#2ecc71', '#3de8ff', '#e74c3c'];
    const legs   = ['FR', 'FL', 'BR', 'BL'];
    const cx     = W / 2, cy = H / 2;
    const sc     = W / 500;

    // Grid cross
    ctx.strokeStyle = '#1e1e28';
    ctx.lineWidth   = 1;
    ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(W, cy); ctx.stroke();

    // Draw paths
    for (let i = 0; i < 4; i++) {
        const path = _footPaths[i];
        if (path.length < 2) continue;
        ctx.strokeStyle = colors[i];
        ctx.lineWidth   = 1.5;
        ctx.globalAlpha = 0.7;
        ctx.beginPath();
        ctx.moveTo(cx + path[0].x * sc, cy - path[0].y * sc);
        for (let s = 1; s < path.length; s++) {
            ctx.lineTo(cx + path[s].x * sc, cy - path[s].y * sc);
        }
        ctx.stroke();
        ctx.globalAlpha = 1;

        // Current foot dot
        const last = path[path.length - 1];
        ctx.fillStyle = colors[i];
        ctx.beginPath();
        ctx.arc(cx + last.x * sc, cy - last.y * sc, 4, 0, Math.PI * 2);
        ctx.fill();

        // Label
        ctx.fillStyle = colors[i];
        ctx.font      = '10px Share Tech Mono';
        ctx.fillText(legs[i], cx + last.x * sc + 6, cy - last.y * sc + 4);
    }
}

// ---- Main loop -----------------------------------------------
let lastFrameTime = Date.now();

function runMainLoop() {
    requestAnimationFrame(runMainLoop);

    const now = Date.now();
    const dt  = Math.min((now - lastFrameTime) / 1000.0, 0.05);
    lastFrameTime = now;

    if (activeTab === 'simulation' || activeTab === 'fusion') {
        // Lugia decision fusion at ~50Hz
        if (Math.random() < 0.25) decisionEngine.evaluateRules();

        // Tick gait and pass swing flags to visualiser
        const angles = gaitController.tick();
        // Pass swing flags to sim visualizer for foot contact rings
        if (simVisualizer) {
            const flags = gaitController.getSwingFlags();
            for (let i = 0; i < 4; i++) {
                const ring = simVisualizer.footRings[i];
                if (ring) {
                    const t = flags[i] ? 0 : 0.7;
                    ring.material.opacity += (t - ring.material.opacity) * 0.15;
                }
            }
        }

        updateJointTelemetryUI(angles);
        updateImuUI();

        // Phase diagram update (always running in sidebar canvas)
        if (rlTrainer && gaitController) {
            rlTrainer.drawPhaseDiagram(gaitController);
        }

    } else if (activeTab === 'training') {
        rlTrainer.update(dt);

        // Draw phase diagram in training sandbox too
        if (rlTrainer && gaitController) {
            rlTrainer.drawPhaseDiagram(gaitController);
        }

    } else if (activeTab === 'analysis') {
        gaitController.tick();
        updateGaitAnalysis();
        // Draw phase diagram into both sidebar canvas and large analysis canvas
        if (rlTrainer && gaitController) {
            rlTrainer.drawPhaseDiagram(gaitController);
            // Also draw into the large analysis-tab canvas
            if (_largePhaseCtx) {
                const origCanvas = rlTrainer.phaseCanvas;
                const origCtx    = rlTrainer.phaseCtx;
                const lpcEl      = document.getElementById('phase-canvas-large');
                if (lpcEl) {
                    rlTrainer.phaseCanvas = lpcEl;
                    rlTrainer.phaseCtx    = _largePhaseCtx;
                    rlTrainer.drawPhaseDiagram(gaitController);
                    rlTrainer.phaseCanvas = origCanvas;
                    rlTrainer.phaseCtx    = origCtx;
                }
            }
        }
    }
}

// ---- UI helpers ----------------------------------------------
// LEG_META is defined in robot.js (loads first):
//   index 0 = FR (isRight:true), 1 = FL, 2 = BR (isRight:true), 3 = BL
function updateJointTelemetryUI(angles) {
    if (!angles) return;
    for (let i = 0; i < 4; i++) {
        const a       = angles[i];
        const idx     = i + 1;
        const meta    = (typeof LEG_META !== 'undefined') ? LEG_META[i] : { isRight: (i === 0 || i === 2) };
        // ikToServo converts raw IK geometry angles → MG996R PWM servo values [0..180]
        const servo   = (typeof ikToServo !== 'undefined')
            ? ikToServo(a, meta.isRight)
            : {
                coxa:  Math.min(180, Math.max(0, Math.round(meta.isRight ? 90 - a.coxa : 90 + a.coxa))),
                femur: Math.min(180, Math.max(0, Math.round(90 - a.femur))),
                tibia: Math.min(180, Math.max(0, Math.round(a.tibia)))
              };
        const set = (id, val) => { const e = document.getElementById(id); if (e) e.innerText = val; };
        set(`val-l${idx}-coxa`,  `${servo.coxa}°`);
        set(`val-l${idx}-femur`, `${servo.femur}°`);
        set(`val-l${idx}-tibia`, `${servo.tibia}°`);
    }
}

function updateImuUI() {
    const pitch = gaitController.bodyRot.pitch;
    const roll  = gaitController.bodyRot.roll;
    const jitter = () => Math.random() * 0.4 - 0.2;
    const osc = gaitController.gaitType !== 'stand'
        ? Math.sin(gaitController.time * Math.PI * 4) * 0.12 : 0;
    const accZ = 1.0 + osc + (Math.random() * 0.04 - 0.02);

    const set = (id, val) => { const e = document.getElementById(id); if (e) e.innerText = val; };
    set('val-imu-pitch', `${(pitch + jitter()).toFixed(2)}°`);
    set('val-imu-roll',  `${(roll  + jitter() * 0.7).toFixed(2)}°`);
    set('val-imu-accz',  `${accZ.toFixed(2)}g`);
}
