/* ============================================================
   MAGI Locomotion Controller — gait.js  (v4.0)
   Bézier foot arcs · Crawl · Trot · Gallop · Bound · Stand
   Spider/Crab Configuration — Legs spread OUTWARD
   ============================================================

   COORDINATE SYSTEM (matches robot.js):
   +X = Forward
   +Y = Up
   +Z = Right

   FOOT POSITIONS are in WORLD coordinates.
   Neutral feet are spread outward laterally (large Z offset).
   ============================================================ */

// ---------- Utility: Cubic Bézier interpolation --------------
function bezier3(t, p0, p1, p2, p3) {
    const u = 1 - t;
    return u*u*u*p0 + 3*u*u*t*p1 + 3*u*t*t*p2 + t*t*t*p3;
}

function bezierStepHeight(legPhase, maxHeight) {
    return bezier3(legPhase, 0, maxHeight * 0.15, maxHeight * 1.1, 0);
}

function bezierStepX(legPhase, startX, endX) {
    return bezier3(legPhase, startX, startX + (endX - startX) * 0.3,
                                     startX + (endX - startX) * 0.7, endX);
}

// ---------- Gait sequence definitions ------------------------
const GAIT_SEQUENCES = {
    crawl: [
        { leg: 0, sw0: 0.00, sw1: 0.25 },   // FR
        { leg: 1, sw0: 0.25, sw1: 0.50 },   // FL
        { leg: 3, sw0: 0.50, sw1: 0.75 },   // BL
        { leg: 2, sw0: 0.75, sw1: 1.00 }    // BR  (duty ≈ 75%)
    ],
    trot: [
        { leg: 0, sw0: 0.00, sw1: 0.40 },   // FR+BL diagonal
        { leg: 3, sw0: 0.00, sw1: 0.40 },
        { leg: 1, sw0: 0.50, sw1: 0.90 },   // FL+BR diagonal
        { leg: 2, sw0: 0.50, sw1: 0.90 }
    ],
    gallop: [
        { leg: 0, sw0: 0.00, sw1: 0.25 },
        { leg: 1, sw0: 0.10, sw1: 0.35 },
        { leg: 2, sw0: 0.50, sw1: 0.75 },
        { leg: 3, sw0: 0.60, sw1: 0.85 }
    ],
    bound: [
        { leg: 0, sw0: 0.00, sw1: 0.35 },
        { leg: 1, sw0: 0.00, sw1: 0.35 },
        { leg: 2, sw0: 0.50, sw1: 0.85 },
        { leg: 3, sw0: 0.50, sw1: 0.85 }
    ],
    stand: []
};

const SWAY_TARGETS = {
    crawl: [
        [15, -10],   // FR swing → sway back-left
        [-15, -10],  // FL swing → back-right
        [10,  15],   // BL swing → front-left
        [-10,  15]   // BR swing → front-right
    ],
    trot:   [[0,0],[0,0],[0,0],[0,0]],
    gallop: [[0,0],[0,0],[0,0],[0,0]],
    bound:  [[0,0],[0,0],[0,0],[0,0]],
    stand:  [[0,0],[0,0],[0,0],[0,0]]
};

// ============================================================
class GaitGenerator {
    constructor(visualizer) {
        this.visualizer = visualizer;

        // Gait parameters
        this.gaitType   = 'crawl';
        this.frequency  = 1.5;   // Hz
        this.stride     = 60;    // mm
        this.stepHeight = 30;    // mm
        this.bodyHeight = 100;   // mm above ground — spider stance

        // Directional inputs (normalized -1 to 1)
        this.targetVx = 1.0;
        this.targetVz = 0.0;
        this.targetYawRate = 0.0;

        // Body pose
        this.bodyPos = { x: 0, y: 0, z: 0 };
        this.bodyRot = { pitch: 0, roll: 0, yaw: 0 };

        // Global position — the robot moves through the world
        this.globalX = 0;
        this.globalZ = 0;

        /* Neutral foot positions relative to body center (spider stance):
           Legs spread far out laterally, not tucked underneath.
           FR: front-right, FL: front-left, BR: back-right, BL: back-left */
        this.neutralFeetLocal = [
            { x:  120, y: 0, z:  140 },  // FR — forward, far right
            { x:  120, y: 0, z: -140 },  // FL — forward, far left
            { x: -120, y: 0, z:  140 },  // BR — backward, far right
            { x: -120, y: 0, z: -140 },  // BL — backward, far left
        ];

        this.currentFeet = this.neutralFeetLocal.map(f => ({ ...f }));

        // Phase
        this.time = 0;
        this.lastTime = Date.now();

        // Energy tracking
        this.cumulativeEnergy = 0;
        this.prevAngles = null;

        // Phase diagram
        this.phaseDiagramData = [[], [], [], []];
        this.PHASE_HISTORY = 120;

        // Swing state
        this.swingFlags = [false, false, false, false];
    }

    setGaitType(type) {
        if (this.gaitType !== type) {
            this.gaitType = type;
            this.bodyPos.z = 0;
            this.bodyPos.x = 0;
        }
    }

    updateParam(key, value) {
        const v = parseFloat(value);
        if (key === 'frequency')  this.frequency  = v;
        if (key === 'stride')     this.stride     = v;
        if (key === 'stepHeight') this.stepHeight = v;
        if (key === 'targetVx')   this.targetVx   = v;
        if (key === 'targetVz')   this.targetVz   = v;
        if (key === 'targetYawRate') this.targetYawRate = v;
    }

    updateBodyParam(key, value) {
        const v = parseFloat(value);
        if (key === 'z') {
            this.bodyHeight = v;
        } else if (key === 'pitch') {
            this.bodyRot.pitch = v;
        } else if (key === 'roll') {
            this.bodyRot.roll = v;
        }
    }

    tick() {
        const now = Date.now();
        const dt  = Math.min((now - this.lastTime) / 1000.0, 0.05);
        this.lastTime = now;

        if (this.gaitType !== 'stand') {
            this.time += dt * this.frequency;
            // Move the robot through the world
            const actualStrideX = this.stride * this.targetVx;
            const actualStrideZ = this.stride * this.targetVz;
            this.globalX += actualStrideX * this.frequency * dt;
            this.globalZ += actualStrideZ * this.frequency * dt;
            this.bodyRot.yaw += this.targetYawRate * 30 * dt; // 30 deg/sec max turn
        }

        const phase = this.time % 1.0;
        const seq   = GAIT_SEQUENCES[this.gaitType] || [];

        // Compute feet in WORLD coordinates
        const swinging = [false, false, false, false];
        const worldFeet = [];

        // Calculate yaw rotation in radians
        const yawRad = this.bodyRot.yaw * Math.PI / 180;
        const cosY = Math.cos(yawRad);
        const sinY = Math.sin(yawRad);

        // Precompute actual strides
        const actualStrideX = this.stride * this.targetVx;
        const actualStrideZ = this.stride * this.targetVz;

        // First, compute terrain heights under each neutral foot position
        let sumGroundY = 0;
        const groundHeights = [];
        for (let i = 0; i < 4; i++) {
            const nf = this.neutralFeetLocal[i];
            // Rotate neutral foot around body yaw
            const rx = nf.x * cosY + nf.z * sinY;
            const rz = -nf.x * sinY + nf.z * cosY;
            
            const wx = this.globalX + rx;
            const wz = this.globalZ + rz;
            const gy = window.getMagiTerrainHeight 
                     ? window.getMagiTerrainHeight(wx, wz, this.visualizer.terrainType || 'soil')
                     : -120;
            groundHeights.push(gy);
            sumGroundY += gy;
        }
        const avgGroundY = sumGroundY / 4;

        for (let i = 0; i < 4; i++) {
            const nf = this.neutralFeetLocal[i];
            const rx = nf.x * cosY + nf.z * sinY;
            const rz = -nf.x * sinY + nf.z * cosY;
            
            // Neutral world position of this foot
            const neutralWx = this.globalX + rx;
            const neutralWz = this.globalZ + rz;
            const groundY = groundHeights[i];

            const foot = { x: neutralWx, y: groundY, z: neutralWz };

            // Find this leg's gait entry
            const entry = seq.find(e => e.leg === i);
            if (!entry) {
                worldFeet.push(foot);
                continue;
            }

            let legPhase, isSwing;
            if (entry.sw1 > entry.sw0) {
                isSwing  = phase >= entry.sw0 && phase < entry.sw1;
                legPhase = isSwing ? (phase - entry.sw0) / (entry.sw1 - entry.sw0) : null;
            } else {
                isSwing  = phase >= entry.sw0 || phase < entry.sw1;
                if (isSwing) {
                    legPhase = phase >= entry.sw0
                        ? (phase - entry.sw0) / (1.0 - entry.sw0 + entry.sw1)
                        : (phase + 1.0 - entry.sw0) / (1.0 - entry.sw0 + entry.sw1);
                } else {
                    legPhase = null;
                }
            }

            if (isSwing) {
                swinging[i] = true;
                const startX = neutralWx - actualStrideX / 2;
                const endX   = neutralWx + actualStrideX / 2;
                const startZ = neutralWz - actualStrideZ / 2;
                const endZ   = neutralWz + actualStrideZ / 2;
                
                foot.x = bezierStepX(legPhase, startX, endX);
                foot.z = bezierStepX(legPhase, startZ, endZ);
                foot.y = groundY + bezierStepHeight(legPhase, this.stepHeight);
            } else {
                // Stance: foot slides backward relative to body
                const swingDuration = entry.sw1 > entry.sw0
                    ? (entry.sw1 - entry.sw0)
                    : (1.0 - entry.sw0 + entry.sw1);
                const stanceDuration = 1.0 - swingDuration;

                let stancePhase;
                if (entry.sw1 > entry.sw0) {
                    stancePhase = phase >= entry.sw1
                        ? (phase - entry.sw1) / stanceDuration
                        : (phase + 1.0 - entry.sw1) / stanceDuration;
                } else {
                    stancePhase = (phase >= entry.sw1 && phase < entry.sw0)
                        ? (phase - entry.sw1) / stanceDuration
                        : 0;
                }
                stancePhase = Math.min(Math.max(stancePhase, 0), 1);
                foot.x = neutralWx + actualStrideX / 2 - actualStrideX * stancePhase;
                foot.z = neutralWz + actualStrideZ / 2 - actualStrideZ * stancePhase;
                foot.y = groundY; // firmly on terrain
            }

            worldFeet.push(foot);
        }

        // Body sway
        if (this.gaitType === 'crawl') {
            const swingSlot = seq.findIndex(e => swinging[e.leg]);
            if (swingSlot >= 0) {
                const sway = SWAY_TARGETS.crawl[swingSlot] || [0, 0];
                this.bodyPos.z += (sway[0] - this.bodyPos.z) * 0.08;
                this.bodyPos.x += (sway[1] - this.bodyPos.x) * 0.08;
            }
        } else if (this.gaitType === 'gallop' || this.gaitType === 'bound') {
            const pitchAmp = this.gaitType === 'gallop' ? 6 : 10;
            this.bodyRot.pitch += (Math.sin(phase * Math.PI * 2) * pitchAmp - this.bodyRot.pitch) * 0.15;
            this.bodyPos.z = 0;
            this.bodyPos.x = 0;
        } else {
            this.bodyPos.z += (0 - this.bodyPos.z) * 0.1;
            this.bodyPos.x += (0 - this.bodyPos.x) * 0.1;
        }

        // Swing flags + telemetry highlights
        this.swingFlags = swinging;
        for (let i = 0; i < 4; i++) {
            const el = document.getElementById(`telemetry-l${i + 1}`);
            if (el) {
                if (swinging[i]) el.classList.add('swing');
                else el.classList.remove('swing');
            }
        }

        // Phase diagram data
        for (let i = 0; i < 4; i++) {
            this.phaseDiagramData[i].push({ phase, swing: swinging[i] });
            if (this.phaseDiagramData[i].length > this.PHASE_HISTORY) {
                this.phaseDiagramData[i].shift();
            }
        }

        // Calculate body world position
        // Body Y = average ground height + configured body height
        const bodyWorldPos = {
            x: this.globalX + this.bodyPos.x,
            y: avgGroundY + this.bodyHeight,
            z: this.globalZ + this.bodyPos.z
        };

        // Store for external access
        this.currentFeet = worldFeet;

        // Send to visualizer
        const jointAngles = this.visualizer.updateRobotPose(
            bodyWorldPos, this.bodyRot, worldFeet, this.swingFlags
        );

        // Energy tracking
        if (this.prevAngles && jointAngles) {
            let energyStep = 0;
            for (let i = 0; i < 4; i++) {
                const prev = this.prevAngles[i];
                const curr = jointAngles[i];
                if (prev && curr) {
                    energyStep += Math.pow(curr.coxa  - prev.coxa,  2);
                    energyStep += Math.pow(curr.femur - prev.femur, 2);
                    energyStep += Math.pow(curr.tibia - prev.tibia, 2);
                }
            }
            this.cumulativeEnergy += energyStep * dt;
        }
        this.prevAngles = jointAngles;

        return jointAngles;
    }

    _copyFeet(src) {
        return src.map(f => ({ x: f.x, y: f.y, z: f.z }));
    }

    getSwingFlags() { return this.swingFlags; }
    getPhaseDiagramData() { return this.phaseDiagramData; }
}
