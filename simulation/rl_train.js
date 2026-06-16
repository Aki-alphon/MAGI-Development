/* ============================================================
   MAGI Reinforcement Learning Training — rl_train.js  (v4.0)
   Genetic Algorithm · Neural Network Policy · Rich Fitness
   Production-Grade Environment — Bot traverses real terrain
   ============================================================ */

// ============================================================
//  Neural Network Policy — 3-layer feedforward net
// ============================================================
class NNPolicy {
    constructor() {
        this.inputSize  = 11;
        this.hiddenSize = 24;
        this.outputSize = 16;
        this.paramCount = this.inputSize * this.hiddenSize + this.hiddenSize
                        + this.hiddenSize * this.outputSize + this.outputSize;
        this.params = new Float32Array(this.paramCount).fill(0);
    }

    randomize() {
        const std1 = Math.sqrt(2 / this.inputSize);
        const std2 = Math.sqrt(2 / this.hiddenSize);
        let idx = 0;
        for (let i = 0; i < this.inputSize * this.hiddenSize; i++)
            this.params[idx++] = (Math.random() * 2 - 1) * std1;
        for (let i = 0; i < this.hiddenSize; i++)
            this.params[idx++] = 0;
        for (let i = 0; i < this.hiddenSize * this.outputSize; i++)
            this.params[idx++] = (Math.random() * 2 - 1) * std2;
        for (let i = 0; i < this.outputSize; i++)
            this.params[idx++] = 0;
    }

    clone() {
        const c = new NNPolicy();
        c.params = new Float32Array(this.params);
        return c;
    }

    forward(input) {
        const W1size = this.inputSize * this.hiddenSize;
        const b1size = this.hiddenSize;
        const W2size = this.hiddenSize * this.outputSize;

        const hidden = new Float32Array(this.hiddenSize);
        for (let j = 0; j < this.hiddenSize; j++) {
            let sum = this.params[W1size + j];
            for (let i = 0; i < this.inputSize; i++) {
                sum += input[i] * this.params[i * this.hiddenSize + j];
            }
            hidden[j] = Math.max(0, sum);
        }

        const output = new Float32Array(this.outputSize);
        const W2off  = W1size + b1size;
        const b2off  = W2off + W2size;
        for (let k = 0; k < this.outputSize; k++) {
            let sum = this.params[b2off + k];
            for (let j = 0; j < this.hiddenSize; j++) {
                sum += hidden[j] * this.params[W2off + j * this.outputSize + k];
            }
            output[k] = Math.tanh(sum);
        }
        return output;
    }

    mutate(rate, scale) {
        for (let i = 0; i < this.paramCount; i++) {
            if (Math.random() < rate) {
                this.params[i] += (Math.random() * 2 - 1) * scale;
            }
        }
    }

    crossover(partner) {
        const child = this.clone();
        for (let i = 0; i < this.paramCount; i++) {
            if (Math.random() < 0.5) child.params[i] = partner.params[i];
        }
        return child;
    }

    toGenes() {
        const b2off = this.inputSize * this.hiddenSize + this.hiddenSize
                    + this.hiddenSize * this.outputSize;
        return Array.from(this.params.slice(b2off, b2off + this.outputSize));
    }
}

// ============================================================
//  RL Agent
// ============================================================
class RLAgent {
    constructor(id, useNN = false) {
        this.id      = id;
        this.useNN   = useNN;
        this.fitness = 0;

        if (useNN) {
            this.policy = new NNPolicy();
            this.policy.randomize();
            this.genes = this.policy.toGenes();
        } else {
            this.genes = [];
            this._randomizeGenes();
        }
    }

    _randomizeGenes() {
        this.genes = [];
        for (let i = 0; i < 16; i++) {
            this.genes.push(i < 4 ? Math.random() : Math.random() * 2 - 1);
        }
    }

    getGenes() {
        if (this.useNN) return this.policy.toGenes();
        return this.genes;
    }

    decode(phase, bodyPitch, bodyRoll, bodyHeight, terrainFriction, targetVx, targetVz, targetYaw) {
        let raw;
        if (this.useNN) {
            const input = [
                Math.sin(phase * Math.PI * 2),
                Math.cos(phase * Math.PI * 2),
                Math.sin(phase * Math.PI * 4),
                Math.cos(phase * Math.PI * 4),
                bodyPitch / 20,
                bodyRoll  / 20,
                bodyHeight / 200,
                terrainFriction,
                targetVx,
                targetVz,
                targetYaw
            ];
            raw = this.policy.forward(input);
        } else {
            raw = this.genes;
        }

        return {
            phaseOffsets: [raw[0], raw[1], raw[2], raw[3]],
            strideX:      [raw[4], raw[5], raw[6], raw[7]],
            strideZ:      [raw[8], raw[9], raw[10], raw[11]],
            stepDur:      raw[12],
            heightBias:   raw[13],
            pitchAmp:     raw[14],
            rollAmp:      raw[15]
        };
    }

    mutate(rate) {
        if (this.useNN) {
            const scale = 0.15 + Math.random() * 0.1;
            this.policy.mutate(rate, scale);
        } else {
            for (let i = 0; i < 16; i++) {
                if (Math.random() < rate) {
                    if (i < 4) {
                        this.genes[i] = (this.genes[i] + (Math.random() * 0.4 - 0.2) + 1.0) % 1.0;
                    } else {
                        this.genes[i] = Math.min(Math.max(
                            this.genes[i] + (Math.random() * 0.4 - 0.2), -1.0), 1.0);
                    }
                }
            }
        }
    }

    crossover(partner) {
        const child = new RLAgent(this.id + '_c', this.useNN);
        if (this.useNN) {
            child.policy = this.policy.crossover(partner.policy);
        } else {
            child.genes = this.genes.map((g, i) =>
                Math.random() < 0.5 ? g : partner.genes[i]);
        }
        return child;
    }
}

// ============================================================
//  RL Trainer — Production-Grade Physics Environment
// ============================================================
class RLTrainer {
    constructor(visualizer) {
        this.visualizer      = visualizer;
        this.popSize         = 30;
        this.mutationRate    = 0.15;
        this.useNN           = false;
        this.population      = [];

        this.generation      = 0;
        this.currentAgentIdx = 0;
        this.bestFitness     = 0;
        this.bestGenes       = null;
        this.allTimeBestAgent = null;

        this.isTraining      = false;
        this.speedMultiplier = 1;
        this.terrainFriction = 1.0;
        this.terrainType     = 'soil';

        // Robot physical state
        this.globalX         = 0;
        this.globalZ         = 0;
        this.startX          = 0;  // track distance per agent

        // Per-agent eval
        this.agentTime    = 0;
        this.maxAgentTime = 6.0;

        // Live scores
        this.liveScores = { velocity: 0, stability: 0, energy: 0, symmetry: 0 };

        // Curriculum
        this.curriculumLevel = 1;

        // Chart
        this.chart         = null;
        this.historyGen    = [];
        this.historyBest   = [];
        this.historyAvg    = [];
        this.historyMin    = [];

        // Canvases
        this.geneCanvas    = null;
        this.geneCtx       = null;
        this.phaseCanvas   = null;
        this.phaseCtx      = null;

        // Previous joint angles for energy calculation
        this._prevJoints   = null;
        // Previous body position for velocity tracking
        this._prevBodyX    = 0;

        this.initPopulation();
        this.initChart();
        this.initGeneCanvas();
        this.initPhaseCanvas();
        this._sendNeutralPose();
    }

    // ---- Initialisation ----

    initPopulation() {
        this.population = [];
        for (let i = 0; i < this.popSize; i++) {
            this.population.push(new RLAgent(i + 1, this.useNN));
        }
        this.generation      = 0;
        this.currentAgentIdx = 0;
        this.bestFitness     = 0;
        this.bestGenes       = null;
        this.historyGen      = [];
        this.historyBest     = [];
        this.historyAvg      = [];
        this.historyMin      = [];
        this.curriculumLevel = 1;
    }

    initChart() {
        const ctx = document.getElementById('reward-chart');
        if (!ctx) return;
        this.chart = new Chart(ctx.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Max Reward',
                        borderColor: '#e67e22',
                        backgroundColor: 'rgba(230,126,34,0.12)',
                        data: [],
                        borderWidth: 2,
                        tension: 0.3,
                        fill: true,
                        pointRadius: 3
                    },
                    {
                        label: 'Avg Reward',
                        borderColor: '#2ecc71',
                        backgroundColor: 'rgba(46,204,113,0.05)',
                        data: [],
                        borderWidth: 1.5,
                        tension: 0.3,
                        pointRadius: 2
                    },
                    {
                        label: 'Min Reward',
                        borderColor: '#e74c3c',
                        backgroundColor: 'transparent',
                        data: [],
                        borderWidth: 1,
                        tension: 0.3,
                        borderDash: [4, 3],
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 0 },
                scales: {
                    x: {
                        grid:  { color: '#1e1e28' },
                        title: { display: true, text: 'Generation', color: '#8b8b9a' },
                        ticks: { color: '#8b8b9a', maxTicksLimit: 12 }
                    },
                    y: {
                        grid:  { color: '#1e1e28' },
                        title: { display: true, text: 'Fitness Score', color: '#8b8b9a' },
                        ticks: { color: '#8b8b9a' }
                    }
                },
                plugins: {
                    legend: { labels: { color: '#e2e2e9', font: { family: 'Outfit', size: 11 } } }
                }
            }
        });
    }

    initGeneCanvas() {
        this.geneCanvas = document.getElementById('gene-canvas');
        if (!this.geneCanvas) return;
        this.geneCtx = this.geneCanvas.getContext('2d');
        this._drawGenes(Array(16).fill(0));
    }

    initPhaseCanvas() {
        this.phaseCanvas = document.getElementById('phase-canvas');
        if (!this.phaseCanvas) return;
        this.phaseCtx = this.phaseCanvas.getContext('2d');
        this._clearPhaseCanvas();
    }

    // ---- Gene visualization ----

    _drawGenes(genes) {
        if (!this.geneCtx || !this.geneCanvas) return;
        const ctx = this.geneCtx;
        const W   = this.geneCanvas.width;
        const H   = this.geneCanvas.height;
        ctx.clearRect(0, 0, W, H);

        const labels = ['φ₀','φ₁','φ₂','φ₃','x₀','x₁','x₂','x₃','z₀','z₁','z₂','z₃','Dur','Hgt','Ptch','Roll'];
        const n      = 16;
        const barW   = (W - 30) / n;
        const midY   = H / 2;

        ctx.strokeStyle = '#2b2b33';
        ctx.lineWidth   = 1;
        ctx.beginPath();
        ctx.moveTo(15, midY);
        ctx.lineTo(W - 15, midY);
        ctx.stroke();

        for (let i = 0; i < n; i++) {
            const g    = genes[i];
            const norm = i < 4 ? (g * 2 - 1) : g;
            const barH = Math.abs(norm) * (midY - 20);
            const x    = 15 + i * barW;
            const y    = norm >= 0 ? midY - barH : midY;

            const hue = norm >= 0
                ? `hsl(${30 + norm * 90}, 85%, 55%)`
                : `hsl(${0}, 75%, 55%)`;

            ctx.fillStyle = hue;
            ctx.fillRect(x + 2, y, barW - 4, Math.max(barH, 1));

            ctx.fillStyle   = '#8b8b9a';
            ctx.font        = '8px Share Tech Mono';
            ctx.textAlign   = 'center';
            ctx.fillText(labels[i], x + barW / 2, H - 4);

            ctx.fillStyle = '#e2e2e9';
            ctx.fillText(g.toFixed(2), x + barW / 2, norm >= 0 ? y - 3 : y + barH + 10);
        }
    }

    _clearPhaseCanvas() {
        if (!this.phaseCtx || !this.phaseCanvas) return;
        const ctx = this.phaseCtx;
        const W   = this.phaseCanvas.width;
        const H   = this.phaseCanvas.height;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#0e0e12';
        ctx.fillRect(0, 0, W, H);
    }

    drawPhaseDiagram(gaitController) {
        if (!this.phaseCtx || !this.phaseCanvas) return;
        const ctx  = this.phaseCtx;
        const W    = this.phaseCanvas.width;
        const H    = this.phaseCanvas.height;
        const data = gaitController.getPhaseDiagramData();
        const legs = ['FR', 'FL', 'BR', 'BL'];
        const rowH = H / 4;

        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#0b0b0e';
        ctx.fillRect(0, 0, W, H);

        const swingColor  = '#e67e22';
        const stanceColor = '#1e3a2b';
        const labelColor  = '#8b8b9a';

        for (let i = 0; i < 4; i++) {
            const rowY    = i * rowH;
            const samples = data[i];
            const n       = samples.length;
            if (n === 0) continue;

            ctx.fillStyle = 'rgba(255,255,255,0.01)';
            ctx.fillRect(0, rowY, W, rowH);

            const labelW = 32;
            const barW   = (W - labelW) / n;
            for (let s = 0; s < n; s++) {
                ctx.fillStyle = samples[s].swing ? swingColor : stanceColor;
                ctx.fillRect(labelW + s * barW, rowY + 4, Math.max(barW, 1), rowH - 8);
            }

            // Live indicator
            const isSwing = gaitController.swingFlags[i];
            const ringX   = W - 14;
            const ringY   = rowY + rowH / 2;
            ctx.beginPath();
            ctx.arc(ringX, ringY, 5, 0, Math.PI * 2);
            ctx.fillStyle = isSwing ? swingColor : '#2ecc71';
            ctx.fill();
            if (isSwing) {
                ctx.beginPath();
                ctx.arc(ringX, ringY, 8, 0, Math.PI * 2);
                ctx.strokeStyle = 'rgba(230,126,34,0.35)';
                ctx.lineWidth   = 2;
                ctx.stroke();
            }

            ctx.fillStyle = labelColor;
            ctx.font      = '10px Share Tech Mono';
            ctx.textAlign = 'left';
            ctx.fillText(legs[i], 4, rowY + rowH / 2 + 4);

            ctx.strokeStyle = '#1e1e28';
            ctx.lineWidth   = 1;
            ctx.beginPath();
            ctx.moveTo(0, rowY + rowH);
            ctx.lineTo(W, rowY + rowH);
            ctx.stroke();
        }

        const phase   = gaitController.time % 1.0;
        const markerX = 32 + phase * (W - 32);
        ctx.strokeStyle = 'rgba(255,255,255,0.25)';
        ctx.lineWidth   = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(markerX, 0);
        ctx.lineTo(markerX, H);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // ---- Training controls ----

    toggle() {
        this.isTraining = !this.isTraining;
        const badge = document.getElementById('train-status-badge');
        const btn   = document.getElementById('btn-train-start');
        if (this.isTraining) {
            badge.className = 'badge training';
            badge.innerText = 'TRAINING';
            btn.innerText   = 'Pause Training';
            btn.className   = 'btn secondary-btn';
        } else {
            badge.className = 'badge idle';
            badge.innerText = 'PAUSED';
            btn.innerText   = 'Resume Training';
            btn.className   = 'btn primary-btn';
        }
    }

    setSpeed(speed) {
        this.speedMultiplier = parseInt(speed);
        ['1x', '5x', '25x', 'max'].forEach(b => {
            const el = document.getElementById(`speed-${b}`);
            if (el) el.classList.remove('active');
        });
        const activeId = speed === 100 ? 'speed-max' : `speed-${speed}x`;
        const el = document.getElementById(activeId);
        if (el) el.classList.add('active');
    }

    setNNMode(enabled) {
        this.useNN = enabled;
        this.reset();
    }

    updateTerrain(type) {
        this.terrainType = type;
        if (type === 'soil') this.terrainFriction = 1.0;
        else if (type === 'sand')  this.terrainFriction = 0.6;
        else if (type === 'rock')  this.terrainFriction = 1.4;
        // Ensure terrain has realistic displacement for training
        window._magiTerrainAmplitude = Math.max(window._magiTerrainAmplitude || 0, 15);
        if (this.visualizer) {
            this.visualizer.setTerrainType(type);
            this.visualizer.setTerrainRoughness(window._magiTerrainAmplitude);
        }
    }

    // ---- Core training loop ----

    update(dt) {
        if (!this.isTraining) return;
        const loops = this.speedMultiplier;
        const simDt = 0.016;
        for (let l = 0; l < loops; l++) {
            this._evaluateStep(simDt);
        }
    }

    _evaluateStep(dt) {
        if (this.agentTime === 0) {
            // Assign a random target direction
            const angle = Math.random() * Math.PI * 2;
            this.targetVx = Math.cos(angle) * 40; // mm/s
            this.targetVz = Math.sin(angle) * 40; // mm/s
            this.targetYaw = 0; // Keeping simple: target is a velocity vector
            this.agentStartX = this.globalX;
            this.agentStartZ = this.globalZ;
        }

        this.agentTime += dt;
        const agent = this.population[this.currentAgentIdx];
        const t = this.agentTime;

        // Decode policy with target inputs
        const phase   = (t * 1.5) % 1.0;
        const decoded = agent.decode(phase, 0, 0, 100, this.terrainFriction, this.targetVx, this.targetVz, this.targetYaw);

        // Phase quality score — reward trot-like alternating diagonal patterns
        const g = decoded.phaseOffsets;
        const d01 = Math.abs(((g[0] - g[1] + 1) % 1) - 0.5);  // FR-FL should be ~0.5 apart
        const d02 = Math.abs(((g[0] - g[2] + 1) % 1) - 0.5);  // FR-BR should be ~0.5 apart
        const d03 = Math.abs(((g[0] - g[3] + 1) % 1));         // FR-BL should be ~0 (same phase for trot)
        const phaseScore = Math.max(0, 1 - (d01 + d02 + d03) / 1.5);

        // Velocity from policy quality and requested strides
        const avgStrideX = decoded.strideX.reduce((a, b) => a + b, 0) / 4;
        const avgStrideZ = decoded.strideZ.reduce((a, b) => a + b, 0) / 4;
        
        // Kinematic translation of stride into body velocity
        const freq = 1.5; // Hz
        const baseStride = 60 * this.terrainFriction * (1 + this.curriculumLevel * 0.1);
        const maxVel = baseStride * freq;
        
        const bodyVx = phaseScore * avgStrideX * maxVel;
        const bodyVz = phaseScore * avgStrideZ * maxVel;

        // Differential turning
        const leftStrideX = (decoded.strideX[1] + decoded.strideX[3]) / 2;
        const rightStrideX = (decoded.strideX[0] + decoded.strideX[2]) / 2;
        const turnRate = (leftStrideX - rightStrideX) * 20; // deg/s

        // Move robot through world
        this.globalX += bodyVx * dt;
        this.globalZ += bodyVz * dt;
        this.bodyYaw = (this.bodyYaw || 0) + turnRate * dt;

        // Step height from decoded policy
        const stepHeight = 30 * (1 + decoded.stepDur * 0.3);

        // Spider-leg neutral positions (wide lateral spread)
        const neutralFeet = [
            { x: this.globalX + 120, z: this.globalZ + 140 },   // FR
            { x: this.globalX + 120, z: this.globalZ - 140 },   // FL
            { x: this.globalX - 120, z: this.globalZ + 140 },   // BR
            { x: this.globalX - 120, z: this.globalZ - 140 },   // BL
        ];

        const feet = [];
        const swingFlags = [];
        let frontGroundY = 0, backGroundY = 0;
        let rightGroundY = 0, leftGroundY = 0;

        for (let i = 0; i < 4; i++) {
            const legPhaseRaw = (phase + decoded.phaseOffsets[i] + 1) % 1.0;
            const isSwing     = legPhaseRaw < 0.35;
            swingFlags.push(isSwing);

            const legStrideX = baseStride * decoded.strideX[i];
            const legStrideZ = baseStride * decoded.strideZ[i];
            const nf = neutralFeet[i];

            const foot = { x: nf.x, y: 0, z: nf.z };

            // 2D position (swing/stance)
            if (isSwing) {
                const sp = legPhaseRaw / 0.35;
                foot.x = nf.x - legStrideX / 2 + legStrideX * sp;
                foot.z = nf.z - legStrideZ / 2 + legStrideZ * sp;
            } else {
                const sp = (legPhaseRaw - 0.35) / 0.65;
                foot.x = nf.x + legStrideX / 2 - legStrideX * sp;
                foot.z = nf.z + legStrideZ / 2 - legStrideZ * sp;
            }

            // Ground height at this foot position
            const groundY = window.getMagiTerrainHeight
                ? window.getMagiTerrainHeight(foot.x, foot.z, this.terrainType)
                : -120;

            if (isSwing) {
                const sp = legPhaseRaw / 0.35;
                foot.y = groundY + bezierStepHeight(sp, stepHeight);
            } else {
                foot.y = groundY;
            }

            feet.push(foot);

            // Accumulate for body posture
            if (i < 2) frontGroundY += groundY; else backGroundY += groundY;
            if (i === 0 || i === 2) rightGroundY += groundY; else leftGroundY += groundY;
        }

        // Body posture from terrain slope
        const physPitch = Math.atan2((frontGroundY - backGroundY) / 2, 240) * 180 / Math.PI;
        const physRoll  = Math.atan2((rightGroundY - leftGroundY) / 2, 280) * 180 / Math.PI;

        // NN can add active correction/sway
        const bodyRot = {
            pitch: physPitch + Math.sin(t * Math.PI * 3) * decoded.pitchAmp * 8,
            roll:  physRoll  + Math.cos(t * Math.PI * 3) * decoded.rollAmp  * 6,
            yaw:   0
        };

        // Body height
        const avgGroundY = (frontGroundY + backGroundY) / 4;
        const bodyHeightTarget = 100 + decoded.heightBias * 25;
        const bodyPos = {
            x: this.globalX,
            y: avgGroundY + bodyHeightTarget,
            z: this.globalZ
        };

        // Render (rate-limited for fast speeds)
        let jointAngles = null;
        if (this.speedMultiplier <= 5 || Math.random() < 0.05) {
            if (this.visualizer) {
                jointAngles = this.visualizer.updateRobotPose(bodyPos, bodyRot, feet, swingFlags);
            }
        }

        // ---- Fitness computation ----

        // Stability penalty
        const pitchPen = Math.abs(bodyRot.pitch) * 2.0;
        const rollPen  = Math.abs(bodyRot.roll)  * 2.0;

        // Height penalty (prefer natural spider height ~100mm)
        const heightPen = Math.abs(bodyHeightTarget - 100) * 4;

        // Energy penalty
        let energyPen = 0;
        if (jointAngles && this._prevJoints) {
            for (let i = 0; i < 4; i++) {
                if (jointAngles[i] && this._prevJoints[i]) {
                    energyPen += Math.pow(jointAngles[i].coxa  - this._prevJoints[i].coxa,  2);
                    energyPen += Math.pow(jointAngles[i].femur - this._prevJoints[i].femur, 2);
                    energyPen += Math.pow(jointAngles[i].tibia - this._prevJoints[i].tibia, 2);
                }
            }
            energyPen *= 0.05;
        }
        if (jointAngles) this._prevJoints = jointAngles;

        // Symmetry reward
        const symScore = 1 - Math.abs(g[0] - g[1]) * 0.5 - Math.abs(g[2] - g[3]) * 0.5;

        // Alignment with target direction
        const dotProd = bodyVx * this.targetVx + bodyVz * this.targetVz;
        const targetMag = Math.sqrt(this.targetVx * this.targetVx + this.targetVz * this.targetVz) + 0.001;
        const distReward = (dotProd / targetMag) * 1.5;

        const yawError = Math.abs((this.bodyYaw || 0) - this.targetYaw);
        const yawPen = yawError * 0.5;

        // Cumulative fitness
        const stepReward = (distReward
                         - pitchPen
                         - rollPen
                         - yawPen
                         - heightPen
                         - energyPen
                         + Math.max(0, symScore) * 4
                         + phaseScore * 5) * dt;
        agent.fitness += stepReward;

        // Live scores
        const currentSpeed = Math.sqrt(bodyVx * bodyVx + bodyVz * bodyVz);
        this.liveScores = {
            velocity:  currentSpeed.toFixed(1),
            stability: (pitchPen + rollPen + yawPen).toFixed(2),
            energy:    energyPen.toFixed(2),
            symmetry:  symScore.toFixed(2)
        };
        this._updateLiveScores();

        // Agent done?
        if (this.agentTime >= this.maxAgentTime) {
            // Calculate actual distance traveled in target direction
            const dx = this.globalX - this.agentStartX;
            const dz = this.globalZ - this.agentStartZ;
            const distInTarget = (dx * this.targetVx + dz * this.targetVz) / targetMag;
            
            // Bonus for distance
            agent.fitness += distInTarget * 0.05;
            agent.fitness = Math.max(0.01, agent.fitness);

            this.currentAgentIdx++;
            this.agentTime   = 0;
            this._prevJoints = null;
            this.bodyYaw = 0; // reset yaw for next agent

            if (this.currentAgentIdx >= this.popSize) {
                this._nextGeneration();
            }

            // Update UI
            const s = id => document.getElementById(id);
            if (s('val-gen'))          s('val-gen').innerText         = this.generation;
            if (s('val-best-id'))      s('val-best-id').innerText    = `#${this.currentAgentIdx + 1}`;
            if (s('val-best-fitness')) s('val-best-fitness').innerText = this.bestFitness.toFixed(2);
            if (s('val-avg-vel'))      s('val-avg-vel').innerText    = `${currentSpeed.toFixed(1)} mm/s`;

            if (this.bestGenes) this._drawGenes(this.bestGenes);
        }
    }

    _updateLiveScores() {
        const ls = this.liveScores;
        const s  = id => document.getElementById(id);
        if (s('live-vel'))  s('live-vel').innerText  = ls.velocity + ' mm/s';
        if (s('live-stab')) s('live-stab').innerText = ls.stability;
        if (s('live-nrg'))  s('live-nrg').innerText  = ls.energy;
        if (s('live-sym'))  s('live-sym').innerText  = ls.symmetry;

        const pct = (this.currentAgentIdx / this.popSize * 100).toFixed(0);
        const bar = document.getElementById('agent-progress-bar');
        if (bar) bar.style.width = pct + '%';
        const lbl = document.getElementById('agent-progress-lbl');
        if (lbl) lbl.innerText = `Agent ${this.currentAgentIdx + 1} / ${this.popSize}`;
    }

    _nextGeneration() {
        this.generation++;
        let totalFitness = 0, maxFit = -Infinity, minFit = Infinity;
        let bestAgent = this.population[0];

        this.population.forEach(agent => {
            totalFitness += agent.fitness;
            if (agent.fitness > maxFit) { maxFit = agent.fitness; bestAgent = agent; }
            if (agent.fitness < minFit) minFit = agent.fitness;
        });

        const avgFit = totalFitness / this.popSize;

        if (maxFit > this.bestFitness) {
            this.bestFitness      = maxFit;
            this.bestGenes        = bestAgent.getGenes().slice();
            this.allTimeBestAgent = bestAgent;
        }

        // Curriculum
        if (avgFit > this.curriculumLevel * 50 && this.curriculumLevel < 5) {
            this.curriculumLevel++;
            this._logCurriculum();
        }

        // History
        this.historyGen.push(this.generation);
        this.historyBest.push(maxFit.toFixed(2));
        this.historyAvg.push(avgFit.toFixed(2));
        this.historyMin.push(minFit.toFixed(2));

        // Chart
        if (this.chart) {
            this.chart.data.labels           = this.historyGen;
            this.chart.data.datasets[0].data = this.historyBest;
            this.chart.data.datasets[1].data = this.historyAvg;
            this.chart.data.datasets[2].data = this.historyMin;
            this.chart.update('none');
        }

        // Selection + Crossover + Mutation
        this.population.sort((a, b) => b.fitness - a.fitness);
        const eliteN  = Math.max(2, Math.floor(this.popSize * 0.2));
        const nextPop = [];

        for (let i = 0; i < eliteN; i++) {
            const elite = new RLAgent(i + 1, this.useNN);
            if (this.useNN) elite.policy = this.population[i].policy.clone();
            else            elite.genes  = [...this.population[i].genes];
            elite.fitness = 0;
            nextPop.push(elite);
        }

        for (let i = 0; i < this.popSize - eliteN; i++) {
            const pA    = this._tournamentSelect();
            const pB    = this._tournamentSelect();
            const child = pA.crossover(pB);
            child.mutate(this.mutationRate);
            child.id      = eliteN + i + 1;
            child.fitness = 0;
            nextPop.push(child);
        }

        this.population      = nextPop;
        this.currentAgentIdx = 0;
    }

    _tournamentSelect(k = 4) {
        let best = null;
        for (let i = 0; i < k; i++) {
            const c = this.population[Math.floor(Math.random() * this.population.length)];
            if (!best || c.fitness > best.fitness) best = c;
        }
        return best;
    }

    _logCurriculum() {
        const el = document.getElementById('curriculum-badge');
        if (el) {
            el.innerText    = `LVL ${this.curriculumLevel}`;
            el.style.color  = '#e67e22';
            setTimeout(() => { if (el) el.style.color = ''; }, 2000);
        }
    }

    exportBestGenome() {
        if (!this.bestGenes) { alert('No best genome yet — run training first.'); return; }
        const payload = {
            generation: this.generation,
            fitness:    this.bestFitness.toFixed(3),
            mode:       this.useNN ? 'NNPolicy' : 'DirectGene',
            genes:      this.bestGenes.map(g => parseFloat(g.toFixed(5)))
        };
        const json = JSON.stringify(payload, null, 2);
        try {
            navigator.clipboard.writeText(json)
                .then(() => alert('✅ Best genome JSON copied to clipboard!'))
                .catch(() => prompt('Copy genome JSON:', json));
        } catch (e) {
            prompt('Copy genome JSON:', json);
        }
    }

    reset() {
        this.initPopulation();
        this.isTraining  = false;
        this.globalX     = 0;
        this.globalZ     = 0;
        this.startX      = 0;
        this._prevJoints = null;
        this._sendNeutralPose();

        if (this.chart) {
            this.chart.data.labels           = [];
            this.chart.data.datasets[0].data = [];
            this.chart.data.datasets[1].data = [];
            this.chart.data.datasets[2].data = [];
            this.chart.update();
        }
        this._drawGenes(Array(16).fill(0));
        
        const s = id => document.getElementById(id);
        if (s('val-gen'))          s('val-gen').innerText         = '0';
        if (s('val-best-id'))      s('val-best-id').innerText    = '-';
        if (s('val-best-fitness')) s('val-best-fitness').innerText = '0.00';
        if (s('val-avg-vel'))      s('val-avg-vel').innerText    = '0.0 mm/s';

        const badge = document.getElementById('train-status-badge');
        const btn   = document.getElementById('btn-train-start');
        if (badge) { badge.className = 'badge idle'; badge.innerText = 'IDLE'; }
        if (btn)   { btn.innerText = 'Start RL Training'; btn.className = 'btn primary-btn'; }
    }

    _sendNeutralPose() {
        if (!this.visualizer) return;
        // Spider stance: body at height 100 above ground (-120), feet touching ground
        const groundY = -120;
        const bodyPos = { x: 0, y: groundY + 100, z: 0 };
        const bodyRot = { pitch: 0, roll: 0, yaw: 0 };
        const feet = [
            { x:  120, y: groundY, z:  140 },  // FR
            { x:  120, y: groundY, z: -140 },  // FL
            { x: -120, y: groundY, z:  140 },  // BR
            { x: -120, y: groundY, z: -140 },  // BL
        ];
        this.visualizer.updateRobotPose(bodyPos, bodyRot, feet);
    }
}
