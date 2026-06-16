/* ============================================================
   MAGI Quadruped Visualiser — robot.js  (v7.0 - Three.js WebGL)
   Spider/Crab Configuration — Legs point OUTWARD from body
   ============================================================

   COORDINATE SYSTEM (Three.js):
   +X = Forward (front of robot)
   +Y = Up
   +Z = Right (when looking from behind)

   ROBOT STRUCTURE (Spider/Crab):
   - Rectangular chassis, wider than it is tall
   - 4 legs mounted at corners, each pointing OUTWARD (laterally)
   - Each leg: Coxa (horizontal rotation) → Femur (vertical swing) → Tibia (knee bend)
   - Legs spread outward like a spider, NOT underneath like a dog

   LEG ORDER: 0=FR, 1=FL, 2=BR, 3=BL
   FR/BR are on the RIGHT side (+Z), FL/BL are on the LEFT side (-Z)
   ============================================================ */

const LINK_COXA  = 30;   // mm — short hip segment
const LINK_FEMUR = 90;   // mm — upper leg
const LINK_TIBIA = 90;   // mm — lower leg

const LEG_META = [
    { name: 'FR', side:  1 },  // +Z side (right)
    { name: 'FL', side: -1 },  // -Z side (left)
    { name: 'BR', side:  1 },  // +Z side (right)
    { name: 'BL', side: -1 },  // -Z side (left)
];

/* ---- Analytical 3-DOF IK Solver ----
   Input: foot position relative to shoulder, in leg-local frame:
     outward = distance laterally from shoulder (always positive)
     forward = distance forward/back from shoulder
     down    = distance below shoulder (positive = below)
   Output: { coxa, femur, tibia } in degrees
*/
function solveIK(outward, forward, down) {
    // Coxa: rotation in the horizontal plane
    const coxaRad = Math.atan2(forward, outward);
    const coxaDeg = coxaRad * 180 / Math.PI;

    // Project into the leg's sagittal plane (perpendicular to body)
    const r  = Math.sqrt(outward * outward + forward * forward);
    const xp = r - LINK_COXA;  // horizontal distance after coxa
    const zp = down;            // vertical distance (positive = down)

    let Ld = Math.sqrt(xp * xp + zp * zp);
    const maxR = (LINK_FEMUR + LINK_TIBIA) * 0.98;
    const minR = Math.abs(LINK_FEMUR - LINK_TIBIA) + 2;
    if (Ld > maxR) { const s = maxR / Ld; Ld = maxR; }
    if (Ld < minR) { Ld = minR; }

    // Tibia angle (knee) — law of cosines
    const cosKnee = (LINK_FEMUR * LINK_FEMUR + LINK_TIBIA * LINK_TIBIA - Ld * Ld) 
                  / (2 * LINK_FEMUR * LINK_TIBIA);
    const tibiaDeg = Math.acos(Math.min(Math.max(cosKnee, -1), 1)) * 180 / Math.PI;

    // Femur angle (hip vertical swing)
    const alpha = Math.atan2(zp, xp);
    const cosHip = (LINK_FEMUR * LINK_FEMUR + Ld * Ld - LINK_TIBIA * LINK_TIBIA) 
                 / (2 * LINK_FEMUR * Ld);
    const beta = Math.acos(Math.min(Math.max(cosHip, -1), 1));
    const femurDeg = (alpha - beta) * 180 / Math.PI;

    return { coxa: coxaDeg, femur: femurDeg, tibia: tibiaDeg };
}

// Global Terrain Function for Physics Contact
// Uses multi-octave procedural noise for realistic terrain
window._magiTerrainAmplitude = 0;  // controlled by slider (0..25)

window.getMagiTerrainHeight = function(x, z, terrainType = 'soil') {
    const groundY = -120;
    const amp = window._magiTerrainAmplitude || 0;
    if (amp < 0.5) return groundY; // flat floor when roughness slider = 0

    // Hash-based pseudo-noise (deterministic, no library needed)
    function hash(px, pz) {
        let n = Math.sin(px * 127.1 + pz * 311.7) * 43758.5453;
        return n - Math.floor(n);
    }
    function smoothNoise(px, pz) {
        const ix = Math.floor(px), iz = Math.floor(pz);
        const fx = px - ix, fz = pz - iz;
        const ux = fx * fx * (3 - 2 * fx);
        const uz = fz * fz * (3 - 2 * fz);
        const a = hash(ix, iz);
        const b = hash(ix + 1, iz);
        const c = hash(ix, iz + 1);
        const d = hash(ix + 1, iz + 1);
        return a + (b - a) * ux + (c - a) * uz + (a - b - c + d) * ux * uz;
    }
    function fbm(px, pz, octaves) {
        let v = 0, a = 0.5, f = 1.0;
        for (let i = 0; i < octaves; i++) {
            v += a * (smoothNoise(px * f, pz * f) - 0.5);
            f *= 2.0;
            a *= 0.5;
        }
        return v;
    }

    const scale = amp;  // 0..25 mm amplitude

    if (terrainType === 'sand') {
        // Sand: smooth dunes with gentle ridges
        const dune = fbm(x * 0.006, z * 0.005, 4) * scale * 1.2;
        const ripple = Math.sin(x * 0.04 + z * 0.02) * scale * 0.15;
        return groundY + dune + ripple;
    }
    if (terrainType === 'rock') {
        // Rock: sharp, craggy features with high-frequency detail
        const base = fbm(x * 0.008, z * 0.007, 5) * scale * 1.5;
        const crag = fbm(x * 0.025, z * 0.03, 3) * scale * 0.6;
        return groundY + base + crag;
    }
    // Soil: gentle rolling hills with occasional bumps
    const hill = fbm(x * 0.005, z * 0.004, 4) * scale;
    const bump = fbm(x * 0.018, z * 0.015, 2) * scale * 0.25;
    return groundY + hill + bump;
};

class MagiVisualizer {
    constructor(containerId, isMinimal = false) {
        this.containerId = containerId;
        this.isMinimal = isMinimal;
        this.ready = false;
        
        this.terrainType = 'soil';
        this.terrainAmplitude = 0;

        /* Shoulder mount points in chassis-local coordinates:
           FR = front-right, FL = front-left, BR = back-right, BL = back-left */
        this.shoulders = [
            { x:  80, y: 0, z:  40 },   // FR: front, right
            { x:  80, y: 0, z: -40 },   // FL: front, left
            { x: -80, y: 0, z:  40 },   // BR: back, right
            { x: -80, y: 0, z: -40 },   // BL: back, left
        ];

        // Legacy compatibility
        this.gridOffset = 0;
        this.supportPolyLine = { visible: true };
        this.comIndicator = { visible: true };
        this.comLine = { visible: true };
        this.footRings = [0, 1, 2, 3].map(() => ({ material: { opacity: 0 } }));

        this._boot();
    }

    _boot() {
        const container = document.getElementById(this.containerId);
        if (!container) return;
        this.container = container;

        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a10);
        this.scene.fog = new THREE.FogExp2(0x0a0a10, 0.0012);

        // Camera
        const W = container.clientWidth || 800;
        const H = container.clientHeight || 600;
        this.camera = new THREE.PerspectiveCamera(50, W / H, 1, 12000);
        this.camera.position.set(250, 180, 250);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        this.renderer.setSize(W, H);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.2;
        container.innerHTML = '';
        container.appendChild(this.renderer.domElement);

        // OrbitControls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;
        this.controls.maxPolarAngle = Math.PI / 2 + 0.15;
        this.controls.target.set(0, -60, 0);

        // Lighting
        const ambient = new THREE.AmbientLight(0xb0c4de, 0.4);
        this.scene.add(ambient);

        const hemiLight = new THREE.HemisphereLight(0x8899bb, 0x443322, 0.3);
        this.scene.add(hemiLight);

        const dirLight = new THREE.DirectionalLight(0xffeedd, 1.0);
        dirLight.position.set(300, 500, -200);
        dirLight.castShadow = true;
        dirLight.shadow.mapSize.width = 2048;
        dirLight.shadow.mapSize.height = 2048;
        dirLight.shadow.camera.near = 10;
        dirLight.shadow.camera.far = 2000;
        dirLight.shadow.camera.left = -600;
        dirLight.shadow.camera.right = 600;
        dirLight.shadow.camera.top = 600;
        dirLight.shadow.camera.bottom = -600;
        this.scene.add(dirLight);

        // Terrain
        this._buildTerrain();

        // Robot
        this.robotGroup = new THREE.Group();
        this.scene.add(this.robotGroup);
        this._buildRobotMeshes();

        // Scenery
        this._buildScenery();

        // Resize
        this._onResize = () => {
            if (this.container && this.container.clientWidth > 0) {
                const w = this.container.clientWidth;
                const h = this.container.clientHeight;
                this.camera.aspect = w / h;
                this.camera.updateProjectionMatrix();
                this.renderer.setSize(w, h);
            }
        };
        window.addEventListener('resize', this._onResize);

        this.ready = true;
        this._animate();
    }

    _buildTerrain() {
        this.terrainSegs = 80;
        this.terrainGeo = new THREE.PlaneGeometry(3000, 3000, this.terrainSegs, this.terrainSegs);
        this.terrainMat = new THREE.MeshStandardMaterial({
            color: 0x2a3025,
            roughness: 0.92,
            metalness: 0.0,
            flatShading: true,
        });
        this.terrainMesh = new THREE.Mesh(this.terrainGeo, this.terrainMat);
        this.terrainMesh.rotation.x = -Math.PI / 2;
        this.terrainMesh.receiveShadow = true;
        this.scene.add(this.terrainMesh);
    }

    _buildRobotMeshes() {
        // ----- Chassis -----
        const chassisGeo = new THREE.BoxGeometry(160, 16, 80);
        const chassisMat = new THREE.MeshStandardMaterial({ 
            color: 0x1a1a24, roughness: 0.4, metalness: 0.2 
        });
        this.chassisMesh = new THREE.Mesh(chassisGeo, chassisMat);
        this.chassisMesh.castShadow = true;
        this.chassisMesh.receiveShadow = true;
        this.robotGroup.add(this.chassisMesh);

        // Orange accent stripe
        const stripeGeo = new THREE.BoxGeometry(162, 3, 82);
        const stripeMat = new THREE.MeshStandardMaterial({ 
            color: 0xe67e22, roughness: 0.3, emissive: 0x331800 
        });
        const stripe = new THREE.Mesh(stripeGeo, stripeMat);
        stripe.position.y = 9;
        this.chassisMesh.add(stripe);

        // Sensor "eye" at front
        const eyeGeo = new THREE.SphereGeometry(5, 16, 16);
        const eyeMat = new THREE.MeshStandardMaterial({ 
            color: 0x2ecc71, emissive: 0x0a4020, roughness: 0.2 
        });
        const eye = new THREE.Mesh(eyeGeo, eyeMat);
        eye.position.set(82, 5, 0);
        this.chassisMesh.add(eye);

        // ----- Legs -----
        this.legMeshes = [];
        const matJoint  = new THREE.MeshStandardMaterial({ color: 0x333342, roughness: 0.6 });
        const matCoxa   = new THREE.MeshStandardMaterial({ color: 0x444455, roughness: 0.5 });
        const matFemur  = new THREE.MeshStandardMaterial({ color: 0xe67e22, roughness: 0.4 });
        const matTibia  = new THREE.MeshStandardMaterial({ color: 0x999999, roughness: 0.3, metalness: 0.4 });
        const matFoot   = new THREE.MeshStandardMaterial({ color: 0x2ecc71, roughness: 0.8 });

        for (let i = 0; i < 4; i++) {
            const meta = LEG_META[i];
            const sh = this.shoulders[i];

            // Leg root — attached to chassis at shoulder position
            const legRoot = new THREE.Group();
            legRoot.position.set(sh.x, sh.y, sh.z);
            this.chassisMesh.add(legRoot);

            // Shoulder joint ball
            const shoulderBall = new THREE.Mesh(new THREE.SphereGeometry(8, 12, 12), matJoint);
            legRoot.add(shoulderBall);

            // Coxa group — rotates around Y axis (yaw / horizontal plane)
            // Default orientation: +Z direction (outward for right side legs)
            // For left side legs, we'll flip the initial rotation
            const coxaGroup = new THREE.Group();
            // Base rotation: point the coxa outward
            coxaGroup.rotation.y = meta.side > 0 ? 0 : Math.PI;
            legRoot.add(coxaGroup);

            // Coxa mesh — cylinder along +Z
            const coxaCyl = new THREE.Mesh(
                new THREE.CylinderGeometry(4, 4, LINK_COXA, 8),
                matCoxa
            );
            coxaCyl.rotation.x = Math.PI / 2;  // Lay along +Z
            coxaCyl.position.z = LINK_COXA / 2;
            coxaGroup.add(coxaCyl);

            // Elbow joint ball at end of coxa
            const elbowBall = new THREE.Mesh(new THREE.SphereGeometry(7, 12, 12), matFemur);
            elbowBall.position.z = LINK_COXA;
            coxaGroup.add(elbowBall);

            // Femur group — pivots at end of coxa, rotates around X axis (pitch / vertical)
            const femurGroup = new THREE.Group();
            femurGroup.position.z = LINK_COXA;
            coxaGroup.add(femurGroup);

            // Femur mesh — cylinder along +Z (will be rotated by femur angle)
            const femurCyl = new THREE.Mesh(
                new THREE.CylinderGeometry(5, 3.5, LINK_FEMUR, 8),
                matFemur
            );
            femurCyl.rotation.x = Math.PI / 2;
            femurCyl.position.z = LINK_FEMUR / 2;
            femurCyl.castShadow = true;
            femurGroup.add(femurCyl);

            // Knee joint
            const kneeBall = new THREE.Mesh(new THREE.SphereGeometry(6, 12, 12), matTibia);
            kneeBall.position.z = LINK_FEMUR;
            femurGroup.add(kneeBall);

            // Tibia group — pivots at end of femur
            const tibiaGroup = new THREE.Group();
            tibiaGroup.position.z = LINK_FEMUR;
            femurGroup.add(tibiaGroup);

            // Tibia mesh
            const tibiaCyl = new THREE.Mesh(
                new THREE.CylinderGeometry(3, 2, LINK_TIBIA, 8),
                matTibia
            );
            tibiaCyl.rotation.x = Math.PI / 2;
            tibiaCyl.position.z = LINK_TIBIA / 2;
            tibiaCyl.castShadow = true;
            tibiaGroup.add(tibiaCyl);

            // Foot ball
            const footBall = new THREE.Mesh(new THREE.SphereGeometry(6, 12, 12), matFoot.clone());
            footBall.position.z = LINK_TIBIA;
            footBall.castShadow = true;
            tibiaGroup.add(footBall);

            this.legMeshes.push({
                legRoot, coxaGroup, femurGroup, tibiaGroup, footBall, meta
            });
        }
    }

    _buildScenery() {
        this.sceneryGroup = new THREE.Group();
        this.scene.add(this.sceneryGroup);
        this.sceneryItems = [];

        // Materials
        const trunkMat    = new THREE.MeshStandardMaterial({ color: 0x5d4037, roughness: 0.9 });
        const treeMat     = new THREE.MeshStandardMaterial({ color: 0x2d8a4e, roughness: 0.85, flatShading: true });
        const treeMat2    = new THREE.MeshStandardMaterial({ color: 0x1e6b3a, roughness: 0.85, flatShading: true });
        const treeMat3    = new THREE.MeshStandardMaterial({ color: 0x3a9d5e, roughness: 0.80, flatShading: true });
        const rockMat     = new THREE.MeshStandardMaterial({ color: 0x6b7b7d, roughness: 0.85, flatShading: true });
        const rockMat2    = new THREE.MeshStandardMaterial({ color: 0x8a9090, roughness: 0.8, flatShading: true });
        const rockDark    = new THREE.MeshStandardMaterial({ color: 0x4a5555, roughness: 0.9, flatShading: true });
        const cropMat     = new THREE.MeshStandardMaterial({ color: 0x4a8a2a, roughness: 0.9, flatShading: true });
        const cropMat2    = new THREE.MeshStandardMaterial({ color: 0x5a9a3a, roughness: 0.85, flatShading: true });
        const grassMat    = new THREE.MeshStandardMaterial({ color: 0x4a7a3f, roughness: 1.0, flatShading: true });
        const flowerMat   = new THREE.MeshStandardMaterial({ color: 0xe8d44d, roughness: 0.7, emissive: 0x332800 });
        const soilPatchMat = new THREE.MeshStandardMaterial({ color: 0x3a2e1a, roughness: 1.0, flatShading: true });

        const rng = (a, b) => a + Math.random() * (b - a);
        const FIELD = 2000; // half-extent of scenery field

        // ---- TREES (large, scattered) ----
        for (let i = 0; i < 60; i++) {
            const x = rng(-FIELD, FIELD);
            const z = rng(-FIELD, FIELD);
            // Keep trees away from the robot's direct path center
            if (Math.abs(z) < 80 && Math.abs(x) < 200) continue;

            const grp = new THREE.Group();
            grp.userData = { baseX: x, baseZ: z };

            const h = rng(60, 160);
            const trunkH = h * rng(0.3, 0.45);
            const trunkR = rng(4, 9);
            const trunk = new THREE.Mesh(
                new THREE.CylinderGeometry(trunkR * 0.7, trunkR, trunkH, 6),
                trunkMat
            );
            trunk.position.y = trunkH / 2;
            trunk.castShadow = true;
            grp.add(trunk);

            // Foliage — 2–3 layers of cones/spheres
            const canopyMats = [treeMat, treeMat2, treeMat3];
            const layers = 2 + Math.floor(Math.random() * 2);
            for (let l = 0; l < layers; l++) {
                const r = h * rng(0.25, 0.5) * (1 - l * 0.2);
                const isRound = Math.random() > 0.5;
                const leafGeo = isRound
                    ? new THREE.DodecahedronGeometry(r, 1)
                    : new THREE.ConeGeometry(r, h * rng(0.3, 0.5), 5 + Math.floor(Math.random() * 3));
                const leaf = new THREE.Mesh(leafGeo, canopyMats[l % canopyMats.length]);
                leaf.position.y = trunkH + h * (0.15 + l * 0.18);
                leaf.rotation.y = Math.random() * Math.PI;
                leaf.castShadow = true;
                grp.add(leaf);
            }

            this.sceneryGroup.add(grp);
            this.sceneryItems.push(grp);
        }

        // ---- CROP PLANTS (the plants the robot inspects — rows of bushes) ----
        for (let row = -6; row <= 6; row++) {
            const rowZ = row * 100 + rng(-15, 15);
            for (let col = -8; col <= 8; col++) {
                if (Math.random() < 0.25) continue; // some gaps
                const x = col * 80 + rng(-20, 20);
                const z = rowZ + rng(-10, 10);

                const grp = new THREE.Group();
                grp.userData = { baseX: x, baseZ: z };

                // Small bush / crop
                const bushH = rng(12, 30);
                const bushR = rng(8, 18);
                const mat = Math.random() > 0.5 ? cropMat : cropMat2;
                const bush = new THREE.Mesh(
                    new THREE.DodecahedronGeometry(bushR, 1),
                    mat
                );
                bush.position.y = bushH;
                bush.scale.y = rng(0.6, 1.0);
                bush.castShadow = true;
                grp.add(bush);

                // Thin stem
                const stem = new THREE.Mesh(
                    new THREE.CylinderGeometry(1.5, 2, bushH, 4),
                    trunkMat
                );
                stem.position.y = bushH / 2;
                grp.add(stem);

                // Occasional flower/leaf marker
                if (Math.random() > 0.7) {
                    const flower = new THREE.Mesh(
                        new THREE.SphereGeometry(3, 8, 8),
                        flowerMat
                    );
                    flower.position.set(rng(-5, 5), bushH + bushR * 0.6, rng(-5, 5));
                    grp.add(flower);
                }

                this.sceneryGroup.add(grp);
                this.sceneryItems.push(grp);
            }
        }

        // ---- BIG ROCKS & BOULDERS ----
        for (let i = 0; i < 80; i++) {
            const x = rng(-FIELD, FIELD);
            const z = rng(-FIELD, FIELD);

            const grp = new THREE.Group();
            grp.userData = { baseX: x, baseZ: z };

            const numRocks = 1 + Math.floor(Math.random() * 4);
            for (let r = 0; r < numRocks; r++) {
                const s = rng(6, 45);
                const mat = Math.random() > 0.6 ? rockMat : (Math.random() > 0.5 ? rockMat2 : rockDark);
                const rock = new THREE.Mesh(
                    new THREE.DodecahedronGeometry(s, Math.random() > 0.5 ? 0 : 1),
                    mat
                );
                rock.position.set(rng(-20, 20), s * 0.35, rng(-20, 20));
                rock.rotation.set(Math.random() * 2, Math.random() * 2, Math.random() * 2);
                rock.castShadow = true;
                rock.receiveShadow = true;
                grp.add(rock);
            }

            this.sceneryGroup.add(grp);
            this.sceneryItems.push(grp);
        }

        // ---- GRASS PATCHES ----
        for (let i = 0; i < 250; i++) {
            const x = rng(-FIELD, FIELD);
            const z = rng(-FIELD, FIELD);

            const grp = new THREE.Group();
            grp.userData = { baseX: x, baseZ: z };

            const blades = 2 + Math.floor(Math.random() * 4);
            for (let b = 0; b < blades; b++) {
                const blade = new THREE.Mesh(
                    new THREE.ConeGeometry(rng(1.5, 4), rng(8, 25), 3),
                    grassMat
                );
                blade.position.set(rng(-6, 6), rng(3, 10), rng(-6, 6));
                blade.rotation.z = rng(-0.2, 0.2);
                grp.add(blade);
            }

            this.sceneryGroup.add(grp);
            this.sceneryItems.push(grp);
        }

        // ---- SOIL PATCHES (bare ground texture variation) ----
        for (let i = 0; i < 40; i++) {
            const x = rng(-FIELD, FIELD);
            const z = rng(-FIELD, FIELD);

            const grp = new THREE.Group();
            grp.userData = { baseX: x, baseZ: z };

            const patch = new THREE.Mesh(
                new THREE.CircleGeometry(rng(15, 40), 6),
                soilPatchMat
            );
            patch.rotation.x = -Math.PI / 2;
            patch.position.y = 0.5; // just above terrain surface
            grp.add(patch);

            this.sceneryGroup.add(grp);
            this.sceneryItems.push(grp);
        }
    }

    _updateTerrain(robotX, robotZ) {
        const snapX = Math.round(robotX / 50) * 50;
        const snapZ = Math.round(robotZ / 50) * 50;
        this.terrainMesh.position.set(snapX, 0, snapZ);

        // Color
        if (this.terrainType === 'sand') this.terrainMat.color.setHex(0x9e8a5e);
        else if (this.terrainType === 'rock') this.terrainMat.color.setHex(0x5a5a5a);
        else this.terrainMat.color.setHex(0x2a3025);

        // Vertex displacement
        const posAttr = this.terrainGeo.attributes.position;
        for (let i = 0; i < posAttr.count; i++) {
            const localX = posAttr.getX(i);
            const localY = posAttr.getY(i);
            const worldX = localX + snapX;
            const worldZ = -localY + snapZ; // PlaneGeometry XY → world XZ (rotated -90 on X)
            const h = window.getMagiTerrainHeight(worldX, worldZ, this.terrainType);
            posAttr.setZ(i, h);
        }
        posAttr.needsUpdate = true;
        this.terrainGeo.computeVertexNormals();

        // Wrap scenery in both X and Z
        const wrapX = 4000;
        const wrapZ = 4000;
        for (const item of this.sceneryItems) {
            let dx = (item.userData.baseX - robotX) % wrapX;
            if (dx < -wrapX / 2) dx += wrapX;
            else if (dx > wrapX / 2) dx -= wrapX;

            let dz = (item.userData.baseZ - robotZ) % wrapZ;
            if (dz < -wrapZ / 2) dz += wrapZ;
            else if (dz > wrapZ / 2) dz -= wrapZ;

            const wx = robotX + dx;
            const wz = robotZ + dz;
            const h = window.getMagiTerrainHeight(wx, wz, this.terrainType);
            item.position.set(wx, h, wz);
        }
    }

    /* ============================================================
       updateRobotPose — THE CORE RENDERING FUNCTION
       ============================================================
       bodyPos: { x, y, z } — world position of chassis center
       bodyRot: { pitch, roll, yaw } — degrees
       footCoords: Array of 4 { x, y, z } — world position of each foot
       swingFlags: Array of 4 boolean — is leg in swing phase
       
       Returns: Array of 4 { coxa, femur, tibia } angle objects (degrees)
       ============================================================ */
    updateRobotPose(bodyPos, bodyRot, footCoords, swingFlags = []) {
        if (!this.ready) return null;

        // Set body position and rotation
        this.robotGroup.position.set(bodyPos.x, bodyPos.y, bodyPos.z);
        this.robotGroup.rotation.set(
            bodyRot.pitch * Math.PI / 180,
            bodyRot.yaw   * Math.PI / 180,
            bodyRot.roll  * Math.PI / 180,
            'YXZ'
        );

        // Camera follow
        const targetPos = new THREE.Vector3(bodyPos.x, bodyPos.y, bodyPos.z);
        this.controls.target.lerp(targetPos, 0.08);
        this.camera.position.x += (bodyPos.x - 250 - this.camera.position.x) * 0.03;
        this.camera.position.z += (bodyPos.z + 250 - this.camera.position.z) * 0.03;

        // Update terrain around robot
        this._updateTerrain(bodyPos.x, bodyPos.z);

        // Solve IK for each leg
        const jointAngles = [];
        for (let i = 0; i < 4; i++) {
            const meta = LEG_META[i];
            const sh = this.shoulders[i];
            const leg = this.legMeshes[i];

            // Transform foot world position to chassis-local coordinates
            const footW = new THREE.Vector3(footCoords[i].x, footCoords[i].y, footCoords[i].z);
            const footLocal = this.robotGroup.worldToLocal(footW.clone());

            // Offset from shoulder mount
            const dx = footLocal.x - sh.x;  // forward/back from shoulder
            const dy = footLocal.y - sh.y;  // up/down from shoulder
            const dz = footLocal.z - sh.z;  // left/right from shoulder

            // Convert to leg-local frame:
            // "outward" = distance laterally from shoulder (always positive for IK)
            // "forward" = distance forward from shoulder
            // "down"    = distance below shoulder (positive = below)
            const outward = Math.abs(dz);
            const forward = dx;
            const down = -dy;

            const angles = solveIK(outward, forward, down);
            jointAngles.push(angles);

            // ---- Apply to Three.js mesh hierarchy ----

            // Coxa: rotation around Y axis in the horizontal plane
            // The base rotation already points +Z outward. Coxa adds yaw.
            const coxaYaw = angles.coxa * Math.PI / 180;
            leg.coxaGroup.rotation.y = (meta.side > 0 ? 0 : Math.PI) + coxaYaw * meta.side;

            // Femur: rotation around X axis (pitch down)
            // Positive femurDeg means the leg points downward
            const femurRad = angles.femur * Math.PI / 180;
            leg.femurGroup.rotation.x = femurRad;

            // Tibia: rotation around X axis relative to femur
            // tibiaDeg is the interior knee angle. We need supplementary for the bend.
            const tibiaRad = angles.tibia * Math.PI / 180;
            leg.tibiaGroup.rotation.x = Math.PI - tibiaRad;

            // Swing highlight
            const isSwing = swingFlags[i] || false;
            leg.footBall.material.color.setHex(isSwing ? 0xe67e22 : 0x2ecc71);
            leg.footBall.scale.setScalar(isSwing ? 1.4 : 1.0);
        }

        return jointAngles;
    }

    setTerrainRoughness(amplitude) { 
        this.terrainAmplitude = amplitude; 
        window._magiTerrainAmplitude = amplitude;
    }
    setTerrainType(type) { this.terrainType = type; }

    _animate() {
        requestAnimationFrame(() => this._animate());
        if (this.controls) this.controls.update();
        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    }
}
