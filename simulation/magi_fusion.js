/* MAGI Caspar Decision Fusion & Rule Engine - magi_fusion.js */

class CasparDecisionEngine {
    constructor(gaitController) {
        this.gaitController = gaitController;
        
        // Sensor states (inputs)
        this.tof = 600; // mm
        this.gpioTrigger = false;
        this.anomalyScore = 0.12;
        this.scene = "normal";
        this.detections = []; // array of objects {label, confidence}

        // Decision states (outputs)
        this.action = "IDLE";
        this.priority = 0;
        this.reason = "No significant events";
        this.decisionCount = 0;

        // Terminal console reference
        this.consoleElement = null;
        this.prevAction = null;
    }

    setConsole(elId) {
        this.consoleElement = document.getElementById(elId);
    }

    logToConsole(type, msg) {
        if (!this.consoleElement) return;
        
        let dateStr = new Date().toLocaleTimeString();
        let logClass = "log-line";
        if (type === "EMERGENCY") logClass = "log-line emergency";
        else if (type === "ALERT") logClass = "log-line alert";
        else if (type === "SYSTEM") logClass = "log-line system";

        this.consoleElement.innerHTML += `<span class="${logClass}">[${dateStr}] [${type}] ${msg}</span>`;
        // Scroll to bottom
        this.consoleElement.scrollTop = this.consoleElement.scrollHeight;
    }

    // Runs the 6 rules in priority order
    evaluateRules() {
        let action = "IDLE";
        let priority = 0;
        let reason = "No significant events";
        let activeRuleId = "rule-default";

        // Priority 10: ToF Obstacle close
        if (this.tof > 0 && this.tof < 200) {
            action = "EMERGENCY";
            priority = 10;
            reason = `Obstacle detected at ${this.tof} mm`;
            activeRuleId = "rule-1";
        }
        // Priority 9: Hardware GPIO alert
        else if (this.gpioTrigger) {
            action = "EMERGENCY";
            priority = 9;
            reason = "Hardware GPIO emergency trigger received";
            activeRuleId = "rule-2";
        }
        // Priority 8: Person in restricted zone
        else if (this.scene === "restricted_zone" && this.detections.some(d => d.label === "person")) {
            action = "EMERGENCY";
            priority = 8;
            reason = "Intruder: Person in restricted zone";
            activeRuleId = "rule-3";
        }
        // Priority 6: High Anomaly Score
        else if (this.anomalyScore > 0.7) {
            action = "ALERT";
            priority = 6;
            reason = `High scene anomaly index: ${this.anomalyScore.toFixed(2)}`;
            activeRuleId = "rule-4";
        }
        // Priority 5: Scene alert
        else if (this.scene === "emergency" || this.scene === "obstacle_close") {
            action = "ANALYZE";
            priority = 5;
            reason = `Scene classifier flag: ${this.scene}`;
            activeRuleId = "rule-5";
        }
        // Priority 4: Detections present (Target Tracking)
        else if (this.detections.length > 0) {
            action = "TRACK";
            priority = 4;
            let target = this.detections[0];
            reason = `Tracking ${target.label} (conf: ${(target.confidence * 100).toFixed(0)}%)`;
            activeRuleId = "rule-5"; // maps to detection rule
        }

        this.action = action;
        this.priority = priority;
        this.reason = reason;
        this.decisionCount++;

        // Trigger logging on state change
        if (action !== this.prevAction) {
            this.logToConsole(action, `${reason} (Priority: ${priority})`);
            this.prevAction = action;
        }

        // Apply feedback overrides to walking gait
        this.applyLocomotionLoopback();

        // Update pipeline diagram highlights
        this.updatePipelineUI(activeRuleId);
    }

    // Translates the high level decision back into mechanical variables (Gait and posturing)
    applyLocomotionLoopback() {
        if (!this.gaitController) return;

        // Reset overlays to nominal slider configurations first
        const freqSlider = document.getElementById('slider-frequency');
        const heightSlider = document.getElementById('slider-body-z');
        const pitchSlider = document.getElementById('slider-body-pitch');
        const yawSlider = document.getElementById('slider-body-roll'); // roll representing yaw in simplified UI
        
        let targetFreq = parseFloat(freqSlider.value);
        let targetHeight = parseFloat(heightSlider.value);
        let targetPitch = 0;
        
        const badgeState = document.getElementById('val-gait-override-state');

        if (this.action === "EMERGENCY") {
            // Immediate lock and drop
            this.gaitController.setGaitType("stand");
            this.gaitController.bodyPos.y = -60; // drop body to ground to lock structure
            this.gaitController.bodyRot.pitch = 0;
            this.gaitController.bodyRot.roll = 0;
            if (badgeState) {
                badgeState.innerText = "HALTED: emergency lock active";
                badgeState.style.color = "var(--color-red)";
            }
        } 
        else if (this.action === "ALERT") {
            // Slow down and widen stance height to lower Center of Mass (increase static safety margin)
            this.gaitController.setGaitType("crawl");
            this.gaitController.frequency = 0.6; // slow, cautious crawl
            this.gaitController.stride = 40;     // small, careful steps
            this.gaitController.bodyPos.y = -30; // lower chassis to ground
            if (badgeState) {
                badgeState.innerText = "CAUTIOUS CRAWL (low velocity)";
                badgeState.style.color = "var(--color-orange)";
            }
        }
        else if (this.action === "TRACK") {
            // point body yaw/pitch towards target
            this.gaitController.setGaitType("crawl");
            this.gaitController.frequency = targetFreq;
            this.gaitController.stride = parseFloat(document.getElementById('slider-stride').value);
            
            // Calculate a mock swing tilt towards weed
            this.gaitController.bodyRot.pitch = 8; // lean forward camera to inspect
            if (badgeState) {
                badgeState.innerText = "TARGET TRACKING active";
                badgeState.style.color = "var(--color-green)";
            }
        }
        else {
            // Normal operation (Restore slider inputs)
            let currentToggleGait = document.getElementById('gait-crawl').classList.contains('active') ? 'crawl' :
                                    document.getElementById('gait-trot').classList.contains('active') ? 'trot' : 'stand';
            this.gaitController.setGaitType(currentToggleGait);
            this.gaitController.frequency = targetFreq;
            this.gaitController.stride = parseFloat(document.getElementById('slider-stride').value);
            this.gaitController.bodyPos.y = targetHeight - 150;
            this.gaitController.bodyRot.pitch = parseFloat(pitchSlider.value);
            if (badgeState) {
                badgeState.innerText = "NOMINAL WALK sequence";
                badgeState.style.color = "var(--color-text-dim)";
            }
        }
    }

    updatePipelineUI(activeRuleId) {
        // Update input nodes summaries
        document.getElementById('pipe-sensors-body').innerHTML = `ToF: ${this.tof} mm<br>GPIO Trg: ${this.gpioTrigger}`;
        
        let melchiorText = "No targets (0.0s)";
        if (this.detections.length > 0) {
            melchiorText = this.detections.map(d => `${d.label} (${(d.confidence*100).toFixed(0)}%)`).join("<br>");
        }
        document.getElementById('pipe-melchior-body').innerHTML = melchiorText;
        
        document.getElementById('pipe-balthasar-body').innerHTML = `Scene: ${this.scene}<br>Anomaly: ${this.anomalyScore.toFixed(2)}`;

        // Update output nodes text
        let badge = document.getElementById('val-decision-badge');
        badge.innerText = this.action;
        badge.className = "decision-badge " + this.action.toLowerCase();
        
        document.getElementById('val-decision-reason').innerText = this.reason;

        // Highlight active rules in Caspar card
        const rules = ["rule-1", "rule-2", "rule-3", "rule-4", "rule-5", "rule-default"];
        rules.forEach(r => {
            let el = document.getElementById(r);
            if (el) {
                el.className = "rule-item";
                if (r === activeRuleId) {
                    if (this.action === "EMERGENCY") el.classList.add('active-danger');
                    else el.classList.add('active');
                }
            }
        });
    }

    setScenario(name) {
        // Clear all active classes from buttons
        const scenarios = ["clear", "obstacle", "intruder", "anomaly", "target"];
        scenarios.forEach(s => {
            document.getElementById(`scen-${s}`).classList.remove('active');
        });
        document.getElementById(`scen-${name}`).classList.add('active');

        if (name === "clear") {
            this.tof = 600;
            this.gpioTrigger = false;
            this.anomalyScore = 0.12;
            this.scene = "normal";
            this.detections = [];
        } 
        else if (name === "obstacle") {
            this.tof = 150;
            this.gpioTrigger = false;
            this.anomalyScore = 0.25;
            this.scene = "normal";
            this.detections = [];
        } 
        else if (name === "intruder") {
            this.tof = 550;
            this.gpioTrigger = false;
            this.anomalyScore = 0.65;
            this.scene = "restricted_zone";
            this.detections = [{ label: "person", confidence: 0.94 }];
        } 
        else if (name === "anomaly") {
            this.tof = 400;
            this.gpioTrigger = false;
            this.anomalyScore = 0.85;
            this.scene = "dense_weed";
            this.detections = [];
        } 
        else if (name === "target") {
            this.tof = 600;
            this.gpioTrigger = false;
            this.anomalyScore = 0.15;
            this.scene = "farm_field";
            this.detections = [{ label: "weed", confidence: 0.89 }];
        }

        // Sync slider positions
        document.getElementById('override-tof').value = this.tof;
        document.getElementById('val-override-tof').innerText = `${this.tof} mm`;
        document.getElementById('override-anomaly').value = this.anomalyScore;
        document.getElementById('val-override-anomaly').innerText = this.anomalyScore.toFixed(2);

        this.evaluateRules();
    }
}
