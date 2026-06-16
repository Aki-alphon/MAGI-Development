# MAGI Quadruped Locomotion Simulator

Browser-based 3D simulation and evaluation environment for the 12-DOF quadruped robot. Used to test kinematics, gait generation, genetic-algorithm-based locomotion policies, and decision logic overrides.

---

## Quick Start (Web Simulator)

1. Start the HTTP server:
   ```bash
   cd simulation
   python3 -m http.server 8765
   ```
2. Open `http://localhost:8765` in your browser.

*For setup issues, see the troubleshooting section in [Guide.md](file:///home/aki/Downloads/MAGI/simulation/Guide.md#troubleshooting).*

---

## Interface Tabs

* **Locomotion Sim**: 3D viewport of the robot walking. Supports Crawl, Trot, Gallop, Bound, and Stand modes with adjustable stride, clearance, pitch/roll, and terrain roughness.
* **RL Gait Training**: Runs a genetic algorithm directly in the browser to evolve gait parameters or weights for an 8-16-12 feedforward neural network (NNPolicy).
* **Decision Fusion**: Tests the rule-based decision logic overrides (ToF and anomaly sensor inputs) in real-time.
* **Gait Analysis**: Plots joint energy consumption, step phase timing diagrams, and top-down foot path traces.
* **Kinematics Math**: Interactive 2D coordinate solver using the leg Inverse Kinematics equations.

---

## File Structure

* **[index.html](file:///home/aki/Downloads/MAGI/simulation/index.html)**: Main page and sidebar dashboard.
* **[style.css](file:///home/aki/Downloads/MAGI/simulation/style.css)**: Stylesheet and dark layout definitions.
* **[dashboard.js](file:///home/aki/Downloads/MAGI/simulation/dashboard.js)**: Main 60fps coordination loop.
* **[robot.js](file:///home/aki/Downloads/MAGI/simulation/robot.js)**: Three.js viewport configuration and IK solver.
* **[gait.js](file:///home/aki/Downloads/MAGI/simulation/gait.js)**: Cubic Bezier foot trajectory generator.
* **[rl_train.js](file:///home/aki/Downloads/MAGI/simulation/rl_train.js)**: Genetic algorithm implementation.
* **[magi_fusion.js](file:///home/aki/Downloads/MAGI/simulation/magi_fusion.js)**: Decision engine simulation rules.
* **[SpiderQ.urdf](file:///home/aki/Downloads/MAGI/simulation/SpiderQ.urdf)** / **[SpiderQ.glb](file:///home/aki/Downloads/MAGI/simulation/SpiderQ.glb)**: Robot geometry and 3D meshes.
* **[Guide.md](file:///home/aki/Downloads/MAGI/simulation/Guide.md)**: Detailed documentation of mathematical equations and variables.

---

## PyBullet Physics Simulation

For rigid-body physics tests, run the standalone Python scripts (requires `pybullet` and `numpy`):

```bash
pip install pybullet numpy
python3 demo_pybullet.py
```

* **[demo_pybullet.py](file:///home/aki/Downloads/MAGI/simulation/demo_pybullet.py)**: Loads URDF into PyBullet to test active gait patterns.
* **[ik.py](file:///home/aki/Downloads/MAGI/simulation/ik.py)** / **[fk.py](file:///home/aki/Downloads/MAGI/simulation/fk.py)**: Reference Python kinematics solvers.
* **[MAGI_gait_learning.py](file:///home/aki/Downloads/MAGI/simulation/MAGI_gait_learning.py)**: Standalone reinforcement learning python code.
