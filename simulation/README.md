# MAGI PyBullet Simulation & Visualization

This directory contains the Python-based kinematic simulation environment (using PyBullet) and local web dashboards for visualizing the robot's gait, sensor fusion, and 3D positioning.

## Key Files
- `demo_pybullet.py`: Launches the physics engine simulation for gait testing.
- `camera_server.py`: Simulates the multispectral camera feed.
- `ik.py` / `fk.py`: Inverse and Forward Kinematics solvers.
- `dashboard.js` / `magi_fusion.js`: Frontend logic for the local observer dashboard.
