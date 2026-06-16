import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.loadURDF("plane.urdf")
robot = p.loadURDF("SpiderQ.urdf", [0,0,1])

p.setGravity(0,0,-9.8)

while True:
    p.stepSimulation()
    time.sleep(1/240)