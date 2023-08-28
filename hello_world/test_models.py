
import pybullet as p
import time
import pybullet_data
from pathlib import Path

physicsClient = p.connect(p.GUI)
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
# physicsClient = p.connect(p.DIRECT)
# urdf_file = "box_small.urdf"
# urdf_file = "square1.urdf"
urdf_file = "apriltag_id1.urdf"
# urdf_file = "apriltag_id0_square.urdf"
robot_folder = "{}/robots".format(Path(__file__).parent.parent)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,0]
startOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF(robot_folder + "/" + urdf_file, startPos, startOrientation) 
cameraDistance = 1.13
cameraYaw = 50
cameraPitch = -35
lookat = [0,0,0]
p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, lookat)
for i in range (100000):
    p.stepSimulation()
    time.sleep(1./240.)
    cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
    # print(cubePos,cubeOrn)
p.disconnect()