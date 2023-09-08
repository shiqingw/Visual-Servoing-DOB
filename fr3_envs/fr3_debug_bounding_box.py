import pathlib

import pybullet as p
import pybullet_data


def main():
    p.connect(p.GUI)
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load plane
    p.loadURDF("plane.urdf")

    # Load Franka Research 3 Robot
    file_directory = str(pathlib.Path(__file__).parent.parent.resolve())
    print(file_directory)
    robot_URDF = file_directory + "/robots/fr3_with_bounding_boxes.urdf"
    robotID = p.loadURDF(robot_URDF, useFixedBase=True)
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=32.67, cameraPitch=-31.46, cameraTargetPosition=[0,0,0])

    # Get number of joints
    n_j = p.getNumJoints(robotID)

    debug_sliders = []
    joint_ids = []

    default_joint_angles = [0.0, -0.75, 0.0, -2.356, 0.0, 1.5, 0.785, 0.001, 0.001]
    counter = 0

    for i in range(n_j):
        # get info of each joint
        _joint_infos = p.getJointInfo(robotID, i)

        if _joint_infos[2] != p.JOINT_FIXED:
            # Add a debug slider for all non-fixed joints
            debug_sliders.append(
                p.addUserDebugParameter(
                    _joint_infos[1].decode("UTF-8"),  # Joint Name
                    _joint_infos[8],  # Lower Joint Limit
                    _joint_infos[9],  # Upper Joint Limit
                    default_joint_angles[counter],  # Default Joint Angle
                )
            )

            # Save the non-fixed joint IDs
            joint_ids.append(_joint_infos[0])
            counter += 1

    while True:
        for slider_id, joint_id in zip(debug_sliders, joint_ids):
            # Get joint angle from debug slider
            try:
                _joint_angle = p.readUserDebugParameter(slider_id)
            except:
                # Sometimes it fails to read the debug slider
                continue

            # Apply joint angle to robot
            p.resetJointState(robotID, joint_id, _joint_angle)

        p.stepSimulation()

    p.disconnect()


if __name__ == "__main__":
    main()
