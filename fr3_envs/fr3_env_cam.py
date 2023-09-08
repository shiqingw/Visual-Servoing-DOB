import copy
from typing import Optional

import numpy as np
import numpy.linalg as LA
import pinocchio as pin
import pybullet as p
import pybullet_data
from gymnasium import Env, spaces
from pinocchio.robot_wrapper import RobotWrapper
from scipy.spatial.transform import Rotation


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from helper_functions import getDataPath
from utils.render_utils import cvPose2BulletView, cvK2BulletP


class FR3CameraSim(Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(self, camera_config, enable_gui_camera_data, base_pos=[0,0,0], base_quat=[0,0,0,1], render_mode= None, record_path=None):
        if render_mode == "human":
            self.client = p.connect(p.GUI)
            # Improves rendering performance on M1 Macs
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, enable_gui_camera_data)
        else:
            self.client = p.connect(p.DIRECT)

        self.record_path = record_path

        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load plane
        p.loadURDF("plane.urdf")

        # Load Franka Research 3 Robot
        model_name = "fr3_with_bounding_boxes"
        package_directory = getDataPath()
        robot_URDF = package_directory + "/robots/{}.urdf".format(model_name)
        urdf_search_path = package_directory + "/robots"
        p.setAdditionalSearchPath(urdf_search_path)
        self.robotID = p.loadURDF("{}.urdf".format(model_name), useFixedBase=True)
        _base_pos, _base_ori = p.getBasePositionAndOrientation(self.robotID)
        self.base_R_offset = Rotation.from_quat(base_quat).as_matrix()
        base_p_offset = (np.array(_base_pos)+ np.array(base_pos, dtype=np.float32)).reshape(-1,1)
        p.resetBasePositionAndOrientation(self.robotID, base_p_offset, base_quat)
        self.base_p_offset = base_p_offset - np.array(_base_pos).reshape(-1,1)

        # Build pin_robot
        self.robot = RobotWrapper.BuildFromURDF(robot_URDF, package_directory)

        # Get active joint ids
        self.active_joint_ids = self.get_active_joint_ids()

        # Enable the velocity control on the joints 
        p.setJointMotorControlArray(
            self.robotID,
            self.active_joint_ids,
            p.VELOCITY_CONTROL,
            forces=1000*np.ones(9),
        )

        # Get frame ID for grasp target
        self.jacobian_frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED

        # Get frame ids for bounding boxes
        self.FR3_LINK3_BB_FRAME_ID = self.robot.model.getFrameId("fr3_link3_bounding_box")
        self.FR3_LINK4_BB_FRAME_ID = self.robot.model.getFrameId("fr3_link4_bounding_box")
        self.FR3_LINK5_1_BB_FRAME_ID = self.robot.model.getFrameId("fr3_link5_1_bounding_box")
        self.FR3_LINK5_2_BB_FRAME_ID = self.robot.model.getFrameId("fr3_link5_2_bounding_box")
        self.FR3_LINK6_BB_FRAME_ID = self.robot.model.getFrameId("fr3_link6_bounding_box")
        self.FR3_LINK7_BB_FRAME_ID = self.robot.model.getFrameId("fr3_link7_bounding_box")
        self.FR3_HAND_BB_FRAME_ID = self.robot.model.getFrameId("fr3_hand_bounding_box")

        # Get frame ID for links
        self.FR3_LINK3_FRAME_ID = self.robot.model.getFrameId("fr3_link3")
        self.FR3_LINK4_FRAME_ID = self.robot.model.getFrameId("fr3_link4")
        self.FR3_LINK5_FRAME_ID = self.robot.model.getFrameId("fr3_link5")
        self.FR3_LINK5_FRAME_ID = self.robot.model.getFrameId("fr3_link5")
        self.FR3_LINK6_FRAME_ID = self.robot.model.getFrameId("fr3_link6")
        self.FR3_LINK7_FRAME_ID = self.robot.model.getFrameId("fr3_link7")
        self.FR3_HAND_FRAME_ID = self.robot.model.getFrameId("fr3_hand")
        self.EE_FRAME_ID = self.robot.model.getFrameId("fr3_hand_tcp")
        self.FR3_CAMERA_FRAME_ID = self.robot.model.getFrameId("fr3_camera")

        # Choose the useful frame names with frame ids 
        self.frame_names_and_ids = {
            "LINK3_BB": self.FR3_LINK3_BB_FRAME_ID,
            "LINK4_BB": self.FR3_LINK4_BB_FRAME_ID,
            "LINK5_1_BB": self.FR3_LINK5_1_BB_FRAME_ID,
            "LINK5_2_BB": self.FR3_LINK5_2_BB_FRAME_ID,
            "LINK6_BB": self.FR3_LINK6_BB_FRAME_ID,
            "LINK7_BB": self.FR3_LINK7_BB_FRAME_ID,
            "HAND_BB": self.FR3_HAND_BB_FRAME_ID,
            "CAMERA": self.FR3_CAMERA_FRAME_ID,
        }

        # Get camera projection matrix
        self.width = camera_config["width"]
        self.height = camera_config["height"]
        near = camera_config["near"]
        far = camera_config["far"]
        camera_intrinsic_matrix = np.array(camera_config["intrinsic_matrix"],dtype=np.float32)
        self.projection_matrix =  cvK2BulletP(camera_intrinsic_matrix, self.width, self.height, near, far)

    def reset(
        self,
        seed = None,
        options = None,
        cameraDistance=1.4,
        cameraYaw=66.4,
        cameraPitch=-16.2,
        lookat=[0.0, 0.0, 0.0],
        target_joint_angles = [
            0.0,
            -0.785398163,
            0.0,
            -2.35619449,
            0.0,
            1.57079632679,
            0.785398163397,
            0.001,
            0.001,
            ]
        ):
        super().reset(seed=seed)

        self.q_nominal = np.array(target_joint_angles)

        for i, joint_ang in enumerate(target_joint_angles):
            p.resetJointState(self.robotID, self.active_joint_ids[i], joint_ang, 0.0)

        q, dq = self.get_state_update_pinocchio()
        info = self.get_info(q, dq)

        p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, lookat)

        if self.record_path is not None:
            self.loggingId = p.startStateLogging(
                p.STATE_LOGGING_VIDEO_MP4, self.record_path
            )

        return info

    def get_info(self, q, dq):
        """
        info contains:
        -------------------------------------
        q: joint position
        dq: joint velocity
        J_{frame_name}: jacobian of frame_name
        P_{frame_name}: position of frame_name
        R_{frame_name}: orientation of frame_name
        """

        # Get Jacobian from grasp target frame
        # preprocessing is done in get_state_update_pinocchio()
        info = {"q": q,
                "dq": dq}
        for frame_name, frame_id in self.frame_names_and_ids.items():
            # Frame jacobian
            info[f"J_{frame_name}"] = self.robot.getFrameJacobian(frame_id, self.jacobian_frame)
            # Frame position and orientation
            (
                info[f"P_{frame_name}"],
                info[f"R_{frame_name}"],
                info[f"q_{frame_name}"],
            ) = self.compute_crude_location(
                self.base_R_offset, self.base_p_offset, frame_id
            )

        # Advanced calculation that is not used in the current version
        # dJ = pin.getFrameJacobianTimeVariation(
        #     self.robot.model, self.robot.data, self.EE_FRAME_ID, self.jacobian_frame
        # )
        # f, g, M, Minv, nle = self.get_dynamics(q, dq)

        return info

    def get_dynamics(self, q, dq):
        """
        f.shape = (18, 1), g.shape = (18, 9)
        """
        Minv = pin.computeMinverse(self.robot.model, self.robot.data, q)
        M = self.robot.mass(q)
        nle = self.robot.nle(q, dq)

        f = np.vstack((dq[:, np.newaxis], -Minv @ nle[:, np.newaxis]))
        g = np.vstack((np.zeros((9, 9)), Minv))

        return f, g, M, Minv, nle

    def step(self, action, return_image=False):

        self.send_joint_command_velocity(action)
        p.stepSimulation()

        q, dq = self.get_state_update_pinocchio()
        info = self.get_info(q, dq)

        if return_image:
            R_camera = info["R_CAMERA"]
            q = Rotation.from_matrix(R_camera).as_quat()
            view_matrix = cvPose2BulletView(q, info["P_CAMERA"])
            
            img = p.getCameraImage(
                self.width,
                self.height,
                viewMatrix=view_matrix,
                projectionMatrix=self.projection_matrix,
                flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )

            info["rgb"] = np.reshape(img[2], (self.height, self.width, 4))
            info["depth"] = np.reshape(img[3], (self.height, self.width))
            # info["seg"] = np.reshape(img[4], (self.height, self.width))* 1. / 255.

        return info
    
    def get_image(self):

        q, dq = self.get_state_update_pinocchio()
        info = self.get_info(q, dq)

        R_camera = info["R_CAMERA"]
        q = Rotation.from_matrix(R_camera).as_quat()
        view_matrix = cvPose2BulletView(q, info["P_CAMERA"])

        img = p.getCameraImage(
            self.width,
            self.height,
            viewMatrix=view_matrix,
            projectionMatrix=self.projection_matrix,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        info["rgb"] = np.reshape(img[2], (self.height, self.width, 4))
        info["depth"] = np.reshape(img[3], (self.height, self.width))
        # info["seg"] = np.reshape(img[4], (self.height, self.width))* 1. / 255.

        return info

    def close(self):
        if self.record_path is not None:
            p.stopStateLogging(self.loggingId)
        p.disconnect()

    def get_state(self):
        q = np.zeros(9)
        dq = np.zeros(9)

        for i, id in enumerate(self.active_joint_ids):
            _joint_state = p.getJointState(self.robotID, id)
            q[i], dq[i] = _joint_state[0], _joint_state[1]

        return q, dq

    def update_pinocchio(self, q, dq):
        self.robot.computeJointJacobians(q)
        self.robot.framesForwardKinematics(q)
        self.robot.centroidalMomentum(q, dq)

    def get_state_update_pinocchio(self):
        q, dq = self.get_state()
        self.update_pinocchio(q, dq)

        return q, dq

    def send_joint_command_velocity(self, dq):
        zeroGains = dq.shape[0] * (0.0,)
        oneGains = dq.shape[0] * (1.0,)

        p.setJointMotorControlArray(
            self.robotID,
            self.active_joint_ids,
            p.VELOCITY_CONTROL,
            targetVelocities=dq,
            positionGains=zeroGains,
            velocityGains=oneGains,
        )

    def compute_crude_location(self, base_R_offset, base_p_offset, frame_id):
        # get link orientation and position
        _p = self.robot.data.oMf[frame_id].translation
        _Rot = self.robot.data.oMf[frame_id].rotation

        # compute link transformation matrix
        _T = np.hstack((_Rot, _p[:, np.newaxis]))
        T = np.vstack((_T, np.array([[0.0, 0.0, 0.0, 1.0]])))

        # compute link offset transformation matrix
        _TW = np.hstack((base_R_offset, base_p_offset))
        TW = np.vstack((_TW, np.array([[0.0, 0.0, 0.0, 1.0]])))
        
        # get transformation matrix
        T_mat = TW @ T 

        # compute crude model location
        p = (T_mat @ np.array([[0.0], [0.0], [0.0], [1.0]]))[:3, 0]

        # compute crude model orientation
        Rot = T_mat[:3, :3]

        # quaternion
        q = Rotation.from_matrix(Rot).as_quat()

        return p, Rot, q
    
    def get_active_joint_ids(self):
        active_joint_ids = []
        n_j = p.getNumJoints(self.robotID)
        for i in range(n_j):
            # get info of each joint
            _joint_infos = p.getJointInfo(self.robotID, i)
            if _joint_infos[2] != p.JOINT_FIXED:
                # Save the non-fixed joint IDs
                active_joint_ids.append(_joint_infos[0])
                
        return active_joint_ids
