from typing import Optional
import pybullet as p

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from fr3_env_cam_obs import FR3CameraSim


class FR3CameraSimCollision(FR3CameraSim):
    def __init__(self, camera_config, enable_gui_camera_data, obs_urdf="box.urdf",
                  render_mode: Optional[str] = None, record_path=None, crude_type="ellipsoid"):
        super().__init__(camera_config, enable_gui_camera_data, obs_urdf,
                  render_mode, record_path)
        
        # load robot bounding primitives
        self.link3 = p.loadURDF(f"fr3_link3_{crude_type}.urdf", useFixedBase=True)
        p.resetBasePositionAndOrientation(self.link3, [2, 2, 2], [0, 0, 0, 1])

        self.link4 = p.loadURDF(f"fr3_link4_{crude_type}.urdf", useFixedBase=True)
        p.resetBasePositionAndOrientation(self.link4, [2, 2, 2], [0, 0, 0, 1])

        self.link5_1 = p.loadURDF(f"fr3_link5_1_{crude_type}.urdf", useFixedBase=True)
        p.resetBasePositionAndOrientation(self.link5_1, [2, 2, 2], [0, 0, 0, 1])

        self.link5_2 = p.loadURDF(f"fr3_link5_2_{crude_type}.urdf", useFixedBase=True)
        p.resetBasePositionAndOrientation(self.link5_2, [2, 2, 2], [0, 0, 0, 1])

        self.link6 = p.loadURDF(f"fr3_link6_{crude_type}.urdf", useFixedBase=True)
        p.resetBasePositionAndOrientation(self.link6, [2, 2, 2], [0, 0, 0, 1])

        self.link7 = p.loadURDF(f"fr3_link7_{crude_type}.urdf", useFixedBase=True)
        p.resetBasePositionAndOrientation(self.link7, [2, 2, 2], [0, 0, 0, 1])

        # load end-effector
        self.hand = p.loadURDF("fr3_hand_collision.urdf", useFixedBase=True)
        p.resetBasePositionAndOrientation(self.hand, [2, 2, 2], [0, 0, 0, 1])

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
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
            0.001]
        ):
        info = super().reset(seed, options, cameraDistance, cameraYaw, cameraPitch,lookat, target_joint_angles)
        
        # update bounding primitives' configuration
        p.resetBasePositionAndOrientation(
            self.link3, info["P_LINK3"].tolist(), info["q_LINK3"].tolist()
        )
        p.resetBasePositionAndOrientation(
            self.link4, info["P_LINK4"].tolist(), info["q_LINK4"].tolist()
        )
        p.resetBasePositionAndOrientation(
            self.link5_1, info["P_LINK5_1"].tolist(), info["q_LINK5_1"].tolist()
        )
        p.resetBasePositionAndOrientation(
            self.link5_2, info["P_LINK5_2"].tolist(), info["q_LINK5_2"].tolist()
        )
        p.resetBasePositionAndOrientation(
            self.link6, info["P_LINK6"].tolist(), info["q_LINK6"].tolist()
        )
        p.resetBasePositionAndOrientation(
            self.link7, info["P_LINK7"].tolist(), info["q_LINK7"].tolist()
        )
        p.resetBasePositionAndOrientation(
            self.hand, info["P_HAND"].tolist(), info["q_HAND"].tolist()
        )

        return info
    
    def step(self, action, return_image=False):
        info = super().step(action, return_image)

        # update bounding primitives' configuration
        p.resetBasePositionAndOrientation(
            self.link3, info["P_LINK3"].tolist(), info["q_LINK3"].tolist()
        )
        p.resetBasePositionAndOrientation(
            self.link4, info["P_LINK4"].tolist(), info["q_LINK4"].tolist()
        )
        p.resetBasePositionAndOrientation(
            self.link5_1, info["P_LINK5_1"].tolist(), info["q_LINK5_1"].tolist()
        )
        p.resetBasePositionAndOrientation(
            self.link5_2, info["P_LINK5_2"].tolist(), info["q_LINK5_2"].tolist()
        )
        p.resetBasePositionAndOrientation(
            self.link6, info["P_LINK6"].tolist(), info["q_LINK6"].tolist()
        )
        p.resetBasePositionAndOrientation(
            self.link7, info["P_LINK7"].tolist(), info["q_LINK7"].tolist()
        )
        p.resetBasePositionAndOrientation(
            self.hand, info["P_HAND"].tolist(), info["q_HAND"].tolist()
        )

        return info