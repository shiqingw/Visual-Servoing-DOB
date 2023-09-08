from typing import Optional
import pybullet as p

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from fr3_env_cam import FR3CameraSim


class FR3CameraSimCollision(FR3CameraSim):
    def __init__(self, camera_config, enable_gui_camera_data, base_pos=[0,0,0], base_quat=[0,0,0,1],
                  render_mode= None, record_path=None, crude_type="ellipsoid"):
        super().__init__(camera_config, enable_gui_camera_data, base_pos, base_quat, render_mode, record_path)

        # load robot bounding primitives
        self.BB_names_and_ids = {}
        links_to_wrap = ["LINK3", 
                        "LINK4", 
                        "LINK5_1", 
                        "LINK5_2", 
                        "LINK6", 
                        "LINK7"]
        for link_name in links_to_wrap:
            BB_name = f"{link_name}_BB"
            self.BB_names_and_ids[BB_name] = p.loadURDF(f"fr3_{link_name.lower()}_{crude_type}.urdf", 
                                                          useFixedBase=True)
            p.resetBasePositionAndOrientation(self.BB_names_and_ids[BB_name], [1, 1, 1], [0, 0, 0, 1])
        # load for the hand
        self.BB_names_and_ids["HAND_BB"] = p.loadURDF("fr3_hand_collision.urdf", useFixedBase=True)
        p.resetBasePositionAndOrientation(self.BB_names_and_ids["HAND_BB"], [1, 1, 1], [0, 0, 0, 1])


    def reset(
        self,
        seed=None,
        options=None,
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
        for BB_name, BB_id in self.BB_names_and_ids.items():
            p.resetBasePositionAndOrientation(BB_id, info[f"P_{BB_name}"].tolist(), 
                                              info[f"q_{BB_name}"].tolist())

        return info
    
    def step(self, action, return_image=False):
        info = super().step(action, return_image)

        # update bounding primitives' configuration
        for BB_name, BB_id in self.BB_names_and_ids.items():
            p.resetBasePositionAndOrientation(BB_id, info[f"P_{BB_name}"].tolist(), 
                                              info[f"q_{BB_name}"].tolist())
        return info