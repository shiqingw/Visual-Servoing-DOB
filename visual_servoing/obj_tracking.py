import argparse
import json
import os
import shutil
import apriltag

import numpy as np
import numpy.linalg as LA
import pybullet as p
from scipy.spatial.transform import Rotation as R
from PIL import Image
import matplotlib.pyplot as plt
import cv2

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fr3_envs.fr3_env_with_cam_moving import FR3CameraSim



def axis_angle_from_rot_mat(rot_mat):
    rotation = R.from_matrix(rot_mat)
    axis_angle = rotation.as_rotvec()
    angle = LA.norm(axis_angle)
    axis = axis_angle / angle

    return axis, angle


def main():
    parser = argparse.ArgumentParser(description="Visual servoing")
    parser.add_argument('--exp_num', default=0, type=int, help="test case number")

    # Set random seed
    seed_num = 0
    np.random.seed(seed_num)

    args = parser.parse_args()
    exp_num = args.exp_num
    results_dir = "{}/results_obj_tracking/exp_{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    test_settings_path = "{}/test_settings/test_settings_{:03d}.json".format(str(Path(__file__).parent), exp_num)
    
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    shutil.copy(test_settings_path, results_dir)

    with open(test_settings_path, "r", encoding="utf8") as f:
        test_settings = json.load(f)

    camera_config = test_settings["camera_config"]
    apriltag_config = test_settings["apriltag_config"]
    enable_gui_camera_data = test_settings["enable_gui_camera_data"]
    intrinsic_matrix = np.array(camera_config["intrinsic_matrix"], dtype=np.float32)

    env = FR3CameraSim(camera_config, enable_gui_camera_data, render_mode="human")
    info = env.reset()

    # reset apriltag position
    april_tag_quat = p.getQuaternionFromEuler([np.pi / 2, 0, np.pi / 2])
    p.resetBasePositionAndOrientation(env.april_tag_ID, apriltag_config["initial_position"], april_tag_quat)

    T = test_settings["horizon"]
    for i in range(T):
        apriltag_angle = 2*np.pi*i/T
        apriltag_radius = 0.1
        apriltag_position = [0.4 + apriltag_radius*np.sin(apriltag_angle), apriltag_radius*np.cos(apriltag_angle), 0.005]
        p.resetBasePositionAndOrientation(env.april_tag_ID, apriltag_position, april_tag_quat)

        J_camera = info["J_CAMERA"]
        if i % 100 == 0:
            vel = np.zeros([np.shape(J_camera)[1], 1], dtype=np.float32)
            info = env.step(vel, return_image=True)

            rgb_opengl = info["rgb"]
            img = 0.2125*rgb_opengl[:,:,0] + 0.7154*rgb_opengl[:,:,1] + 0.0721*rgb_opengl[:,:,2]
            img = img.astype(np.uint8)
            detector = apriltag.Detector()
            result = detector.detect(img)
            corners = result[0].corners
            # print(corners)

            depth_buffer_opengl = info["depth"]
            near = camera_config["near"]
            far = camera_config["far"]
            depth_opengl = far * near / (far - (far - near) * depth_buffer_opengl)
            
            corner_depths = np.zeros([corners.shape[0],1], dtype=np.float32)
            for ii in range(len(corners)):
                x, y = corners[ii,:]
                corner_depths[ii] = depth_opengl[int(y), int(x)]
            
            pixel_coord = np.hstack((corners, np.ones((corners.shape[0],1), dtype=np.float32)))
            pixel_coord_denomalized = pixel_coord*corner_depths
            
            coord_in_cam = pixel_coord_denomalized @ LA.inv(intrinsic_matrix.T)
            coord_in_cam = np.hstack((coord_in_cam, np.ones((coord_in_cam.shape[0],1), dtype=np.float32)))

            _H = np.hstack((info["R_CAMERA"], np.reshape(info["P_CAMERA"],(3,1))))
            H = np.vstack((_H, np.array([[0.0, 0.0, 0.0, 1.0]])))

            coord_in_world = coord_in_cam @ H.T
            # print(coord_in_world)

            colors = [[0.5,0.5,0.5],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
            p.addUserDebugPoints(coord_in_world[:,0:3], colors, pointSize=5, lifeTime=0.01)

            if test_settings["save_rgb"] == "true":
                rgb_opengl = info["rgb"]
                rgbim = Image.fromarray(rgb_opengl)
                rgbim_no_alpha = rgbim.convert('RGB')
                rgbim_no_alpha.save(results_dir+'/rgb_'+str(i)+'.{}'.format(test_settings["image_save_format"]))

            if test_settings["save_depth"] == "true":
                plt.imsave(results_dir+'/depth_'+str(i)+'.{}'.format(test_settings["image_save_format"]), depth_opengl)

            if test_settings["save_detection"] == "true":
                for ii in range(len(corners)):
                    x, y = corners[ii,:]
                    img = cv2.circle(img, (int(x),int(y)), radius=5, color=(0, 0, 255), thickness=-1)
                cv2.imwrite(results_dir+'/detect_'+str(i)+'.{}'.format(test_settings["image_save_format"]), img)

        else:
            info = env.step(vel)

        q, dq = info["q"], info["dq"]

        if i % 500 == 0:
            print("Iter {:.2e}".format(i))
        

    env.close()


if __name__ == "__main__":
    main()
