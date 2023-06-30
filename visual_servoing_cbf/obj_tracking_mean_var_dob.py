import argparse
import json
import os
import shutil
import apriltag
import time

import numpy as np
import numpy.linalg as LA
import pybullet as p
from scipy.spatial.transform import Rotation as R
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import Bounds, LinearConstraint, minimize, BFGS, SR1

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fr3_envs.fr3_env_with_cam_obs import FR3CameraSim
from utils.dict_utils import save_dict
from configuration import Configuration


def axis_angle_from_rot_mat(rot_mat):
    rotation = R.from_matrix(rot_mat)
    axis_angle = rotation.as_rotvec()
    angle = LA.norm(axis_angle)
    axis = axis_angle / angle

    return axis, angle

def one_point_image_jacobian(coord_in_cam, fx, fy):
    X = coord_in_cam[0]
    Y = coord_in_cam[1]
    Z = coord_in_cam[2]
    J1 = np.array([-fx/Z, 0, fx*X/Z**2, fx*X*Y/Z**2, fx*(-1-X**2/Z**2), fx*Y/Z], dtype=np.float32)
    J2 = np.array([0, -fy/Z, fy*Y/Z**2, fy*(1+Y**2/Z**2), -fy*X*Y/Z**2, -fy*X/Z], dtype=np.float32)

    return np.vstack((J1, J2))

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]], dtype=np.float32)

def skew_to_vector(S):
    return np.array([S[2,1], S[0,2], S[1,0]], dtype=np.float32)

def point_in_image(x, y, width, height):
    if (0 <= x and x < width):
        if (0 <= y and y < height):
            return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Visual servoing")
    parser.add_argument('--exp_num', default=1, type=int, help="test case number")

    # Set random seed
    seed_num = 0
    np.random.seed(seed_num)

    args = parser.parse_args()
    exp_num = args.exp_num
    results_dir = "{}/results_cbf/exp_{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    test_settings_path = "{}/test_settings/test_settings_{:03d}.json".format(str(Path(__file__).parent), exp_num)
    
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    shutil.copy(test_settings_path, results_dir)

    config = Configuration()

    with open(test_settings_path, "r", encoding="utf8") as f:
        test_settings = json.load(f)

    camera_config = test_settings["camera_config"]
    apriltag_config = test_settings["apriltag_config"]
    controller_config = test_settings["controller_config"]
    observer_config = test_settings["observer_config"]
    CBF_config = test_settings["CBF_config"]
    enable_gui_camera_data = test_settings["enable_gui_camera_data"]
    intrinsic_matrix = np.array(camera_config["intrinsic_matrix"], dtype=np.float32)

    # Create and reset simulation
    env = FR3CameraSim(camera_config, enable_gui_camera_data, render_mode="human")
    info = env.reset(target_joint_angles = test_settings["target_joint_angles"])

    # Reset apriltag position
    april_tag_quat = p.getQuaternionFromEuler([np.pi / 2, 0, np.pi / 2])
    p.resetBasePositionAndOrientation(env.april_tag_ID, apriltag_config["initial_position"], april_tag_quat)
    info = env.get_image()
    last_apriltag_position =  np.array(apriltag_config["initial_position"])

    # observer initialization
    observer_gain = np.diag(observer_config["gain"]*observer_config["num_points"])

    # Records
    dt = 1.0/240
    T = test_settings["horizon"]
    step_every = test_settings["step_every"]
    num_data = (T-1)//step_every + 1
    times = np.zeros(num_data, dtype = np.float32)
    mean_errs = np.zeros((num_data,2), dtype = np.float32)
    variance_errs = np.zeros((num_data,2), dtype = np.float32)
    observer_errs = np.zeros(num_data, dtype = np.float32)

    # Obstacle line
    obstacle_config = test_settings["obstacle_config"]
    p.addUserDebugLine(lineFromXYZ = obstacle_config["lineFromXYZ"],
                       lineToXYZ = obstacle_config["lineToXYZ"],
                       lineColorRGB = obstacle_config["lineColorRGB"],
                       lineWidth = obstacle_config["lineWidth"],
                       lifeTime = obstacle_config["lifeTime"])
    
    obstacle_quat = p.getQuaternionFromEuler([0, 0, 0])
    p.resetBasePositionAndOrientation(env.obstacle_ID, (np.array(obstacle_config["lineFromXYZ"]) + np.array(obstacle_config["lineToXYZ"]))/2.0, obstacle_quat)
    p.changeVisualShape(env.obstacle_ID, -1, rgbaColor=[1., 1., 1., obstacle_config["obstacle_alpha"]])
    
    for i in range(T):
        augular_velocity = apriltag_config["augular_velocity"]
        apriltag_angle = augular_velocity*i*dt + apriltag_config["offset_angle"]
        apriltag_radius = apriltag_config["apriltag_radius"]
        apriltag_position = np.array([apriltag_radius*np.cos(apriltag_angle), apriltag_radius*np.sin(apriltag_angle), 0]) + apriltag_config["center_position"]
        # p.resetBasePositionAndOrientation(env.april_tag_ID, apriltag_position, april_tag_quat)
        apriltag_speed_in_world = (apriltag_position - last_apriltag_position)/dt
        last_apriltag_position = apriltag_position
 
        if i % step_every == 0:
            info = env.get_image()
            rgb_opengl = info["rgb"]
            img = 0.2125*rgb_opengl[:,:,0] + 0.7154*rgb_opengl[:,:,1] + 0.0721*rgb_opengl[:,:,2]
            img = img.astype(np.uint8)
            detector = apriltag.Detector()
            result = detector.detect(img)
            corners = result[0].corners

            cv2.namedWindow("Input")
            cv2.imshow("Input", img)

            # Initialize the observer such that d_hat = 0 at t = 0
            if i == 0:
                epsilon = observer_gain @ np.reshape(corners, (2*len(corners),))

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

            # colors = [[0.5,0.5,0.5],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
            # p.addUserDebugPoints(coord_in_world[:,0:3], colors, pointSize=5, lifeTime=0.01)

            # Compute image jaccobian due to camera speed
            J_image_cam = np.zeros((2*corners.shape[0], 6), dtype=np.float32)
            fx = intrinsic_matrix[0, 0]
            fy = intrinsic_matrix[1, 1]
            for ii in range(len(corners)):
                J_image_cam[2*ii:2*ii+2] = one_point_image_jacobian(coord_in_cam[ii], fx, fy)

            # Compute desired pixel velocity (mean)
            mean_gain = np.diag(controller_config["mean_gain"])
            x0 = intrinsic_matrix[0, 2]
            y0 = intrinsic_matrix[1, 2]
            J_mean = 1/len(corners)*np.tile(np.eye(2), len(corners))
            error_mean = np.mean(corners, axis=0) - [x0,y0]
            xd_yd_mean = - LA.pinv(J_mean) @ mean_gain @ error_mean

            # Compute desired pixel velocity (variance)
            variance_gain = np.diag(controller_config["variance_gain"])
            variance_target = controller_config["variance_target"]
            J_variance = np.tile(- np.diag(np.mean(corners, axis=0)), len(corners))
            J_variance[0,0::2] += corners[:,0]
            J_variance[1,1::2] += corners[:,1]
            J_variance = 2/len(corners)*J_variance
            error_variance = np.var(corners, axis = 0) - variance_target
            xd_yd_variance = - LA.pinv(J_variance) @ variance_gain @ error_variance

            # Compute desired pixel velocity (orientation)
            orientation_gain = np.diag([controller_config["horizontal_gain"], controller_config["vertical_gain"]])
            J_orientation = np.zeros((2, 2*len(corners)), dtype=np.float32)
            tmp1 = np.arange(0,len(corners),2,dtype=np.int_)
            tmp2 = np.arange(1,len(corners),2,dtype=np.int_)
            tmp = np.zeros(len(corners), dtype=np.int_)
            tmp[0::2] = tmp2
            tmp[1::2] = tmp1
            J_orientation[0,1::2] += corners[:,1] - corners[tmp,1]
            J_orientation[1,0::2] += corners[:,0] - np.flip(corners[:,0])
            J_orientation = 2*J_orientation
            error_orientation = np.sum(J_orientation**2, axis=1)/8.0
            xd_yd_orientation = - LA.pinv(J_orientation) @ orientation_gain @ error_orientation

            # Speed contribution due to movement of the apriltag
            apriltag_speed_in_cam = info["R_CAMERA"] @ apriltag_speed_in_world
            d_true = -J_image_cam[:,0:3] @ apriltag_speed_in_cam

            # Update the observer
            d_hat = observer_gain @ np.reshape(corners, (2*len(corners),)) - epsilon

            # Map to the camera speed expressed in the camera frame
            null_mean = np.eye(2*len(corners), dtype=np.float32) - LA.pinv(J_mean) @ J_mean
            null_variance = np.eye(2*len(corners), dtype=np.float32) - LA.pinv(J_variance) @ J_variance
            xd_yd = xd_yd_mean + xd_yd_variance + null_mean @ null_variance @ xd_yd_orientation
            speeds_in_cam_desired = LA.pinv(J_image_cam) @ (xd_yd - d_hat)

            # Map obstacle edge to image
            two_ends_in_world = np.array([obstacle_config["lineFromXYZ"] + [1], obstacle_config["lineToXYZ"] + [1]], dtype = np.float32)
            two_ends_in_cam = two_ends_in_world @ LA.inv(H).T 
            two_ends_in_image = two_ends_in_cam[:,0:3] @ intrinsic_matrix.T
            two_ends_in_image = two_ends_in_image/two_ends_in_image[:,-1][:,np.newaxis]
            two_ends_pixel_coord = two_ends_in_image[:,0:2]

            if CBF_config["active"] == 1:
                # Construct CBF and its constraint
                x1, y1 = two_ends_pixel_coord[0,:]
                x2, y2 = two_ends_pixel_coord[1,:]
                CBF = lambda x, y : (y2-y1)*(x-x1) - (y-y1)*(x2-x1)
                grad_CBF = lambda x, y: np.array([y2-y1, -x2+x1, y-y2, -x+x2, -y+y1, x-x1], dtype=np.float32)
                grad_CBF_disturbance = lambda x, y: np.array([y2-y1, -x2+x1], dtype=np.float32)
                # CBF_values = (corners - two_ends_pixel_coord[0,:])@np.array([y2-y1, -(x2-x1)])
                A_CBF = np.zeros((len(corners), 6), dtype=np.float32)
                lb_CBF = np.zeros(len(corners), dtype=np.float32)
                ub_CBF = np.array([np.inf]*len(corners))
                for ii in range(len(corners)):
                    actuation_matrix = np.zeros((6,6), dtype = np.float32)
                    actuation_matrix[0:2,:] = one_point_image_jacobian(coord_in_cam[ii], fx, fy)
                    actuation_matrix[2:4,:] = one_point_image_jacobian(two_ends_in_cam[0,0:3], fx, fy)
                    actuation_matrix[4:6,:] = one_point_image_jacobian(two_ends_in_cam[1,0:3], fx, fy)
                    A_CBF[ii,:] = grad_CBF(*corners[ii]) @ actuation_matrix
                    lb_CBF[ii] = -CBF_config["barrier_alpha"]*CBF(*corners[ii]) + CBF_config["compensation"]\
                        - grad_CBF_disturbance(*corners[ii]) @ d_hat[2*ii:2*ii+2]
                
                CBF_constraint = LinearConstraint(A_CBF, lb_CBF, ub_CBF)

                # solve optimization
                def obj(u):
                    return (u-speeds_in_cam_desired).T @ (u-speeds_in_cam_desired)
                
                def obj_der(u):
                    return 2*(u-speeds_in_cam_desired)
                
                u0 = speeds_in_cam_desired
                res = minimize(obj, u0, method='trust-constr', jac=obj_der, hess=BFGS(),
                            constraints = [CBF_constraint], options={'verbose': 0})
                speeds_in_cam = np.array(res.x, dtype=np.float32)
            else: 
                speeds_in_cam = speeds_in_cam_desired

            # Transform the speed back to the world frame
            v_in_cam = speeds_in_cam[0:3]
            omega_in_cam = speeds_in_cam[3:6]
            R_cam_to_world = info["R_CAMERA"]
            v_in_world = R_cam_to_world @ v_in_cam
            S_in_world = R_cam_to_world @ skew(omega_in_cam) @ R_cam_to_world.T
            omega_in_world = skew_to_vector(S_in_world)
            
            # Map the desired camera speed to joint velocities
            J_camera = info["J_CAMERA"]
            pinv_J_camera = LA.pinv(J_camera)
            vel = pinv_J_camera @ np.concatenate((v_in_world, omega_in_world)) \
                + (np.eye(9) - pinv_J_camera @ J_camera) @ info["GRAD_MANIPULABILITY"] * controller_config["manipulability_gain"]
            # vel[-2:] = 0
            vel[:] = 0
            # print(np.sqrt(LA.det(J_camera @ J_camera.T)))

            if test_settings["save_rgb"] == 1:
                rgb_opengl = info["rgb"]
                rgbim = Image.fromarray(rgb_opengl)
                rgbim_no_alpha = rgbim.convert('RGB')
                rgbim_no_alpha.save(results_dir+'/rgb_'+str(i)+'.{}'.format(test_settings["image_save_format"]))

            if test_settings["save_depth"] == 1:
                plt.imsave(results_dir+'/depth_'+str(i)+'.{}'.format(test_settings["image_save_format"]), depth_opengl)

            if test_settings["save_detection"] == 1:
                for ii in range(len(corners)):
                    x, y = corners[ii,:]
                    img = cv2.circle(img, (int(x),int(y)), radius=5, color=(0, 0, 255), thickness=-1)
                x1, y1 = two_ends_pixel_coord[0,:]
                x2, y2 = two_ends_pixel_coord[1,:]

                y_left = int((y1-y2)/(x1-x2)*(0-x1) + y1)
                y_right = int((y1-y2)/(x1-x2)*(camera_config["width"]-1-x1) + y1)
                x_top = int((x1-x2)/(y1-y2)*(0-y1) + x1)
                x_bottom = int((x1-x2)/(y1-y2)*(camera_config["height"]-1-y1) + x1)

                potential_ends = np.array([[0, y_left],[camera_config["width"]-1, y_right],[x_top, 0],[x_bottom, camera_config["height"]-1]], dtype=np.int_)
                if_in = np.zeros(4, dtype=bool)
                for ii in range(len(potential_ends)):
                    x, y = potential_ends[ii,:]
                    if_in[ii] = point_in_image(x, y, camera_config["width"], camera_config["height"])
                if np.sum(if_in) >= 2:
                    final_ends = potential_ends[if_in]
                    cv2.line(img, final_ends[0,:], final_ends[1,:], color=(0, 255, 0), thickness=1) 

                cv2.imwrite(results_dir+'/detect_'+str(i)+'.{}'.format(test_settings["image_save_format"]), img)

            # Step the simulation
            info = env.step(vel, return_image=False)

            # Records
            times[i//step_every] = i*step_every*dt
            mean_errs[i//step_every,:] = error_mean
            variance_errs[i//step_every,:] = error_variance
            observer_errs[i//step_every] = LA.norm(d_hat - d_true)

            # Step the observer
            epsilon += step_every * dt * observer_gain @ (J_image_cam @speeds_in_cam + d_hat)


        else:
            info = env.step(vel)

        # Step the observer
        # epsilon += dt * observer_gain @ (J_image_cam @speeds_in_cam + d_hat)
        q, dq = info["q"], info["dq"]

        if i % 500 == 0:
            print("Iter {:.2e}".format(i))
        

    env.close()
    
    print("==> Drawing plots ...")
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi, frameon=True)
    plt.plot(times, mean_errs[:,0], label="x")
    plt.plot(times, mean_errs[:,1], label="y")
    plt.legend()
    plt.axhline(y = 0.0, color = 'black', linestyle = 'dotted')
    plt.savefig(os.path.join(results_dir, 'plot_mean_errs.png'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi, frameon=True)
    plt.plot(times, variance_errs[:,0], label="x")
    plt.plot(times, variance_errs[:,1], label="y")
    plt.legend()
    plt.axhline(y = 0.0, color = 'black', linestyle = 'dotted')
    plt.savefig(os.path.join(results_dir, 'plot_variance_errs.png'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi, frameon=True)
    plt.plot(times, observer_errs, label="observer_errs")
    plt.legend()
    plt.axhline(y = 0.0, color = 'black', linestyle = 'dotted')
    plt.savefig(os.path.join(results_dir, 'plot_observer_errs.png'))
    plt.close(fig)

    summary = {'times': times,
                'mean_errs': mean_errs,
                'variance_errs': variance_errs,
                'observer_errs': observer_errs}
    print("==> Saving summary ...")
    save_dict(summary, os.path.join(results_dir, "summary.npy"))


if __name__ == "__main__":
    main()
