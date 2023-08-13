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
import proxsuite
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fr3_envs.fr3_env_with_cam_obs import FR3CameraSim
from utils.dict_utils import save_dict, load_dict
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
    results_dir = "{}/results_no_cbf/exp_{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    test_settings_path = "{}/test_settings/test_settings_{:03d}.json".format(str(Path(__file__).parent), exp_num)
    
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    shutil.copy(test_settings_path, results_dir)

    config = Configuration()

    with open(test_settings_path, "r", encoding="utf8") as f:
        test_settings = json.load(f)

    # Various configs
    simulator_config = test_settings["simulator_config"]
    screenshot_config = test_settings["screenshot_config"]
    camera_config = test_settings["camera_config"]
    apriltag_config = test_settings["apriltag_config"]
    controller_config = test_settings["controller_config"]
    observer_config = test_settings["observer_config"]
    CBF_config = test_settings["CBF_config"]
    optimization_config = test_settings["optimization_config"]

    # Joint limits
    joint_limits_config = test_settings["joint_limits_config"]
    joint_lb = np.array(joint_limits_config["lb"], dtype=np.float32)
    joint_ub = np.array(joint_limits_config["ub"], dtype=np.float32)

    # Camera parameters
    intrinsic_matrix = np.array(camera_config["intrinsic_matrix"], dtype=np.float32)
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    x0 = intrinsic_matrix[0, 2]
    y0 = intrinsic_matrix[1, 2]

    # Create and reset simulation
    enable_gui_camera_data = simulator_config["enable_gui_camera_data"]
    obs_urdf = simulator_config["obs_urdf"]
    cameraDistance = simulator_config["cameraDistance"]
    cameraYaw = simulator_config["cameraYaw"]
    cameraPitch = simulator_config["cameraPitch"]
    lookat = simulator_config["lookat"]

    if test_settings["record"] == 1:
        env = FR3CameraSim(camera_config, enable_gui_camera_data, obs_urdf, render_mode="human", record_path=os.path.join(results_dir, 'record.mp4'))
    else:
        env = FR3CameraSim(camera_config, enable_gui_camera_data, obs_urdf, render_mode="human", record_path=None)
    
    info = env.reset(cameraDistance = cameraDistance,
                     cameraYaw = cameraYaw,
                     cameraPitch = cameraPitch,
                     lookat = lookat,
                     target_joint_angles = test_settings["initial_joint_angles"])
    
    # Pybullet GUI screenshot
    cameraDistance = screenshot_config["cameraDistance"]
    cameraYaw = screenshot_config["cameraYaw"]
    cameraPitch = screenshot_config["cameraPitch"]
    lookat = screenshot_config["lookat"]
    pixelWidth = screenshot_config["pixelWidth"]
    pixelHeight = screenshot_config["pixelHeight"]
    nearPlane = screenshot_config["nearPlane"]
    farPlane = screenshot_config["farPlane"]
    fov = screenshot_config["fov"]
    viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=lookat,
                                                     distance=cameraDistance, 
                                                     yaw=cameraYaw, 
                                                     pitch=cameraPitch,
                                                     roll = 0,
                                                     upAxisIndex = 2)
    projectionMatrix = p.computeProjectionMatrixFOV(fov, pixelWidth / pixelHeight, nearPlane, farPlane)

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
    save_every = test_settings["save_every"]
    num_data = (T-1)//step_every + 1
    times = np.arange(0, num_data)*step_every*dt
    mean_errs = np.zeros((num_data,2), dtype = np.float32)
    orientation_errs = np.zeros((num_data,2), dtype = np.float32)
    observer_errs = np.zeros(num_data, dtype = np.float32)
    manipulability = np.zeros(num_data, dtype = np.float32)
    vels = np.zeros((num_data,9), dtype = np.float32)
    joint_values = np.zeros((num_data,9), dtype = np.float32)
    cvxpylayer_computation_time = np.zeros(num_data, dtype = np.float32)
    cbf_value = np.zeros(num_data, dtype = np.float32)
    target_center = np.zeros((num_data,3), dtype = np.float32)
    camera_position = np.zeros((num_data,3), dtype = np.float32)

    # Obstacle line
    obstacle_config = test_settings["obstacle_config"]
    # p.addUserDebugLine(lineFromXYZ = obstacle_config["lineFromXYZ"],
    #                    lineToXYZ = obstacle_config["lineToXYZ"],
    #                    lineColorRGB = obstacle_config["lineColorRGB"],
    #                    lineWidth = obstacle_config["lineWidth"],
    #                    lifeTime = obstacle_config["lifeTime"])
    obstacle_corner_in_world = np.array([np.array(obstacle_config["lineFromXYZ"]),
                                         np.array(obstacle_config["lineToXYZ"]),
                                         np.array(obstacle_config["lineToXYZ"])+[0.2,0,0],
                                         np.array(obstacle_config["lineFromXYZ"])+[0.2,0,0]], dtype=np.float32)
    obstacle_corner_in_world = np.hstack((obstacle_corner_in_world, np.ones((obstacle_corner_in_world.shape[0],1), dtype=np.float32)))
    
    obstacle_quat = p.getQuaternionFromEuler([0, 0, 0])
    p.resetBasePositionAndOrientation(env.obstacle_ID, (np.array(obstacle_config["lineFromXYZ"]) + np.array(obstacle_config["lineToXYZ"]))/2.0, obstacle_quat)
    p.changeVisualShape(env.obstacle_ID, -1, rgbaColor=[1., 0.87, 0.68, obstacle_config["obstacle_alpha"]])

    # Differential optimization layer
    nv = 2
    nc_target = optimization_config["n_cons_target"]
    nc_obstacle = optimization_config["n_cons_obstacle"]
    kappa = optimization_config["exp_coef"]

    _p = cp.Variable(nv)
    _alpha = cp.Variable(1)

    _A_target = cp.Parameter((nc_target, nv))
    _b_target = cp.Parameter(nc_target)
    _A_obstacle = cp.Parameter((nc_obstacle, nv))
    _b_obstacle = cp.Parameter(nc_obstacle)

    obj = cp.Minimize(_alpha)
    cons = [cp.sum(cp.exp(kappa*(_A_target @ _p - _b_target))) <= nc_target*_alpha, cp.sum(cp.exp(kappa*(_A_obstacle @ _p - _b_obstacle))) <= nc_obstacle*_alpha]
    problem = cp.Problem(obj, cons)
    assert problem.is_dpp()

    cvxpylayer = CvxpyLayer(problem, parameters=[_A_target, _b_target, _A_obstacle, _b_obstacle], variables=[_alpha, _p], gp=False)

    # CBF-QP
    n = 6
    n_eq = 0
    n_in = 1
    cbf_qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
    cbf_qp.init(np.eye(6), None, None, None, None, None, None)
    cbf_qp.settings.eps_abs = 1.0e-9
    cbf_qp.solve()

    # Joint limits QP
    n = 9
    n_eq = 0
    n_in = 9
    joint_limits_qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
    joint_limits_qp.init(np.eye(9), None, None, None, None, None, None)
    joint_limits_qp.settings.eps_abs = 1.0e-9
    joint_limits_qp.solve()

    # Adjust mean and variance target to 3 points
    num_points = 3
    x0 = intrinsic_matrix[0, 2]
    y0 = intrinsic_matrix[1, 2]
    old_mean_target = np.array([x0,y0], dtype=np.float32)
    old_variance_target = np.array(controller_config["variance_target"], dtype=np.float32)
    desired_coords = np.array([[-1, -1],
                                [ 1, -1],
                                [ 1,  1],
                                [-1,  1]], dtype = np.float32)
    desired_coords = desired_coords*np.sqrt(old_variance_target) + old_mean_target
    mean_target = np.mean(desired_coords[0:num_points,:], axis=0)
    variance_target = np.var(desired_coords[0:num_points,:], axis = 0)

    # Display trajs from last
    if test_settings["visualize_target_traj_from_last"] == 1:
        results_dir_keep = "{}/results_diff_opt_3_points/exp_{:03d}_w_cbf".format(str(Path(__file__).parent.parent), exp_num)
        summary = load_dict("{}/summary.npy".format(results_dir_keep))
        p.addUserDebugPoints(summary["target_center"], [[1.,0.,0.]]*len(summary["target_center"]), pointSize=13, lifeTime=0.01)

    if test_settings["visualize_camera_traj_from_last"] == 1:
        results_dir_keep = "{}/results_diff_opt_3_points/exp_{:03d}_w_cbf".format(str(Path(__file__).parent.parent), exp_num)
        summary = load_dict("{}/summary.npy".format(results_dir_keep))
        p.addUserDebugPoints(summary["camera_position"], [[0.,0.,1.]]*len(summary["camera_position"]), pointSize=13, lifeTime=0.01)
    
    for i in range(T):
        augular_velocity = apriltag_config["augular_velocity"]
        apriltag_angle = augular_velocity*i*dt + apriltag_config["offset_angle"]
        apriltag_radius = apriltag_config["apriltag_radius"]
        apriltag_position = np.array([apriltag_radius*np.cos(apriltag_angle), apriltag_radius*np.sin(apriltag_angle), 0]) + apriltag_config["center_position"]
        april_tag_quat = p.getQuaternionFromEuler([np.pi / 2, 0, np.pi / 2 + 0.1*i*dt])
        p.resetBasePositionAndOrientation(env.april_tag_ID, apriltag_position, april_tag_quat)
        apriltag_speed_in_world = (apriltag_position - last_apriltag_position)/dt
        last_apriltag_position = apriltag_position
 
        if i % step_every == 0:
            info = env.get_image()
            rgb_opengl = info["rgb"]
            img = 0.2125*rgb_opengl[:,:,0] + 0.7154*rgb_opengl[:,:,1] + 0.0721*rgb_opengl[:,:,2]
            img = img.astype(np.uint8)
            detector = apriltag.Detector()
            result = detector.detect(img)

            # Break if no corner detected
            if len(result) == 0:
                break
            corners = result[0].corners

            # Break if corner out of image
            if_in = np.zeros(4, dtype=bool)
            for ii in range(len(corners)):
                x, y = corners[ii,:]
                if_in[ii] = point_in_image(x, y, camera_config["width"], camera_config["height"])

            if np.sum(if_in) != len(corners):
                break

            # Use cv2 window to display image
            if enable_gui_camera_data == 0:
                cv2.namedWindow("RGB")
                BGR = cv2.cvtColor(rgb_opengl, cv2.COLOR_RGB2BGR)
                cv2.imshow("RGB", BGR)
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

            # Record
            target_center[i//step_every,:] = np.mean(coord_in_world[:,0:3], axis=0)
            camera_position[i//step_every,:] = np.reshape(info["P_CAMERA"],(1,3))

            # Draw apritag vertices in world
            if test_settings["visualize_target_traj"] == 1:
                # colors = [[0.5,0.5,0.5],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
                # p.addUserDebugPoints(coord_in_world[:,0:3], colors, pointSize=5, lifeTime=0.01)
                p.addUserDebugPoints([np.mean(coord_in_world[:,0:3], axis=0)], [[1.,0.,0.]], pointSize=5, lifeTime=0.01)
            if test_settings["visualize_camera_traj"] == 1:
                p.addUserDebugPoints(np.reshape(info["P_CAMERA"],(1,3)), [[0.,0.,1.]], pointSize=5, lifeTime=0.01)

            # Draw obstacle vertices in world
            # colors = [[0.5,0.5,0.5],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
            # p.addUserDebugPoints(obstacle_corner_in_world[:,0:3], colors, pointSize=5, lifeTime=0.01)

            # Compute image jaccobian due to camera speed
            J_image_cam = np.zeros((2*corners.shape[0], 6), dtype=np.float32)
            fx = intrinsic_matrix[0, 0]
            fy = intrinsic_matrix[1, 1]
            for ii in range(len(corners)):
                J_image_cam[2*ii:2*ii+2] = one_point_image_jacobian(coord_in_cam[ii], fx, fy)

            # Compute desired pixel velocity (mean)
            mean_gain = np.diag(controller_config["mean_gain"])
            J_mean = 1/num_points*np.tile(np.eye(2), num_points)
            error_mean = np.mean(corners[0:num_points,:], axis=0) - mean_target
            xd_yd_mean = - LA.pinv(J_mean) @ mean_gain @ error_mean

            # Compute desired pixel velocity (variance)
            variance_gain = np.diag(controller_config["variance_gain"])
            J_variance = np.tile(- np.diag(np.mean(corners[0:num_points,:], axis=0)), num_points)
            J_variance[0,0::2] += corners[0:num_points,0]
            J_variance[1,1::2] += corners[0:num_points,1]
            J_variance = 2/num_points*J_variance
            error_variance = np.var(corners[0:num_points,:], axis = 0) - variance_target
            xd_yd_variance = - LA.pinv(J_variance) @ variance_gain @ error_variance

            # # Compute desired pixel velocity (orientation A)
            # orientation_gain = np.diag([controller_config["horizontal_gain"], controller_config["vertical_gain"]])
            # J_orientation = np.zeros((2, 2*len(corners)), dtype=np.float32)
            # tmp1 = np.arange(0,len(corners),2,dtype=np.int_)
            # tmp2 = np.arange(1,len(corners),2,dtype=np.int_)
            # tmp = np.zeros(len(corners), dtype=np.int_)
            # tmp[0::2] = tmp2
            # tmp[1::2] = tmp1
            # J_orientation[0,1::2] += corners[:,1] - corners[tmp,1]
            # J_orientation[1,0::2] += corners[:,0] - np.flip(corners[:,0])
            # J_orientation = 2*J_orientation
            # J_orientation = J_orientation[:,0:2*num_points]
            # J_orientation[1,0] = 0
            # J_orientation[0,5] = 0
            # error_orientation = np.sum(J_orientation**2, axis=1)/8.0
            # xd_yd_orientation = - LA.pinv(J_orientation) @ orientation_gain @ error_orientation

            # Compute desired pixel velocity (orientation B)
            orientation_gain = np.diag([controller_config["horizontal_gain"], controller_config["vertical_gain"]])
            J_orientation = np.zeros((2, 2*len(corners)), dtype=np.float32)
            J_orientation[0,1] = 1
            J_orientation[0,3] = -1
            J_orientation[1,2] = 1
            J_orientation[1,4] = -1
            error_orientation = J_orientation @ corners.flatten()
            J_orientation = J_orientation[:,0:2*num_points]
            xd_yd_orientation = - LA.pinv(J_orientation) @ orientation_gain @ error_orientation

            # Speed contribution due to movement of the apriltag
            apriltag_speed_in_cam = info["R_CAMERA"] @ apriltag_speed_in_world
            d_true = -J_image_cam[:,0:3] @ apriltag_speed_in_cam

            # Update the observer
            d_hat = observer_gain @ np.reshape(corners, (2*len(corners),)) - epsilon

            # Map to the camera speed expressed in the camera frame
            # xd_yd_mean and xd_yd_variance does not interfere each other, see Gans TRO 2011
            null_mean = np.eye(2*num_points, dtype=np.float32) - LA.pinv(J_mean) @ J_mean
            null_variance = np.eye(2*num_points, dtype=np.float32) - LA.pinv(J_variance) @ J_variance
            xd_yd = xd_yd_mean + xd_yd_variance + null_mean @ null_variance @ xd_yd_orientation
            J_active = J_image_cam[0:2*num_points]
            if observer_config["active"] == 1:
                speeds_in_cam_desired = J_active.T @ LA.inv(J_active @ J_active.T + 1*np.eye(2*num_points)) @ (xd_yd - d_hat[0:2*num_points])
            else:
                speeds_in_cam_desired = J_active.T @ LA.inv(J_active @ J_active.T + 1*np.eye(2*num_points)) @ xd_yd

            # Map obstacle vertices to image
            obstacle_corner_in_cam = obstacle_corner_in_world @ LA.inv(H).T 
            obstacle_corner_in_image = obstacle_corner_in_cam[:,0:3] @ intrinsic_matrix.T
            obstacle_corner_in_image = obstacle_corner_in_image/obstacle_corner_in_image[:,-1][:,np.newaxis]
            obstacle_corner_in_image = obstacle_corner_in_image[:,0:2]

            if CBF_config["active"] == 1:
                # Construct CBF and its constraint
                target_coords = torch.tensor(corners, dtype=torch.float32, requires_grad=True)
                x_target = target_coords[:,0]
                y_target = target_coords[:,1]
                A_target_val = torch.vstack((-y_target+torch.roll(y_target,-1), -torch.roll(x_target,-1)+x_target)).T
                b_target_val = -y_target*torch.roll(x_target,-1) + torch.roll(y_target,-1)*x_target

                obstacle_coords = torch.tensor(obstacle_corner_in_image, dtype=torch.float32, requires_grad=True)
                x_obstacle = obstacle_coords[:,0]
                y_obstacle = obstacle_coords[:,1]
                A_obstacle_val = torch.vstack((-y_obstacle+torch.roll(y_obstacle,-1), -torch.roll(x_obstacle,-1)+x_obstacle)).T
                b_obstacle_val = -y_obstacle*torch.roll(x_obstacle,-1) + torch.roll(y_obstacle,-1)*x_obstacle

                # check if the obstacle is far to avoid numerical instability
                A_obstacle_np = A_obstacle_val.detach().numpy()
                b_obstacle_np = b_obstacle_val.detach().numpy()
                tmp = kappa*(corners @ A_obstacle_np.T - b_obstacle_np)
                tmp = np.max(tmp, axis=1)
     
                if np.min(tmp) <= CBF_config["threshold_lb"] and np.max(tmp) <= CBF_config["threshold_ub"]:
                    time1 = time.time()
                    alpha_sol, p_sol = cvxpylayer(A_target_val, b_target_val, A_obstacle_val, b_obstacle_val, 
                                                  solver_args=optimization_config["solver_args"])
                    CBF = alpha_sol.detach().numpy() - CBF_config["scaling_lb"]
                    # print(alpha_sol, p_sol)
                    print(CBF)
                    alpha_sol.backward()
                    time2 = time.time()
                    cvxpylayer_computation_time[i//step_every] = time2-time1
                    cbf_value[i//step_every] = CBF

                    target_coords_grad = np.array(target_coords.grad)
                    obstacle_coords_grad = np.array(obstacle_coords.grad)
                    grad_CBF = np.hstack((target_coords_grad.reshape(-1), obstacle_coords_grad.reshape(-1)))
                    grad_CBF_disturbance = target_coords_grad.reshape(-1)

                    actuation_matrix = np.zeros((len(grad_CBF), 6), dtype=np.float32)
                    actuation_matrix[0:2*len(target_coords_grad)] = J_image_cam
                    for ii in range(len(obstacle_coords_grad)):
                        actuation_matrix[2*ii+2*len(target_coords_grad):2*ii+2+2*len(target_coords_grad)] = one_point_image_jacobian(obstacle_corner_in_cam[ii,0:3], fx, fy)
                    
                    # CBF_QP
                    A_CBF = (grad_CBF @ actuation_matrix)[np.newaxis, :]
                    lb_CBF = -CBF_config["barrier_alpha"]*CBF + CBF_config["compensation"]\
                            - grad_CBF_disturbance @ d_hat
                    H = np.eye(6)
                    g = -speeds_in_cam_desired
                    cbf_qp.settings.initial_guess = (
                        proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
                    )
                    cbf_qp.update(g=g, C=A_CBF, l=lb_CBF)
                    cbf_qp.settings.eps_abs = 1.0e-9
                    cbf_qp.solve()

                    speeds_in_cam = cbf_qp.results.x

                else: 
                    speeds_in_cam = speeds_in_cam_desired
                    print("CBF skipped")
                    cvxpylayer_computation_time[i//step_every] = None
                    cbf_value[i//step_every] = None
            else: 
                speeds_in_cam = speeds_in_cam_desired
                if CBF_config["cbf_value_record"] == 1:
                    # Construct CBF and its constraint
                    target_coords = torch.tensor(corners, dtype=torch.float32, requires_grad=True)
                    x_target = target_coords[:,0]
                    y_target = target_coords[:,1]
                    A_target_val = torch.vstack((-y_target+torch.roll(y_target,-1), -torch.roll(x_target,-1)+x_target)).T
                    b_target_val = -y_target*torch.roll(x_target,-1) + torch.roll(y_target,-1)*x_target

                    obstacle_coords = torch.tensor(obstacle_corner_in_image, dtype=torch.float32, requires_grad=True)
                    x_obstacle = obstacle_coords[:,0]
                    y_obstacle = obstacle_coords[:,1]
                    A_obstacle_val = torch.vstack((-y_obstacle+torch.roll(y_obstacle,-1), -torch.roll(x_obstacle,-1)+x_obstacle)).T
                    b_obstacle_val = -y_obstacle*torch.roll(x_obstacle,-1) + torch.roll(y_obstacle,-1)*x_obstacle

                    # check if the obstacle is far to avoid numerical instability
                    A_obstacle_np = A_obstacle_val.detach().numpy()
                    b_obstacle_np = b_obstacle_val.detach().numpy()
                    tmp = kappa*(corners @ A_obstacle_np.T - b_obstacle_np)
                    tmp = np.max(tmp, axis=1)

                    if np.min(tmp) <= CBF_config["threshold_lb"] and np.max(tmp) <= CBF_config["threshold_ub"]:
                        time1 = time.time()
                        alpha_sol, p_sol = cvxpylayer(A_target_val, b_target_val, A_obstacle_val, b_obstacle_val, 
                                                    solver_args=optimization_config["solver_args"])
                        CBF = alpha_sol.detach().numpy() - CBF_config["scaling_lb"]
                        # print(alpha_sol, p_sol)
                        print(CBF)
                        alpha_sol.backward()
                        time2 = time.time()
                        cvxpylayer_computation_time[i//step_every] = time2-time1
                        cbf_value[i//step_every] = CBF
                    else:
                        print("CBF value skipped")
                        cvxpylayer_computation_time[i//step_every] = None
                        cbf_value[i//step_every] = None

            # Transform the speed back to the world frame
            v_in_cam = speeds_in_cam[0:3]
            omega_in_cam = speeds_in_cam[3:6]
            R_cam_to_world = info["R_CAMERA"]
            v_in_world = R_cam_to_world @ v_in_cam
            S_in_world = R_cam_to_world @ skew(omega_in_cam) @ R_cam_to_world.T
            omega_in_world = skew_to_vector(S_in_world)
            u_desired = np.hstack((v_in_world, omega_in_world))

            # Secondary objective: encourage the joints to stay in the middle of joint limits
            W = np.diag(-1.0/(joint_ub-joint_lb)**2) /len(joint_lb)
            q = info["q"]
            grad_joint = controller_config["joint_limit_gain"]* W @ (q - (joint_ub+joint_lb)/2)
            
            # Map the desired camera speed to joint velocities
            J_camera = info["J_CAMERA"]
            pinv_J_camera = LA.pinv(J_camera)
            dq_nominal = pinv_J_camera @ u_desired + (np.eye(9) - pinv_J_camera @ J_camera) @ grad_joint

            # QP-for joint limits
            q = info["q"]
            H = np.eye(9)
            g = - dq_nominal
            C = np.eye(9)*dt*step_every
            joint_limits_qp.settings.initial_guess = (
                    proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
                )
            joint_limits_qp.update(H=H, g=g, l=joint_lb - q, u=joint_ub - q, C=C)
            joint_limits_qp.settings.eps_abs = 1.0e-9
            joint_limits_qp.solve()
            vel = joint_limits_qp.results.x
            vel[-2:] = 0

            if test_settings["save_screeshot"] == 1 and i % save_every == 0:
                screenshot = p.getCameraImage(pixelWidth,
                                              pixelHeight,
                                              viewMatrix=viewMatrix,
                                              projectionMatrix=projectionMatrix,
                                              flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL
                                              )
                screenshot = np.reshape(screenshot[2], (pixelHeight, pixelWidth, 4))
                screenshot = Image.fromarray(screenshot)
                screenshot = screenshot.convert('RGB')
                screenshot.save(results_dir+'/screenshot_'+'{:04d}.{}'.format(i, test_settings["image_save_format"]))

            if test_settings["save_rgb"] == 1 and i % save_every == 0:
                rgb_opengl = info["rgb"]
                rgbim = Image.fromarray(rgb_opengl)
                rgbim_no_alpha = rgbim.convert('RGB')
                rgbim_no_alpha.save(results_dir+'/rgb_'+'{:04d}.{}'.format(i, test_settings["image_save_format"]))
                
            if test_settings["save_depth"] == 1 and i % save_every == 0:
                plt.imsave(results_dir+'/depth_'+'{:04d}.{}'.format(i, test_settings["image_save_format"]), depth_opengl)

            if test_settings["save_detection"] == 1 and i % save_every == 0:
                for ii in range(len(corners)):
                    x, y = corners[ii,:]
                    img = cv2.circle(img, (int(x),int(y)), radius=5, color=(0, 0, 255), thickness=-1)
                for ii in range(len(obstacle_corner_in_image)):
                    x, y = obstacle_corner_in_image[ii,:]
                    img = cv2.circle(img, (int(x),int(y)), radius=5, color=(0, 0, 255), thickness=-1)

                cv2.imwrite(results_dir+'/detect_'+'{:04d}.{}'.format(i, test_settings["image_save_format"]), img)
            
            if test_settings["save_scaling_function"] == 1 and i % save_every == 0:
                blank_img = np.ones_like(img)*255

                A_target_val = A_target_val.detach().numpy()
                b_target_val = b_target_val.detach().numpy()
                A_obstacle_val = A_obstacle_val.detach().numpy()
                b_obstacle_val = b_obstacle_val.detach().numpy()

                for ii in range(camera_config["width"]):
                    for jj in range(camera_config["height"]):
                        pp = np.array([ii,jj])
                        if np.sum(np.exp(kappa * (A_target_val @ pp - b_target_val))) <= 4:
                            x, y = pp
                            img = cv2.circle(blank_img, (int(x),int(y)), radius=1, color=(0, 0, 255), thickness=-1)
                        if np.sum(np.exp(kappa * (A_obstacle_val @ pp - b_obstacle_val))) <= 4:
                            x, y = pp
                            img = cv2.circle(blank_img, (int(x),int(y)), radius=1, color=(0, 0, 255), thickness=-1)
                cv2.imwrite(results_dir+'/scaling_functions_'+'{:04d}.{}'.format(i, test_settings["image_save_format"]), img)

            # Step the simulation
            if test_settings["zero_vel"] == 1:
                vel = np.zeros_like(vel)
            info = env.step(vel, return_image=False)

            # Records
            mean_errs[i//step_every,:] = error_mean
            orientation_errs[i//step_every,:] = error_orientation
            observer_errs[i//step_every] = LA.norm(d_hat - d_true)
            manipulability[i//step_every] = np.sqrt(LA.det(J_camera @ J_camera.T))
            vels[i//step_every] = vel
            joint_values[i//step_every] = q

            # Step the observer
            epsilon += step_every * dt * observer_gain @ (J_image_cam @speeds_in_cam + d_hat)


        else:
            info = env.step(vel)

        q, dq = info["q"], info["dq"]

        if i % 500 == 0:
            print("Iter {:.2e}".format(i))
        
    env.close()

    summary = {'times': times,
            'mean_errs': mean_errs,
            'orientation_errs': orientation_errs,
            'observer_errs': observer_errs,
            'manipulability': manipulability,
            'joint_vels': vels,
            'joint_values': joint_values,
            'cvxpylayer_computation_time': cvxpylayer_computation_time,
            'cbf_value': cbf_value,
            'stop_ind': i//step_every,
            'target_center': target_center,
            'camera_position': camera_position}
    print("==> Saving summary ...")
    save_dict(summary, os.path.join(results_dir, "summary.npy"))
    
    print("==> Drawing plots ...")
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi, frameon=True)
    plt.plot(times, mean_errs[:,0], label="x")
    plt.plot(times, mean_errs[:,1], label="y")
    plt.legend()
    plt.axhline(y = 0.0, color = 'black', linestyle = 'dotted')
    plt.savefig(os.path.join(results_dir, 'plot_mean_errs.png'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi, frameon=True)
    plt.plot(times, orientation_errs[:,0], label="x")
    plt.plot(times, orientation_errs[:,1], label="y")
    plt.legend()
    plt.axhline(y = 0.0, color = 'black', linestyle = 'dotted')
    plt.savefig(os.path.join(results_dir, 'plot_orientation_errs.png'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi, frameon=True)
    plt.plot(times, observer_errs, label="observer_errs")
    plt.legend()
    plt.axhline(y = 0.0, color = 'black', linestyle = 'dotted')
    plt.savefig(os.path.join(results_dir, 'plot_observer_errs.png'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi, frameon=True)
    plt.plot(times, manipulability, label="manipulability")
    plt.legend()
    plt.axhline(y = 0.0, color = 'black', linestyle = 'dotted')
    plt.savefig(os.path.join(results_dir, 'plot_manipulability.png'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi, frameon=True)
    plt.plot(times, cvxpylayer_computation_time, label="cvxpylayer_computation_time")
    plt.legend()
    plt.axhline(y = 0.0, color = 'black', linestyle = 'dotted')
    plt.savefig(os.path.join(results_dir, 'cvxpylayer_computation_time.png'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi, frameon=True)
    plt.plot(times, cbf_value, label="cbf_value")
    plt.legend()
    plt.axhline(y = 0.0, color = 'black', linestyle = 'dotted')
    plt.savefig(os.path.join(results_dir, 'cbf_value.png'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi, frameon=True)
    plt.plot(times, vels)
    plt.legend()
    plt.axhline(y = 0.0, color = 'black', linestyle = 'dotted')
    plt.savefig(os.path.join(results_dir, 'plot_joint_vel.png'))
    plt.close(fig)

    for i in range(joint_values.shape[1]):
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi, frameon=True)
        plt.plot(times, joint_values[:,i], label="{}".format(i))
        plt.legend()
        plt.axhline(y = joint_ub[i], color = 'black', linestyle = 'dotted')
        plt.axhline(y = joint_lb[i], color = 'black', linestyle = 'dotted')
        plt.savefig(os.path.join(results_dir, 'plot_joint_value_{}.png'.format(i)))
        plt.close(fig)

if __name__ == "__main__":
    main()
