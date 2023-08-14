import argparse
import json
import signal
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path
import tf

import apriltag
import cvxpy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import proxsuite
import rospy
import torch
from cv_bridge import CvBridge
from cvxpylayers.torch import CvxpyLayer
from PIL import Image
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image
from RobotModel import RobotModel
from FR3Py.interfaces import FR3Real
from datetime import datetime
import os
import shutil
import pickle


sys.path.append(str(Path(__file__).parent.parent))

def get_homogenius_transformation(p, R):
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = p

    return H

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

def signal_handler(signal, frame):
    global target_thread_stop_global, obstacle_thread_stop_global
    print("==> Ctrl+C received. Terminating threads...")
    target_thread_stop_global = True
    apriltag_thread.join()
    print("==> Threads terminated...")
    sys.exit()

def tf_listener_target_thread_func():

    global target_pose_global, target_ori_global
    global target_latest_timestamp_gloabl
    global target_thread_stop_global
    global target_frame_global

    listener_target = tf.TransformListener()

    listener_target.waitForTransform(target_frame_global, "/target", rospy.Time(), rospy.Duration(4.0))

    # Main loop to continuously update the pose
    while (not rospy.is_shutdown()) and (target_thread_stop_global == False):
        try:
            (trans, quat) = listener_target.lookupTransform(target_frame_global, "/target", rospy.Time(0))
            target_pose_global = np.array(trans)
            target_ori_global = np.array(quat)
            target_latest_timestamp_gloabl = listener_target.getLatestCommonTime(target_frame_global, "/target")

            rospy.sleep(0.001)  # Adjust the sleep duration as needed
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Error occurred while retrieving TF transform.") 

def tf_listener_obstacle_thread_func():

    global obstacle_pose_global, obstacle_ori_global
    global obstacle_thread_stop_global
    global target_frame_global

    listener_obstacle = tf.TransformListener()

    listener_obstacle.waitForTransform(target_frame_global, "/obstacle", rospy.Time(), rospy.Duration(4.0))

    # Main loop to continuously update the pose
    while (not rospy.is_shutdown()) and (obstacle_thread_stop_global == False):
        try:

            (trans, quat) = listener_obstacle.lookupTransform(target_frame_global, "/obstacle", rospy.Time(0))
            obstacle_pose_global = np.array(trans)
            obstacle_ori_global = np.array(quat)

            rospy.sleep(0.001)  # Adjust the sleep duration as needed
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Error occurred while retrieving TF transform.") 


if __name__ == '__main__':
    # Register the signal handler for SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    # Choose test settings
    parser = argparse.ArgumentParser(description="Visual servoing")
    parser.add_argument('--exp_num', default=2, type=int, help="test case number")

    # Set random seed
    seed_num = 0
    np.random.seed(seed_num)

    # Load test settings and create result_dir
    args = parser.parse_args()
    exp_num = args.exp_num
    now = datetime.now()
    results_dir = "{}/results/exp_{:03d}_{}".format(str(Path(__file__).parent), exp_num, now.strftime("%Y-%m-%d-%H-%M-%S"))
    test_settings_path = "{}/test_settings/test_settings_{:03d}.json".format(str(Path(__file__).parent), exp_num)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    shutil.copy(test_settings_path, results_dir)

    with open(test_settings_path, "r", encoding="utf8") as f:
        test_settings = json.load(f)

    # Robot model
    print("==> Loading robot model...")
    pin_robot = RobotModel()
    
    # Real robot interface
    print("==> Loading real robot interface...")
    robot = FR3Real()
        
    # Various configs
    camera_config = test_settings["camera_config"]
    controller_config = test_settings["controller_config"]
    observer_config = test_settings["observer_config"]
    CBF_config = test_settings["CBF_config"]
    optimization_config = test_settings["optimization_config"]
    obstacle_config = test_settings["obstacle_config"]
    target_config = test_settings["target_config"]

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
    depth_scale = camera_config["depth_scale"]

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

    # Proxsuite for CBF-QP
    n = 6
    n_eq = 0
    n_in = 1
    cbf_qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
    # Randomly initialize the QP
    cbf_qp.init(np.eye(n), None, None, None, None, None, None)
    cbf_qp.settings.eps_abs = 1.0e-9
    cbf_qp.solve()

    # Proxsuite for inverse kinematics with joint limits
    n = 9
    n_eq = 0
    n_in = 9
    joint_limits_qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
    # Randomly initialize the QP
    joint_limits_qp.init(np.eye(n), None, None, None, None, None, None)
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

    # Initialize the global variables
    target_frame_global = camera_config["target_frame"]
    obstacle_pose_global, obstacle_ori_global, target_pose_global, target_ori_global = [], [], [], []
    target_latest_timestamp_gloabl = 0
    target_thread_stop_global = False
    obstacle_thread_stop_global = False

    # Capture obstacle world coordinates
    print("==> Turn camera toward the obstacle")
    input("==> Press Enter to if obstacle is in sight")
    print("==> Starting obstacle thread...")
    rospy.init_node('apriltag_detection_node', anonymous=True)
    obstacle_thread = threading.Thread(target=tf_listener_obstacle_thread_func)
    obstacle_thread.daemon = True
    obstacle_thread.start()

    while len(obstacle_pose_global) == 0:
        print("==> Wait a little bit for the obstacle thread...")
        time.sleep(1)

    apriltag_size = obstacle_config["apriltag_size"]
    obstacle_corners_in_obs = np.array([[-1,1,0],[1,1,0],[1,-1,0],[-1,-1,0]], dtype=np.float32)*apriltag_size/2
    obstacle_corners_in_obs = np.concatenate([obstacle_corners_in_obs, np.ones([4,1], dtype=np.float32)], axis=1)
    obstacle_pose = deepcopy(obstacle_pose_global)
    obstacle_ori = deepcopy(obstacle_ori_global)
    H_obs_to_cam = get_homogenius_transformation(obstacle_pose, R.from_quat(obstacle_ori).as_matrix())

    state = robot.get_state()
    q, dq = state['q'], state['dq']
    info = pin_robot.getInfo(q,dq)
    # H_cam_to_world = np.eye(4) # Homogeneous transformation matrix from camera to world frame
    _H = np.hstack((info["R_CAMERA"], np.reshape(info["P_CAMERA"],(3,1))))
    H_cam_to_world = np.vstack((_H, np.array([[0.0, 0.0, 0.0, 1.0]])))
    H_obs_to_world = H_cam_to_world @ H_obs_to_cam
    obstacle_corner_in_world = obstacle_corners_in_obs @ H_obs_to_world.T
    print("==> Obstacle world coordinates captured")
    print(obstacle_corner_in_world)

    # Kill obstacle thread
    obstacle_thread_stop_global = True
    obstacle_thread.join()
    print("==> Obstacle thread terminated")

    # Create and start the apriltag thread
    print("==> Creating target thread...")
    apriltag_thread = threading.Thread(target=tf_listener_target_thread_func)
    apriltag_thread.daemon = True
    apriltag_thread.start()

    # Wait a little bit for the apriltag thread
    while len(target_pose_global) == 0:
        print("==> Wait a little bit for the target thread...")
        time.sleep(1)

    # Observer initialization
    observer_gain = np.diag(observer_config["gain"]*observer_config["num_points"])

    # History
    history = {"time": [],
                "q": [],
                "dq": [],
                "corners": [],
                "corner_depths": [],
                "obstacle_corner_in_world": [],
                "obstacle_corner_in_image": [],
                "error_mean": [],
                "error_variance": [],
                "error_orientation": [],
                "cbf": [],
                "joint_vel_command":[], 
                "info":[],
                "d_hat": []
                }
    
    # Start the control loop
    print("==> Start the control loop")
    control_loop_wait_time = test_settings["control_loop_wait_time"]
    time_last = time.time()
    for i in range(100000):
        target_latest_timestamp = deepcopy(target_latest_timestamp_gloabl)
        time_now = time.time()
        print(np.abs(target_latest_timestamp.to_sec() - time_now))
        if np.abs(target_latest_timestamp.to_sec() - time_now) > 0.2:
            print("==> TF timestamp too old. Break the loop...")
            robot.send_joint_command(np.zeros(7))
            break
        dt = time_now - time_last

        # Get the current state of the robot
        state = robot.get_state()
        q, dq = state['q'], state['dq']
        info = pin_robot.getInfo(q,dq)

        # Get target corners and depths
        apriltag_size = target_config["apriltag_size"]
        target_corners_in_target = np.array([[-1,1,0],[1,1,0],[1,-1,0],[-1,-1,0]], dtype=np.float32)*apriltag_size/2
        target_corners_in_target = np.concatenate([target_corners_in_target, np.ones([4,1], dtype=np.float32)], axis=1)
        target_pose = deepcopy(target_pose_global)
        target_ori = deepcopy(target_ori_global)
        H_target_to_cam = get_homogenius_transformation(target_pose, R.from_quat(target_ori).as_matrix())
        # H_cam_to_world = np.eye(4) # Homogeneous transformation matrix from camera to world frame
        _H = np.hstack((info["R_CAMERA"], np.reshape(info["P_CAMERA"],(3,1))))
        H_cam_to_world = np.vstack((_H, np.array([[0.0, 0.0, 0.0, 1.0]])))
        coord_in_cam = target_corners_in_target @ H_target_to_cam.T
        corner_depths = (coord_in_cam[:,2]).reshape(-1,1)
        pixel_coord_denomalized = coord_in_cam[:,0:3] @ intrinsic_matrix.T
        pixel_coord = pixel_coord_denomalized/corner_depths
        corners = pixel_coord[:,0:2]
        # print(corners)

        # Initialize the observer such that d_hat = 0 at t = 0
        if not('epsilon' in locals()):
            epsilon = observer_gain @ np.reshape(corners, (2*len(corners),))

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
        J_orientation = J_orientation[:,0:2*num_points]
        J_orientation[1,0] = 0
        J_orientation[0,5] = 0
        error_orientation = np.sum(J_orientation**2, axis=1)/8.0
        xd_yd_orientation = - LA.pinv(J_orientation) @ orientation_gain @ error_orientation

        # Update the observer
        d_hat = observer_gain @ np.reshape(corners, (2*len(corners),)) - epsilon

        # Compute the desired speed in camera frame
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
        _H = np.hstack((info["R_CAMERA"], np.reshape(info["P_CAMERA"],(3,1))))
        H = np.vstack((_H, np.array([[0.0, 0.0, 0.0, 1.0]]))) # Homogeneous transformation matrix from camera to world frame
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
                # print("Time for CBF: ", time2-time1)

                target_coords_grad = np.array(target_coords.grad)
                obstacle_coords_grad = np.array(obstacle_coords.grad)
                grad_CBF = np.hstack((target_coords_grad.reshape(-1), obstacle_coords_grad.reshape(-1)))
                grad_CBF_disturbance = target_coords_grad.reshape(-1)

                actuation_matrix = np.zeros((len(grad_CBF), 6), dtype=np.float32)
                actuation_matrix[0:2*len(target_coords_grad)] = J_image_cam
                for ii in range(len(obstacle_coords_grad)):
                    actuation_matrix[2*ii+2*len(target_coords_grad):2*ii+2+2*len(target_coords_grad)] = one_point_image_jacobian(obstacle_corner_in_cam[ii,0:3], fx, fy)
                
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
                CBF = 0
                print("CBF active but skipped")
        else: 
            speeds_in_cam = speeds_in_cam_desired

        if np.any(np.isnan(d_hat)) or np.any(np.isnan(speeds_in_cam)):
            break

        # Transform the speed back to the world frame
        v_in_cam = speeds_in_cam[0:3]
        omega_in_cam = speeds_in_cam[3:6]
        R_cam_to_world = info["R_CAMERA"]
        # R_cam_to_world = np.eye(3)
        v_in_world = R_cam_to_world @ v_in_cam
        S_in_world = R_cam_to_world @ skew(omega_in_cam) @ R_cam_to_world.T
        omega_in_world = skew_to_vector(S_in_world)
        u_desired = np.hstack((v_in_world, omega_in_world))

        # Secondary objective: encourage the joints to stay in the middle of the joint limits
        W = np.diag(-1.0/(joint_ub-joint_lb)**2)/len(joint_lb)
        q = info["q"]
        grad_joint = controller_config["joint_limit_gain"]* W @ (q - (joint_ub+joint_lb)/2)

        # Map the desired camera speed to joint velocities
        J_camera = info["J_CAMERA"]
        pinv_J_camera = LA.pinv(J_camera)
        dq_nominal = pinv_J_camera @ u_desired + (np.eye(9) - pinv_J_camera @ J_camera) @ grad_joint

        # Inverse kinematic with joint limits
        q = info["q"]
        H = np.eye(9)
        g = - dq_nominal
        C = np.eye(9)*control_loop_wait_time
        joint_limits_qp.settings.initial_guess = (
                proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
            )
        joint_limits_qp.update(H=H, g=g, l=joint_lb - q, u=joint_ub - q, C=C)
        joint_limits_qp.settings.eps_abs = 1.0e-9
        joint_limits_qp.solve()
        vel = joint_limits_qp.results.x
        vel[-2:] = 0

        # Robot velocity control
        vel = np.clip(vel, -np.pi, np.pi)
        robot.send_joint_command(vel[:7])

        # Step the observer
        epsilon += dt * observer_gain @ (J_image_cam @speeds_in_cam + d_hat)
        time_last = time_now
        
        # Record data to history
        history["time"].append(time_now)
        history["q"].append(q)
        history["dq"].append(dq)
        history["corners"].append(corners)
        history["corner_depths"].append(corner_depths)
        history["obstacle_corner_in_world"].append(obstacle_corner_in_world)
        history["obstacle_corner_in_image"].append(obstacle_corner_in_cam)
        history["error_mean"].append(error_mean)
        history["error_variance"].append(error_variance)
        history["error_orientation"].append(error_orientation)
        history["joint_vel_command"].append(vel)
        history["info"].append(info)
        history["d_hat"].append(d_hat)
        if CBF_config["active"] == 1:
            history["cbf"].append(CBF)

        # Wait for the next control loop
        time.sleep(control_loop_wait_time)

    # Finished
    robot.send_joint_command(np.zeros(7))
    target_thread_stop_global = True
    apriltag_thread.join()

    # Save history data to result_dir
    with open(os.path.join(results_dir, "history.pkl"), "wb") as f:
        pickle.dump(history, f)

    print("==> Done")
