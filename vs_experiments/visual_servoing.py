import signal
import sys
import threading
import time
from copy import deepcopy
import argparse
import json

import apriltag
import cv2
import matplotlib
import numpy as np
import numpy.linalg as LA
import rospy
from cv_bridge import CvBridge
from PIL import Image
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image

matplotlib.use('Agg')
import sys
from pathlib import Path

import cv2
import cvxpy as cp
import matplotlib.pyplot as plt
import proxsuite
import torch
from cvxpylayers.torch import CvxpyLayer

sys.path.append(str(Path(__file__).parent.parent))

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

def signal_handler(signal, frame):
    print("\nCtrl+C received. Terminating threads...")
    # Set the exit_event to notify threads to stop
    sys.exit()

def apriltag_thread_func():
    def image_callback(data):
        global corners_global, detector_global
        try:
            # Process the image and perform AprilTag detection
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")

            # Convert the image to grayscale (required by the AprilTag detector)
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Detect AprilTags in the image
            results = detector_global.detect(gray_image)

            if len(results) > 0:
                corners_global = results[0].corners

            # rospy.sleep(0.001)
        except Exception as e:
            rospy.logerr("Error processing the image: %s", str(e))

    # Initialize the ROS subscriber
    rospy.Subscriber('/camera/infra1/image_rect_raw', Image, image_callback)
    rospy.spin()
    return 

def depth_thread_func():
    def depth_callback(data):
        global depth_data_global
        try:
            bridge = CvBridge()
            depth_data_global = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            # rospy.sleep(0.001)
        except Exception as e:
            rospy.logerr("Error processing the image: %s", str(e))

    # Initialize the ROS subscriber
    rospy.Subscriber('/camera/depth/image_rect_raw', Image, depth_callback)
    rospy.spin()
    return 

if __name__ == '__main__':
    # Register the signal handler for SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    # Choose test settings
    parser = argparse.ArgumentParser(description="Visual servoing")
    parser.add_argument('--exp_num', default=1, type=int, help="test case number")

    # Set random seed
    seed_num = 0
    np.random.seed(seed_num)

    # Load test settings and create result_dir
    args = parser.parse_args()
    exp_num = args.exp_num
    results_dir = "{}/results_real_robot/exp_{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    test_settings_path = "{}/test_settings/test_settings_{:03d}.json".format(str(Path(__file__).parent), exp_num)

    with open(test_settings_path, "r", encoding="utf8") as f:
        test_settings = json.load(f)
        
    # Various configs
    camera_config = test_settings["camera_config"]
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
    depth_scale = camera_config["depth_scale"]

    # Initialize the global variables
    detector_global = apriltag.Detector()
    corners_global = []
    depth_data_global = []

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

    # Proxsuite for inverse kinematics with joint limits
    n = 9
    n_eq = 0
    n_in = 9
    inv_kin_qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)

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

    # Create and start the apriltag thread
    rospy.init_node('apriltag_detection_node', anonymous=True)
    apriltag_thread = threading.Thread(target=apriltag_thread_func)
    apriltag_thread.start()

    # Create and start the depth thread
    depth_thread = threading.Thread(target=depth_thread_func)
    depth_thread.start()

    # Observer initialization
    observer_gain = np.diag(observer_config["gain"]*observer_config["num_points"])

    # Wait a little bit for the two threads
    print("==> Wait a little bit for the two threads...")
    time.sleep(1)

    print("==> Start the control loop")
    time_last = time.time()
    for i in range(1000):
        if len(corners_global) == 0 or len(depth_data_global) == 0: 
            time.sleep(1.0/30)
            print("==> No corners detected")
            continue

        corners = deepcopy(corners_global)
        depth_data = deepcopy(depth_data_global)

        # Initialize the observer such that d_hat = 0 at t = 0
        if i == 0:
            epsilon = observer_gain @ np.reshape(corners, (2*len(corners),))
        
        corner_depths = np.zeros([corners.shape[0],1], dtype=np.float32)
        for ii in range(len(corners)):
            x, y = corners[ii,:]
            corner_depths[ii] = depth_data[int(y), int(x)]
        corner_depths = depth_scale * corner_depths

        # Pixel coordinates to camera coordinates
        pixel_coord = np.hstack((corners, np.ones((corners.shape[0],1), dtype=np.float32)))
        pixel_coord_denomalized = pixel_coord*corner_depths
        
        coord_in_cam = pixel_coord_denomalized @ LA.inv(intrinsic_matrix.T)
        coord_in_cam = np.hstack((coord_in_cam, np.ones((coord_in_cam.shape[0],1), dtype=np.float32)))

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
        speeds_in_cam_desired = J_active.T @ LA.inv(J_active @ J_active.T + 0.1*np.eye(2*num_points)) @ (xd_yd - d_hat[0:2*num_points])
        speeds_in_cam = speeds_in_cam_desired

        # Transform the speed back to the world frame
        v_in_cam = speeds_in_cam[0:3]
        omega_in_cam = speeds_in_cam[3:6]
        R_cam_to_world = info["R_CAMERA"]
        v_in_world = R_cam_to_world @ v_in_cam
        S_in_world = R_cam_to_world @ skew(omega_in_cam) @ R_cam_to_world.T
        omega_in_world = skew_to_vector(S_in_world)
        u_desired = np.hstack((v_in_world, omega_in_world))

        # Inverse kinematic with joint limits
        q = info["q"]
        J_camera = info["J_CAMERA"]
        H = J_camera.T @ J_camera
        g = - J_camera.T @ u_desired
        time_now = time.time()
        C = np.eye(9)*(time_now-time_last)
        if i == 0:
            inv_kin_qp.init(H, g, None, None, C, joint_lb - q, joint_ub - q)
            cbf_qp.settings.eps_abs = 1.0e-9
            inv_kin_qp.solve()
        else:
            inv_kin_qp.settings.initial_guess = (
                    proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
                )
            inv_kin_qp.update(H=H, g=g, l=joint_lb - q, u=joint_ub - q)
            cbf_qp.settings.eps_abs = 1.0e-9
            inv_kin_qp.solve()
        vel = inv_kin_qp.results.x
        vel[-2:] = 0

        # Step the observer
        time_now = time.time()
        epsilon += (time_now-time_last) * observer_gain @ (J_image_cam @speeds_in_cam + d_hat)
        time_last = time_now

        time.sleep(1.0/100)
        
    print("Done")