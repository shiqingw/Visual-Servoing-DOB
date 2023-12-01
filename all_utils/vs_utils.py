import numpy as np
from scipy.spatial.transform import Rotation as R

def axis_angle_from_rot_mat(rot_mat):
    rotation = R.from_matrix(rot_mat)
    axis_angle = rotation.as_rotvec()
    # angle = LA.norm(axis_angle)
    # axis = axis_angle / angle
    
    return axis_angle

def get_homogeneous_transformation(p, R):
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

def one_point_image_jacobian_normalized(x, y, Z):
    """
    x, y: normalized pixel coordinates (x = (x-cx)/fx, y = (y-cy)/fy)
    Z: depth of the point in camera frame
    """
    J1 = np.array([-1/Z, 0, x/Z, x*y, -(1+x**2), y], dtype=np.float32)
    J2 = np.array([0, -1/Z, y/Z, 1+y**2, -x*y, -x], dtype=np.float32)

    return np.vstack((J1, J2))

def one_point_depth_jacobian_normalized(x, y, Z):
    """
    x, y: normalized pixel coordinates (x = (x-cx)/fx, y = (y-cy)/fy)
    Z: depth of the point in camera frame
    """
    J = np.array([0, 0, -1, -y*Z, x*Z, 0], dtype=np.float32)
    return J

def normalize_one_image_point(x, y, fx, fy, cx, cy):
    x_norm = (x - cx) / fx
    y_norm = (y - cy) / fy
    return x_norm, y_norm

def denormalize_one_image_point(x_norm, y_norm, fx, fy, cx, cy):
    x = x_norm * fx + cx
    y = y_norm * fy + cy
    return x, y

def normalize_corners(corners, fx, fy, cx, cy):
    """
    corners: 2D array of shape (N, 2)
    """
    corners_norm = np.zeros_like(corners)
    corners_norm[:, 0] = (corners[:, 0] - cx) / fx
    corners_norm[:, 1] = (corners[:, 1] - cy) / fy

    return corners_norm

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