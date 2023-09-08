import numpy as np
import copy
from scipy.spatial.transform import Rotation

def compute_rs_qs(info, frame_names):
        # update link position and oriention in DifferentiableCollisions
        link_rs = []
        link_qs = []

        for frame_name in frame_names:
            _link_r, _link_q = get_frame_config(frame_name, info)
            _link_q = change_quat_format(_link_q)
            link_rs.append(_link_r)
            link_qs.append(_link_q)

        rs = np.concatenate(link_rs)
        qs = np.concatenate(link_qs)

        return rs, qs
    
def get_frame_config(frame_name, info):
    """
    Get the position and orientation of the link from pybullet
    and change the rotation format from pybullet to DifferentiableCollisions.jl
    """
    link_r = info[f"P_{frame_name}"]
    link_R = info[f"R_{frame_name}"] @ np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    link_q = Rotation.from_matrix(link_R).as_quat()

    return link_r, link_q

def change_quat_format(q):
    """
    change from q = [x y z w] to quat = [w x y z]
    """
    quat = np.zeros(4)
    quat[0] = q[3]
    quat[1] = q[0]
    quat[2] = q[1]
    quat[3] = q[2]

    return quat

def get_Q_mat(q):
    """
    q = [x y z w]
    """
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]

    Q = np.array([[-qx, -qy, -qz], [qw, -qz, qy], [qz, qw, -qx], [-qy, qx, qw]])

    return Q