import numpy as np
from julia import Main
from all_utils.proxsuite_utils import init_prosuite_qp
from differentiable_collision_utils.dc_utils import compute_rs_qs, get_Q_mat
import copy
from scipy.linalg import block_diag
import proxsuite

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

class DifferentiableCollisionCBF():
    def __init__(self, polygon_b_in_body, obstacle_r, obstacle_q, gamma=5.0, alpha_offset=1.03):
        self.num_obstacles = 1
        self.num_link_ellipsoids = 7
        self.gamma = gamma
        self.alpha_offset = alpha_offset

        # Create the obstacle polygon
        self.create_obstacle_polygon = Main.include("differentiable_collision_utils/create_obstacle_polygon.jl")
        if type(polygon_b_in_body) != np.ndarray:
            polygon_b_in_body = np.array(polygon_b_in_body)
        if type(obstacle_r) != np.ndarray:
            obstacle_r = np.array(obstacle_r)
        if type(obstacle_q) != np.ndarray:
            obstacle_q = np.array(obstacle_q)
        polygon_b_in_body = np.squeeze(polygon_b_in_body)
        obstacle_r = np.squeeze(obstacle_r)
        obstacle_q = np.squeeze(obstacle_q)
        self.create_obstacle_polygon(polygon_b_in_body, obstacle_r, obstacle_q)

        # Create the link ellipsoids
        self.create_arm_ellipsoid = Main.include("differentiable_collision_utils/create_arm_ellipsoid.jl")
        self.create_arm_ellipsoid()

        # Create the differentiable collision
        self.get_alpha_and_grad = Main.include("differentiable_collision_utils/get_alpha_and_grad.jl")

        self.collision_cbf_qp = init_prosuite_qp(n_v=9, n_eq=0, n_in=self.num_link_ellipsoids*self.num_obstacles)
        self.collision_cbf_qp.settings.eps_abs = 1.0e-6
        self.collision_cbf_qp.settings.max_iter = 20

        self.frame_names = [
            "LINK3_BB",
            "LINK4_BB",
            "LINK5_1_BB",
            "LINK5_2_BB",
            "LINK6_BB",
            "LINK7_BB",
            "HAND_BB",
        ]

    def filter_dq(self, dq_ref, info):
        # update link position and oriention in DifferentiableCollisions
        rs, qs_w_first = compute_rs_qs(info, self.frame_names)
        # compute α's and J's
        _alphas, Js = self.get_alpha_and_grad(rs, qs_w_first)
        # compute α's and J's
        alphas = []
        Cs = []

        for k, link in enumerate(self.frame_names):
            _Q_mat_link = get_Q_mat(info[f"q_{link}"])
            Q_mat_link = block_diag(np.eye(3), 0.5 * _Q_mat_link)

            for j in range(self.num_obstacles):
                alpha, J_link = _alphas[j][k], np.array(Js[j][k])
                alphas.append(copy.deepcopy(alpha))
                Cs.append(
                    J_link[-1, 7:][np.newaxis, :] @ Q_mat_link @ info[f"J_{link}"]
                )

        lb = -self.gamma * (np.array(alphas)[:, np.newaxis] - self.alpha_offset)
        C = np.concatenate(Cs, axis=0)

        self.collision_cbf_qp.update(H =2 * np.eye(9), g=-2 * dq_ref, C=C, l=lb)
        self.collision_cbf_qp.solve()

        dq_target = self.collision_cbf_qp.results.x

        return dq_target

    