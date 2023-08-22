import numpy as np
from sympy import Matrix, diff, eye, lambdify, simplify, symbols

from tvcbf_control.state_estimation import StateEstimator


class StateEstimatorQuaternion(StateEstimator):
    def __init__(self, dt=1e-3):
        super().__init__(dt=dt)

    def initialize_orientation_estimation(self):
        qx, qy, qz, qw, ωx, ωy, ωz, Δt = symbols("q_x q_y q_z q_w ω_x ω_y ω_z Δt")

        q = Matrix([[qw], [qx], [qy], [qz]])
        Ω = Matrix(
            [
                [0, -ωx, -ωy, -ωz],
                [ωx, 0.0, ωz, -ωy],
                [ωy, -ωz, 0.0, ωx],
                [ωz, ωy, -ωx, 0.0],
            ]
        )

        _g_q = simplify((eye(4) + 0.5 * Δt * Ω) @ q)
        g_q = Matrix([[_g_q[1]], [_g_q[2]], [_g_q[3]], [_g_q[0]]])
        g_w = Matrix([[ωx], [ωy], [ωz]])
        g = Matrix.vstack(g_q, g_w)

        G_qx = simplify(diff(g, qx))
        G_qy = simplify(diff(g, qy))
        G_qz = simplify(diff(g, qz))
        G_qw = simplify(diff(g, qw))
        G_ωx = simplify(diff(g, ωx))
        G_ωy = simplify(diff(g, ωy))
        G_ωz = simplify(diff(g, ωz))
        G = Matrix.hstack(G_qx, G_qy, G_qz, G_qw, G_ωx, G_ωy, G_ωz)

        self.g_discrete_np = lambdify([qx, qy, qz, qw, ωx, ωy, ωz, Δt], g, "numpy")
        self.G_discrete_np = lambdify([qx, qy, qz, qw, ωx, ωy, ωz, Δt], G, "numpy")

        self.ori_R = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 10.0, 10.0, 10.0])
        self.ori_Q = np.eye(4) * 0.2
        self.ori_C = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )

    def orientation_estimation(self, mean_tm1, covariance_tm1, observation_t, dt=None):
        """
        Extended Kalman Filter for Orientation Estimation
        Note: the Euler angles are in the "zyx" form

        mean_tm1: np.array of size (7, 1) containing [qx, qy, qz, qw, ωx, ωy, ωz]
        covariance_tm1: np.array of size (7, 7)
        observation_t: np.array of size (4, 1) containing [qx, qy, qz, qw]
        dt: float
        """
        if dt is None:
            dt = self.dt

        # unpack states
        qx, qy, qz, qw = mean_tm1[0, 0], mean_tm1[1, 0], mean_tm1[2, 0], mean_tm1[3, 0]
        ωx, ωy, ωz = mean_tm1[4, 0], mean_tm1[5, 0], mean_tm1[6, 0]

        # prediction steps
        predict_mean_t = self.g_discrete_np(qx, qy, qz, qw, ωx, ωy, ωz, dt)
        Gt = self.G_discrete_np(qx, qy, qz, qw, ωx, ωy, ωz, dt)
        predict_covariance_t = Gt @ covariance_tm1 @ Gt.T + self.ori_R

        # correction steps
        kalman_gain = (
            predict_covariance_t
            @ self.ori_C.T
            @ np.linalg.inv(
                self.ori_C @ predict_covariance_t @ self.ori_C.T + dt * self.ori_Q
            )
        )
        corrected_mean_t = predict_mean_t + kalman_gain @ (
            observation_t - self.ori_C @ predict_mean_t
        )
        corrected_covariance_t = (
            np.eye(7) - kalman_gain @ self.ori_C @ predict_covariance_t
        )

        # normalize the quaternion
        corrected_mean_t[:4] = corrected_mean_t[:4] / np.linalg.norm(
            corrected_mean_t[:4]
        )

        # make the qw positive
        corrected_mean_t[:4] = corrected_mean_t[:4] * (
            (corrected_mean_t[3] > 0) * 2 - 1
        )

        return corrected_mean_t, corrected_covariance_t

    def get_next_orientation(self, mean_t, dt=None):
        if dt is None:
            dt = self.dt

        # unpack states
        qx, qy, qz, qw = mean_t[0, 0], mean_t[1, 0], mean_t[2, 0], mean_t[3, 0]
        ωx, ωy, ωz = mean_t[4, 0], mean_t[5, 0], mean_t[6, 0]

        next_orientation = self.g_discrete_np(qx, qy, qz, qw, ωx, ωy, ωz, dt)

        return next_orientation
