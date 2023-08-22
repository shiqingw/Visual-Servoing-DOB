import numpy as np
from sympy import Matrix, cos, diff, lambdify, simplify, sin, symbols


class StateEstimator:
    """
    State estimator, where the positional states are estimated
    using a Kalman filter, and the orientation states are estimated
    using an extended Kalman filter.
    """

    def __init__(self, dt=1e-3):
        self.dt = dt

        self.initialize_position_estimation()
        self.initialize_orientation_estimation()

    def initialize_position_estimation(self):
        self.pos_A = np.array(
            [
                [1.0, 0.0, 0.0, self.dt, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, self.dt, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, self.dt],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )

        self.pos_C = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )

        self.pos_R = np.diag([1e-6, 1e-6, 1e-6, 2e-5, 2e-5, 2e-5])
        self.pos_Q = np.eye(3) * 4e-4

    def position_estimation(self, mean_tm1, covariance_tm1, observation_t, dt=None):
        """
        Kalman Filter for Position Estimation

        mean_tm1: np.array of size (6, 1)
        covariance_tm1: np.array of size (6, 6)
        observation_t: np.array of size (3, 1)
        dt: float
        """
        if dt is None:
            dt = self.dt
            pos_A = self.pos_A
        else:
            # for position estimation
            pos_A = np.array(
                [
                    [1.0, 0.0, 0.0, dt, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, dt, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, dt],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            )

        # prediction steps
        predict_mean_t = pos_A @ mean_tm1
        predict_covariance_t = pos_A @ covariance_tm1 @ pos_A.T + self.pos_R

        # correction steps
        kalman_gain = (
            predict_covariance_t
            @ self.pos_C.T
            @ np.linalg.inv(
                self.pos_C @ predict_covariance_t @ self.pos_C.T + self.pos_Q
            )
        )
        corrected_mean_t = predict_mean_t + kalman_gain @ (
            observation_t - self.pos_C @ predict_mean_t
        )
        corrected_covariance_t = (
            np.eye(6) - kalman_gain @ self.pos_C
        ) @ predict_covariance_t

        return corrected_mean_t, corrected_covariance_t

    def initialize_orientation_estimation(self):
        # for orientation estimation
        θx, θy, θz, ωx, ωy, ωz, Δt = symbols("θx θy θz ωx ωy ωz Δt")

        euler_continuous_time_dynamics = (
            (1 / cos(θy))
            * Matrix(
                [
                    [cos(θy), sin(θx) * sin(θy), cos(θx) * sin(θy)],
                    [0, cos(θx) * cos(θy), -sin(θx) * cos(θy)],
                    [0, sin(θx), cos(θx)],
                    [cos(θy), 0, 0],
                    [0, cos(θy), 0],
                    [0, 0, cos(θy)],
                ]
            )
            @ Matrix([[ωx], [ωy], [ωz]])
        )

        euler_discrete_time_dynamics = (
            Matrix([[θx], [θy], [θz], [ωx], [ωy], [ωz]])
            + euler_continuous_time_dynamics * Δt
        )
        euler_discrete_time_dynamics = simplify(euler_discrete_time_dynamics)

        G_θx = simplify(diff(euler_discrete_time_dynamics, θx))
        G_θy = simplify(diff(euler_discrete_time_dynamics, θy))
        G_θz = simplify(diff(euler_discrete_time_dynamics, θz))
        G_ωx = simplify(diff(euler_discrete_time_dynamics, ωx))
        G_ωy = simplify(diff(euler_discrete_time_dynamics, ωy))
        G_ωz = simplify(diff(euler_discrete_time_dynamics, ωz))

        G = Matrix.hstack(G_θx, G_θy, G_θz, G_ωx, G_ωy, G_ωz)

        self.g_discrete_np = lambdify(
            [θx, θy, θz, ωx, ωy, ωz, Δt], euler_discrete_time_dynamics, "numpy"
        )
        self.G_discrete_np = lambdify([θx, θy, θz, ωx, ωy, ωz, Δt], G, "numpy")

        self.ori_C = np.array(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        self.ori_R = np.diag([1e-6, 1e-6, 1e-6, 2e-5, 2e-5, 2e-5])
        self.ori_Q = np.eye(3) * (2 * np.pi / 180) ** 2

    def orientation_estimation(self, mean_tm1, covariance_tm1, observation_t, dt=None):
        """
        Extended Kalman Filter for Orientation Estimation
        Note: the Euler angles are in the "zyx" form

        mean_tm1: np.array of size (6, 1) containing [θx, θy, θz, ωx, ωy, ωz]
        covariance_tm1: np.array of size (6, 6)
        observation_t: np.array of size (3, 1) containing [θz, θy, θx]
        dt: float
        """
        if dt is None:
            dt = self.dt

        # unpack states
        θx, θy, θz = mean_tm1[0, 0], mean_tm1[1, 0], mean_tm1[2, 0]
        ωx, ωy, ωz = mean_tm1[3, 0], mean_tm1[4, 0], mean_tm1[5, 0]

        # prediction steps
        predict_mean_t = self.g_discrete_np(θx, θy, θz, ωx, ωy, ωz, dt)
        Gt = self.G_discrete_np(θx, θy, θz, ωx, ωy, ωz, dt)
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
            np.eye(6) - kalman_gain @ self.ori_C
        ) @ predict_covariance_t

        return corrected_mean_t, corrected_covariance_t

    def get_next_position(self, mean_t, dt=None):
        if dt is None:
            dt = self.dt

        x, y, z = mean_t[0, 0], mean_t[1, 0], mean_t[2, 0]
        vx, vy, vz = mean_t[3, 0], mean_t[4, 0], mean_t[5, 0]
        next_position = np.array([[x + vx * dt], [y + vy * dt], [z + vz * dt]])

        return next_position

    def get_next_orientation(self, mean_t, dt=None):
        if dt is None:
            dt = self.dt

        θx, θy, θz = mean_t[0, 0], mean_t[1, 0], mean_t[2, 0]
        ωx, ωy, ωz = mean_t[3, 0], mean_t[4, 0], mean_t[5, 0]
        θ_tp1 = self.g_discrete_np(θx, θy, θz, ωx, ωy, ωz, dt)

        return θ_tp1
