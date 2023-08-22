import numpy as np
from sympy import Matrix, cos, diff, lambdify, simplify, sin, symbols

class EKF_IBVS_One_Point(object):
    """EKF for IBVS with image points features. Note, the image point (x,y) should be 
        transformed to ((x-cx)/fx, (y-cy)/fy) before passing to this class, where
            fx (float): focal length in x direction
            fy (float): focal length in y direction
            cx (float): principal point in x direction
            cy (float): principal point in y direction
        Implemented based on Eq 12 in Towards Dynamic Visual Servoing for Interaction
        Control and Moving Targets, ICRA 2022.
    """
    def __init__(self, x0, P0, Q, R):   
        """Initialize EKF for IBVS.

        Args:
            x0 (np.array): initial state with normalized x and y, size 9
            P0 (np.array): initial covariance, size 9x9
            Q (np.array): process noise covariance, size 9x9
            R (np.array): measurement noise covariance, size 3x3
        
        """
        self.x = x0
        self.P = P0
        self.Q = Q
        self.R = R
        self.P_pre = P0
        self.x_pre = x0

        # Define symbolic variables
        x, y, Z, d1, d2, d3, dd1, dd2, dd3 = symbols('x y Z d1 d2 d3 dd1 dd2 dd3')
        dt, Vx, Vy, Vz, Wx, Wy, Wz = symbols('dt Vx Vy Vz Wx Wy Wz')

        # Define continuous dynamics
        dot_x_y_Z = Matrix([[-1/Z, 0, x/Z, x*y, -(1+x**2), y],
                    [0, -1/Z, y/Z, 1+y**2, -x*y, -x],
                    [0, 0, -1, -y*Z, x*Z, 0]]) @ Matrix([[Vx], [Vy], [Vz], [Wx], [Wy], [Wz]]) + Matrix([[d1], [d2], [d3]])
        dot_d = Matrix([[dd1], [dd2], [dd3]])
        dot_dot_d = Matrix([[0], [0], [0]])
        continuous_dynamics = Matrix.vstack(dot_x_y_Z, dot_d, dot_dot_d)

        # Define discrete dynamics
        discrete_dynamics = continuous_dynamics * dt + Matrix([[x], [y], [Z], [d1], [d2], [d3], [dd1], [dd2], [dd3]])
        self.discrete_dynamics = lambdify([dt, x, y, Z, d1, d2, d3, dd1, dd2, dd3, Vx, Vy, Vz, Wx, Wy, Wz], discrete_dynamics, "numpy")

        # Define process partial derivatives
        discrete_process_jacobian = simplify(discrete_dynamics.jacobian(Matrix([x, y, Z, d1, d2, d3, dd1, dd2, dd3])))
        self.discrete_process_jacobian = lambdify([dt, x, y, Z, d1, d2, d3, dd1, dd2, dd3, Vx, Vy, Vz, Wx, Wy, Wz], discrete_process_jacobian, "numpy")

        # Define measurement partial derivatives
        self.measurement_jacobian = np.eye(3,9)

    def predict(self, dt, u):
        """
        Step the EKF without updating with measurements. 
        u = [Vx Vy Vz Wx Wy Wz] is the control input, all expressed in the camera frame.
        Args:   
            dt: time interval between two frames, float
            u (np.array): control input, size 6
        """
        # Update state
        if u.ndim > 1:
            u = np.squeeze(u)

        self.x_pre = np.squeeze(self.discrete_dynamics(dt, *self.x_pre, *u))

        # Update covariance
        A = self.discrete_process_jacobian(dt, *self.x_pre, *u)
        self.P_pre = A @ self.P_pre @ A.T + self.Q * dt
    
    def update(self, z):
        """
        Update the EKF with measurements.
        z = [x y Z] is the image point feature, expressed in the camera frame.
        Args:
            z (np.array): measurement, size 3
        """
        if z.ndim > 1:
            z = np.squeeze(z)

        # Compute Kalman gain
        C = self.measurement_jacobian
        K = self.P_pre @ C.T @ np.linalg.inv(C @ self.P_pre @ C.T + self.R)
        # Update state
        tmp_x = self.x_pre + K @ (z - self.x_pre[:3])
        self.x = tmp_x
        # Update covariance
        tmp_P = (np.eye(9) - K @ C) @ self.P_pre
        self.P = tmp_P
        # Update pre state and covariance. This is for the stepping the EKF without updating with measurements.
        self.x_pre = tmp_x
        self.P_pre = tmp_P


class EKF_IBVS(object):
    """EKF for IBVS with image points features. This is a collection many EKF_IBVS_One_Point.
    """
    def __init__(self, num_points, x0, P0, Q, R):   
        """
        Args:
            num_points: number of points features
            x0: initial state with !!!NORMALIZED!!! x and y, size num_pointsx9
            P0: initial covariance, size 9x9
            Q: process noise covariance, size 9x9
            R: measurement noise covariance, size 3x3
        """
        # Check the size of x0.
        if x0.shape != (num_points, 9):
            raise ValueError('x0 should have size num_pointsx9.')

        # Create an EKF for each point feature.
        self.ekf_list = []
        for i in range(num_points):
            self.ekf_list.append(EKF_IBVS_One_Point(x0[i,:], P0, Q, R))

    def predict(self, dt, u):
        """Predict the state and covariance of each EKF.
        Args:
            u: control input, size 6x1
        """
        for ekf in self.ekf_list:
            ekf.predict(dt, u)

    def update(self, z):
        """Update the state and covariance of each EKF.
        Args:
            z: measurement with normalized x and y, size 3xnum_points
        """

        for i, ekf in enumerate(self.ekf_list):
            ekf.update(z[i,:])
    
    def get_updated_state(self):
        """Get the state of all EKF.
        Returns:
            x: state, size num_pointsx9
        """
        x = np.zeros((len(self.ekf_list),9))
        for i, ekf in enumerate(self.ekf_list):
            x[i,:] = ekf.x

        return x
    
    def get_updated_covariance(self):
        """Get the covariance of all EKF.
        Returns:
            P: covariance, size num_pointsx9x9
        """
        P = np.zeros((len(self.ekf_list), 9, 9))
        for i, ekf in enumerate(self.ekf_list):
            P[i,:,:] = ekf.P
        return P
    
    def get_predicted_state(self):
        """Get the predicted state of all EKF.
        Returns:
            x: state, size num_pointsx9
        """
        x = np.zeros((len(self.ekf_list),9))
        for i, ekf in enumerate(self.ekf_list):
            x[i,:] = ekf.x_pre

        return x
    
    def get_predicted_covariance(self):
        """Get the predicted covariance of all EKF.
        Returns:
            P: covariance, size 9x9xnum_points
        """
        P = np.zeros((len(self.ekf_list), 9, 9))
        for i, ekf in enumerate(self.ekf_list):
            P[i,:,:] = ekf.P_pre
        return P


