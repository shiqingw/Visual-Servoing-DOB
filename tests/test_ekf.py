# In this file, we test functionality of the class EKF_IBVS

import numpy as np
import matplotlib.pyplot as plt
import time
from ekf.ekf_ibvs import EKF_IBVS

# Define the parameters needed by EKF_IBVS

# Camera intrinsic parameters
fx = 1.0
fy = 1.0
cx = 0.0
cy = 0.0

# Initial values 
dt = 0.01
num_points = 4
x0 = np.random.rand(9,num_points)
P0 = np.eye(9)
Q = np.eye(9)
R = np.eye(3)

# Create an instance of EKF_IBVS
ekf = EKF_IBVS(dt, num_points, x0, P0, Q, R, fx, fy, cx, cy)

for i in range(10):
    time1 = time.time()
    ekf.predict(np.random.rand(6))
    ekf.update(np.random.rand(3,num_points))
    time2 = time.time()
    print(time2-time1)


# x_pred = ekf.get_predicted_state()
# P_pre = ekf.get_predicted_covariance()
# print(x_pred, P_pre)





