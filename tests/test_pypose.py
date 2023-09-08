import torch
import pypose as pp
import numpy as np

# SE3 vector [tx, ty, tz, qx, qy, qz, qw]
SE3_true = np.array([1,1,1,0,0,0,1])

# Create 40 measurements using gaussian noise
N = 100
noise = np.random.normal(0, 0, size=(N,7))
SE3_measurements = SE3_true + noise
# Normalize quaternions
for i in range(N):
    SE3_measurements[i,3:] /= np.linalg.norm(SE3_measurements[i,3:])

# Convert to pypose SE3 tensors
SE3_measurements = pp.SE3(SE3_measurements)
# Compute mean of SE3 measurements
first_measurement = SE3_measurements[0,:] 
deltas = torch.zeros_like(SE3_measurements)
for i in range(N):
    deltas[i,:] = first_measurement.Inv() @ SE3_measurements[i,:]
deltas = pp.SE3(deltas)
deltas_in_log = pp.Log(deltas)
mean_deltas_in_log = pp.se3(deltas_in_log.mean(dim=0))
mean_delta = pp.Exp(mean_deltas_in_log)
SE3_optimal = first_measurement @ mean_delta
print("Average: ", SE3_optimal)
print("True: ", SE3_true)