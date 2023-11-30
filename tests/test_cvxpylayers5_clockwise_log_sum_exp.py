import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt

def to_np(x):
        return x.cpu().detach().double().numpy()

nv = 2
nc1 = 4
nc2 = 4
kappa = 1e-5
eps = 0.1
# solver_args={"solve_method": "SCS"}
solver_args={"solve_method": "ECOS"}

_p = cp.Variable(nv)
_alpha = cp.Variable(1, pos=True)

_A1 = cp.Parameter((nc1, nv))
_b1 = cp.Parameter(nc1)
_A2 = cp.Parameter((nc2, nv))
_b2 = cp.Parameter(nc2)


obj = cp.Minimize(_alpha)
cons = [cp.log_sum_exp(kappa*(_A1 @ _p - _b1)) <= cp.log(nc1*_alpha), cp.log_sum_exp(kappa*(_A2 @ _p - _b2)) <= cp.log(nc2*_alpha)]
# cons = [cp.sum(cp.exp(kappa*(_A1 @ _p - _b1))) <= nc1*_alpha, cp.sum(cp.exp(kappa*(_A2 @ _p - _b2))) <= nc2*_alpha]
problem = cp.Problem(obj, cons)
assert problem.is_dpp()
assert problem.is_dcp(dpp = True)

cvxpylayer = CvxpyLayer(problem, parameters=[_A1, _b1, _A2, _b2], variables=[_alpha, _p], gp=False)
solve_time = []
time_start = time.time()
M = 1000
for i in range(M):
        if i % 100 == 0:
                print(i)
        time1 = time.time()
        pixel_coords1 = torch.tensor([[580.0915, 276.8857],
                              [701.8141, 277.6409],
                              [706.8201, 390.0564],
                              [574.7156, 390.0147]], requires_grad=True)

        x1 = pixel_coords1[:,0]
        y1 = pixel_coords1[:,1]
        A1_val = -torch.vstack((y1-torch.roll(y1,-1), torch.roll(x1,-1)-x1)).T
        b1_val = -y1*torch.roll(x1,-1) - torch.roll(y1,-1)*x1
        time2 = time.time()

        pixel_coords2 = torch.tensor([[1283.8926,  103.4684],
                                [  -2.1936,  101.9336],
                                [  91.0984,  -59.8135],
                                [1190.9801,  -58.5009]], requires_grad=True)

        x2 = pixel_coords2[:,0]
        y2 = pixel_coords2[:,1]
        A2_val = -torch.vstack((y2-torch.roll(y2,-1), torch.roll(x2,-1)-x2)).T
        b2_val = -y2*torch.roll(x2,-1) - torch.roll(y2,-1)*x2
        time3 = time.time()

        alpha_sol, p_sol = cvxpylayer(A1_val, b1_val, A2_val, b2_val, solver_args=solver_args)
        # print(cvxpylayer.info)
        time4 = time.time()

        alpha_sol.backward()
        time5 = time.time()
        # print('Time taken to create A1 and b1: ', time2 - time1)
        # print('Time taken to create A2 and b2: ', time3 - time2)
        # print('Time taken to solve the problem: ', time4 - time3)
        # print('Time taken to get the gradient: ', time5 - time4)
        # print("")
        solve_time.append(time4 - time3)

time_end = time.time()
# Print average time
print('Average time: ', (time_end - time_start)/M)
plt.hist(solve_time, bins=30)
plt.show()
# print(alpha_sol, p_sol)
# print(pixel_coords1.grad)
# print(pixel_coords2.grad)

