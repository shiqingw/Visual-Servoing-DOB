import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

def to_np(x):
    return x.cpu().detach().double().numpy()

nv = 2
nc = 4
_p = cp.Variable(nv)
_p0 = cp.Parameter(nv)
_A = cp.Parameter((nc, nv))
_b = cp.Parameter(nc)


obj = cp.Minimize(cp.sum_squares(_p-_p0))
cons = [cp.sum(cp.exp(10*(_A @ _p - _b))) <= nc]
# cons = [_A @ _p - _b <= 0]
problem = cp.Problem(obj, cons)
assert problem.is_dpp()

cvxpylayer = CvxpyLayer(problem, parameters=[_p0, _A, _b], variables=[_p], gp=False)

p0_val = torch.tensor([2, 0.5], dtype=torch.float32, requires_grad=True)
pixel_coords = torch.tensor([[0, 0],
                            [0, 1],
                            [1, 1],
                            [1, 0]], dtype=torch.float32, requires_grad=True)

x = pixel_coords[:,0]
y = pixel_coords[:,1]
A_val = torch.vstack((y-torch.roll(y,-1), torch.roll(x,-1)-x)).T
b_val = y*torch.roll(x,-1) - torch.roll(y,-1)*x

p_sol, = cvxpylayer(p0_val, A_val, b_val)
optimal_value = torch.norm(p_sol-p0_val, p=2)
print(optimal_value)

optimal_value.backward()
print(pixel_coords.grad)
print(p0_val.grad)
