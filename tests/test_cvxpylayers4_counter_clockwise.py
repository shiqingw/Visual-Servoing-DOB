import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

def to_np(x):
    return x.cpu().detach().double().numpy()

nv = 2
nc1 = 4
nc2 = 4
kappa = 3

_p = cp.Variable(nv)
_alpha = cp.Variable(1)

_A1 = cp.Parameter((nc1, nv))
_b1 = cp.Parameter(nc1)
_A2 = cp.Parameter((nc2, nv))
_b2 = cp.Parameter(nc2)


obj = cp.Minimize(_alpha)
cons = [cp.sum(cp.exp(kappa*(_A1 @ _p - _b1))) <= nc1*_alpha, cp.sum(cp.exp(kappa*(_A2 @ _p - _b2))) <= nc2*_alpha]
problem = cp.Problem(obj, cons)
assert problem.is_dpp()

cvxpylayer = CvxpyLayer(problem, parameters=[_A1, _b1, _A2, _b2], variables=[_alpha, _p], gp=False)


pixel_coords1 = torch.tensor([[0, 0],
                            [0, 1],
                            [1, 1],
                            [1, 0]], dtype=torch.float32, requires_grad=True)

x1 = pixel_coords1[:,0]
y1 = pixel_coords1[:,1]
A1_val = torch.vstack((y1-torch.roll(y1,-1), torch.roll(x1,-1)-x1)).T
b1_val = y1*torch.roll(x1,-1) - torch.roll(y1,-1)*x1

pixel_coords2 = torch.tensor([[2, 0],
                            [2, 1],
                            [3, 1],
                            [3, 0]], dtype=torch.float32, requires_grad=True)

x2 = pixel_coords2[:,0]
y2 = pixel_coords2[:,1]
A2_val = torch.vstack((y2-torch.roll(y2,-1), torch.roll(x2,-1)-x2)).T
b2_val = y2*torch.roll(x2,-1) - torch.roll(y2,-1)*x2


alpha_sol, p_sol = cvxpylayer(A1_val, b1_val, A2_val, b2_val)

print(alpha_sol, p_sol)

alpha_sol.backward()
print(pixel_coords1.grad)
print(pixel_coords2.grad)

