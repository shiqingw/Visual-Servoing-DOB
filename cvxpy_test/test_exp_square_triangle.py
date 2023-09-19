import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.lines import Line2D
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import time
import torch

def get_line(point1, point2):
    line = Line2D([point1[0], point2[0]],
              [point1[1], point2[1]], linewidth=4, color='black', linestyle='--')
    return line

def to_np(x):
    return x.cpu().detach().numpy()

############################################################################################################
# Hyperparameters
kappa = 1
points1 = np.array([[-1,-1],[1,-1],[1,1],[-1,1]], dtype=np.float64)
points2 = np.array([[9,-1],[11,-1],[11,1]], dtype=np.float64) + np.array([5,-5], dtype=np.float64)
dist = np.linalg.norm(np.mean(points1, axis=0) - np.mean(points2, axis=0))
plot_x_lim = [-10,30]
plot_y_lim = [-10,10]
# solver_args={"solve_method": "SCS", "eps": 1e-7, "max_iters": 2000}
solver_args={"solve_method": "ECOS", "max_iters": 1000} # Cannot use eps for ECOS

############################################################################################################
# Solve problem
nv = 2
nc1 = 4
nc2 = 3
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

time1 = time.time()
pixel_coords1 = torch.tensor(points1, requires_grad=True)
x1 = pixel_coords1[:,0]
y1 = pixel_coords1[:,1]
A1_val = -torch.vstack((y1-torch.roll(y1,-1), torch.roll(x1,-1)-x1)).T
b1_val = -y1*torch.roll(x1,-1) + torch.roll(y1,-1)*x1

time2 = time.time()
pixel_coords2 = torch.tensor(points2, requires_grad=True)
x2 = pixel_coords2[:,0]
y2 = pixel_coords2[:,1]
A2_val = -torch.vstack((y2-torch.roll(y2,-1), torch.roll(x2,-1)-x2)).T
b2_val = -y2*torch.roll(x2,-1) + torch.roll(y2,-1)*x2

time3 = time.time()
alpha_sol, p_sol = cvxpylayer(A1_val, b1_val, A2_val, b2_val, solver_args=solver_args)
alpha_sol.backward()
time4 = time.time()

print(alpha_sol, p_sol)
print(pixel_coords1.grad)
print(pixel_coords2.grad)

# print time taken in each step
print('Time taken to create A1 and b1: ', time2 - time1)
print('Time taken to create A2 and b2: ', time3 - time2)
print('Time taken to solve the problem: ', time4 - time3)

############################################################################################################
# Create plot
fig = plt.figure(figsize=(5, 5), dpi=200)
ax = fig.add_subplot()
alpha_sol = to_np(alpha_sol)
p_sol = to_np(p_sol)

############################################################################################################
# Square 1
center_position = np.sum(points1, axis=0)/4
shape_range = np.array([dist,dist])
N = 4000
cx, cy = center_position
sx, sy = shape_range
dx = np.linspace(cx-sx, cx+sx, N)
dy = np.linspace(cy-sy, cy+sy, N)
X,Y = np.meshgrid(dx,dy)
n_ineq = 4
A_val = A1_val.detach().numpy()
b_val = b1_val.detach().numpy()
aprox_exp = lambda x: np.exp(kappa* x)
aprox_square_exp = lambda x,y : aprox_exp(A_val[0,0]*x + A_val[0,1]*y - b_val[0]) + aprox_exp(A_val[1,0]*x + A_val[1,1]*y - b_val[1]) + aprox_exp(A_val[2,0]*x + A_val[2,1]*y - b_val[2]) + aprox_exp(A_val[3,0]*x + A_val[3,1]*y - b_val[3])
ax.add_line(get_line(points1[0], points1[1]))
ax.add_line(get_line(points1[1], points1[2]))
ax.add_line(get_line(points1[2], points1[3]))
ax.add_line(get_line(points1[3], points1[0]))
in_inds = aprox_square_exp(X,Y)/(n_ineq) <= alpha_sol
points = np.hstack((X[in_inds].reshape(-1,1), Y[in_inds].reshape(-1,1)))
hull = ConvexHull(points)
polygon = plt.Polygon(points[hull.vertices], closed=True, fill="tab:blue", edgecolor=None, alpha=0.7, zorder=2)
ax.add_patch(polygon)

############################################################################################################
# Square 2
center_position = np.sum(points2, axis=0)/4
shape_range = np.array([dist,dist])
N = 4000
cx, cy = center_position
sx, sy = shape_range
dx = np.linspace(cx-sx, cx+sx, N)
dy = np.linspace(cy-sy, cy+sy, N)
X,Y = np.meshgrid(dx,dy)
n_ineq = 4
A_val = A2_val.detach().numpy()
b_val = b2_val.detach().numpy()
aprox_exp = lambda x: np.exp(kappa* x)
aprox_square_exp = lambda x,y : aprox_exp(A_val[0,0]*x + A_val[0,1]*y - b_val[0]) + aprox_exp(A_val[1,0]*x + A_val[1,1]*y - b_val[1]) + aprox_exp(A_val[2,0]*x + A_val[2,1]*y - b_val[2])
ax.add_line(get_line(points2[0], points2[1]))
ax.add_line(get_line(points2[1], points2[2]))
ax.add_line(get_line(points2[2], points2[0]))
# ax.add_line(get_line(points2[3], points2[0]))
in_inds = aprox_square_exp(X,Y)/(n_ineq) <= alpha_sol
points = np.hstack((X[in_inds].reshape(-1,1), Y[in_inds].reshape(-1,1)))
hull = ConvexHull(points)
polygon = plt.Polygon(points[hull.vertices], closed=True, fill="tab:blue", edgecolor=None, alpha=0.7, zorder=2)
ax.add_patch(polygon)

############################################################################################################
# Adjust plot
plt.plot(p_sol[0], p_sol[1],'ro') 
tickfontsize = 20
plt.xticks(fontsize=tickfontsize)
plt.yticks(fontsize=tickfontsize)
# plt.xlim(plot_x_lim)
# plt.ylim(plot_y_lim)

plt.gca().set_aspect('equal')
plt.grid()
plt.tight_layout()
plt.show()