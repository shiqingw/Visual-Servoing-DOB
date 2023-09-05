import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D

def get_line(point1, point2):
    line = Line2D([point1[0], point2[0]],
              [point1[1], point2[1]], linewidth=4, color='black', linestyle='--')
    return line

# N = 3000
# d = np.linspace(-2,7,N)
# X,Y = np.meshgrid(d,d)
# kappa = 5
# n_ineq = 4

# aprox_exp = lambda x: np.exp(kappa*x)
# aprox_triangle_exp = lambda x,y : aprox_exp(-x) + aprox_exp(-y) + aprox_exp(x+y-5)

# fig = plt.figure(figsize=(5, 5), dpi=200)
# ax = fig.add_subplot()

# points = np.array([[0,0],[5,0],[0,5]])
# ax.add_line(get_line(points[0], points[1]))
# ax.add_line(get_line(points[1], points[2]))
# ax.add_line(get_line(points[2], points[0]))

# in_inds = aprox_triangle_exp(X,Y) <= n_ineq
# points = np.hstack((X[in_inds].reshape(-1,1), Y[in_inds].reshape(-1,1)))
# hull = ConvexHull(points)
# polygon = plt.Polygon(points[hull.vertices], closed=True, fill="tab:blue", edgecolor=None, alpha=0.7, zorder=2)
# ax.add_patch(polygon)

# tickfontsize = 20
# plt.xticks(fontsize=tickfontsize)
# plt.yticks(fontsize=tickfontsize)
# plt.xlim(-2,7)
# plt.ylim(-2,7)

# plt.gca().set_aspect('equal')
# plt.tight_layout()
# plt.savefig("triangle_keq{}.png".format(kappa))

############################################################################################################

N = 3000
d = np.linspace(-2,7,N)
X,Y = np.meshgrid(d,d)
kappa = 2
n_ineq = 4

aprox_exp = lambda x: np.exp(kappa*x)
aprox_square_exp = lambda x,y : aprox_exp(-x) + aprox_exp(-y) + aprox_exp(x-5) + aprox_exp(y-5)

fig = plt.figure(figsize=(5, 5), dpi=200)
ax = fig.add_subplot()

points = np.array([[0,0],[5,0],[5,5],[0,5]])
ax.add_line(get_line(points[0], points[1]))
ax.add_line(get_line(points[1], points[2]))
ax.add_line(get_line(points[2], points[3]))
ax.add_line(get_line(points[3], points[0]))

in_inds = aprox_square_exp(X,Y) <= n_ineq
points = np.hstack((X[in_inds].reshape(-1,1), Y[in_inds].reshape(-1,1)))
hull = ConvexHull(points)
polygon = plt.Polygon(points[hull.vertices], closed=True, fill="tab:blue", edgecolor=None, alpha=0.7, zorder=2)
ax.add_patch(polygon)

tickfontsize = 20
plt.xticks(fontsize=tickfontsize)
plt.yticks(fontsize=tickfontsize)
plt.xlim(-2,7)
plt.ylim(-2,7)

plt.gca().set_aspect('equal')
plt.tight_layout()
plt.savefig("square_keq{}.png".format(kappa))