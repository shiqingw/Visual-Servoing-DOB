import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D

def get_line(point1, point2):
    line = Line2D([point1[0], point2[0]],
              [point1[1], point2[1]], linewidth=4, color='black', linestyle='--')
    return line

N = 3000
d = np.linspace(-2,7,N)
x,y = np.meshgrid(d,d)
kappa = 5

aprox_exp = lambda x: np.exp(kappa*x)
aprox_triangle_exp = lambda x,y : aprox_exp(-x) + aprox_exp(-y) + aprox_exp(x+y-5)

fig = plt.figure(figsize=(5, 5), dpi=200)
ax = fig.add_subplot()
cmap = colors.ListedColormap(['white', 'tab:blue'])
bounds=[0,1,10]
norm = colors.BoundaryNorm(bounds, cmap.N)

points = np.array([[0,0],[5,0],[0,5]])
ax.add_line(get_line(points[0], points[1]))
ax.add_line(get_line(points[1], points[2]))
ax.add_line(get_line(points[2], points[0]))

im = plt.imshow((aprox_triangle_exp(x,y)<=3).astype(int) , 
                extent=(x.min(),x.max(),y.min(),y.max()),origin="lower",cmap=cmap, alpha=0.8)

tickfontsize = 20
plt.xticks(fontsize=tickfontsize)
plt.yticks(fontsize=tickfontsize)
# plt.show()
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.savefig("triangle_keq{}.png".format(kappa))



####### Square #######
# N = 3000
# d = np.linspace(-2,7,N)
# x,y = np.meshgrid(d,d)
# kappa = 5

# aprox_exp = lambda x: np.exp(kappa*x)
# aprox_square_exp = lambda x,y : aprox_exp(-x) + aprox_exp(-y) + aprox_exp(x-5) + aprox_exp(y-5)

# fig = plt.figure(figsize=(5, 5), dpi=100)
# ax = fig.add_subplot()
# cmap = colors.ListedColormap(['white', 'tab:blue'])
# bounds=[0,1,10]
# norm = colors.BoundaryNorm(bounds, cmap.N)
# im = plt.imshow((aprox_square_exp(x,y)<=4).astype(int) , 
#                 extent=(x.min(),x.max(),y.min(),y.max()),origin="lower", cmap=cmap)

# points = np.array([[0,0],[5,0],[5,5],[0,5]])
# ax.add_line(get_line(points[0], points[1]))
# ax.add_line(get_line(points[1], points[2]))
# ax.add_line(get_line(points[2], points[3]))
# ax.add_line(get_line(points[3], points[0]))

# tickfontsize = 20
# plt.xticks(fontsize=tickfontsize)
# plt.yticks(fontsize=tickfontsize)
# # plt.show()
# plt.gca().set_aspect('equal')
# plt.tight_layout()
# plt.savefig("square_keq{}.png".format(kappa))