import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


# N = 3000
# d = np.linspace(-2,7,N)
# x,y = np.meshgrid(d,d)
# kappa = 5

# aprox_exp = lambda x: np.exp(kappa*x)
# aprox_triangle_exp = lambda x,y : aprox_exp(-x) + aprox_exp(-y) + aprox_exp(x+y-5)

# plt.figure(figsize=(5, 5), dpi=200)
# cmap = colors.ListedColormap(['white', 'tab:blue'])
# bounds=[0,1,10]
# norm = colors.BoundaryNorm(bounds, cmap.N)

# im = plt.imshow((aprox_triangle_exp(x,y)<=3).astype(int) , 
#                 extent=(x.min(),x.max(),y.min(),y.max()),origin="lower",cmap=cmap)

# linewidth=4
# plt.plot(d,-d+5, color='tab:red', linestyle='--', linewidth=linewidth)
# plt.axhline(y=0.0, color='tab:red', linestyle='--', linewidth=linewidth)
# plt.axvline(x=0.0, color='tab:red', linestyle='--', linewidth=linewidth)

# tickfontsize = 20
# plt.xticks(fontsize=tickfontsize)
# plt.yticks(fontsize=tickfontsize)
# # plt.show()
# plt.gca().set_aspect('equal')
# plt.tight_layout()
# plt.savefig("triangle_keq{}.png".format(kappa))



####### Square #######
N = 3000
d = np.linspace(-2,7,N)
x,y = np.meshgrid(d,d)
kappa = 5

aprox_exp = lambda x: np.exp(kappa*x)
aprox_square_exp = lambda x,y : aprox_exp(-x) + aprox_exp(-y) + aprox_exp(x-5) + aprox_exp(y-5)

plt.figure(figsize=(5, 5), dpi=100)
cmap = colors.ListedColormap(['white', 'tab:blue'])
bounds=[0,1,10]
norm = colors.BoundaryNorm(bounds, cmap.N)
im = plt.imshow((aprox_square_exp(x,y)<=4).astype(int) , 
                extent=(x.min(),x.max(),y.min(),y.max()),origin="lower", cmap=cmap)

linewidth=4
plt.axhline(y=0.0, color='tab:red', linestyle='--', linewidth=linewidth)
plt.axvline(x=0.0, color='tab:red', linestyle='--', linewidth=linewidth)
plt.axhline(y=5.0, color='tab:red', linestyle='--', linewidth=linewidth)
plt.axvline(x=5.0, color='tab:red', linestyle='--', linewidth=linewidth)

tickfontsize = 20
plt.xticks(fontsize=tickfontsize)
plt.yticks(fontsize=tickfontsize)
# plt.show()
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.savefig("square_keq{}.png".format(kappa))