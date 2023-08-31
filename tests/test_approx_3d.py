import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

def get_line(point1, point2):
    line = Line3D([point1[0], point2[0]],
              [point1[1], point2[1]],
              [point1[2], point2[2]], linewidth=4, color='black', linestyle='--')
    return line

kappa = 5
n_ineq = 6
aprox_exp = lambda x: np.exp(kappa*x)
aprox_cube = lambda x,y,z : aprox_exp(-x) + aprox_exp(x-1) + aprox_exp(-y) + aprox_exp(y-1) + aprox_exp(-z) + aprox_exp(z-1)

n = 200
x = np.linspace(-0.5, 1.5, n)
y = np.linspace(-0.5, 1.5, n)
z = np.linspace(-0.5, 1.5, n)
X,Y,Z = np.meshgrid(x,y,z)
in_inds = aprox_cube(X,Y,Z) <= n_ineq

# Create a 3D plot
fig = plt.figure(figsize=(5, 5), dpi=100)
ax = fig.add_subplot(111, projection='3d')

# Plot the edges
vertices = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0], 
                   [0,0,1],[1,0,1],[1,1,1],[0,1,1]])
for i in range(4):
    ax.add_line(get_line(vertices[i], vertices[(i+1)%4]))
    ax.add_line(get_line(vertices[i+4], vertices[((i+1)%4)+4]))
    ax.add_line(get_line(vertices[i], vertices[i+4]))

# Plot the approximation
points = np.hstack((X[in_inds].reshape(-1,1), Y[in_inds].reshape(-1,1), Z[in_inds].reshape(-1,1)))
hull = ConvexHull(points)
polygons = []
for simplex in hull.simplices:
    vertices = points[simplex]
    polygons.append(vertices)
    
ax.add_collection3d(Poly3DCollection(polygons, facecolors='tab:blue', alpha=0.5))

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-0.3, 1.3)
ax.set_ylim(-0.3, 1.3)
ax.set_zlim(-0.3, 1.3)

# Set the view angle (elevation and azimuth)
ax.view_init(elev=20, azim=30)  # Adjust these angles

# Show the plot
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.savefig("cube_keq{}.png".format(kappa))

####################################################################################################

# kappa = 5
# n_ineq = 4
# aprox_exp = lambda x: np.exp(kappa*x)
# aprox_cube = lambda x,y,z : aprox_exp(-x) + aprox_exp(-y) + aprox_exp(-z) + aprox_exp(x+y+z-1)

# n = 200
# x = np.linspace(-0.5, 1.5, n)
# y = np.linspace(-0.5, 1.5, n)
# z = np.linspace(-0.5, 1.5, n)
# X,Y,Z = np.meshgrid(x,y,z)
# in_inds = aprox_cube(X,Y,Z) <= n_ineq

# # Create a 3D plot
# fig = plt.figure(figsize=(5, 5), dpi=100)
# ax = fig.add_subplot(111, projection='3d')

# # Plot the edges
# vertices = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
# ax.add_line(get_line(vertices[0], vertices[1]))
# ax.add_line(get_line(vertices[1], vertices[2]))
# ax.add_line(get_line(vertices[0], vertices[2]))
# ax.add_line(get_line(vertices[0], vertices[3]))
# ax.add_line(get_line(vertices[1], vertices[3]))
# ax.add_line(get_line(vertices[2], vertices[3]))

# # Plot the approximation
# points = np.hstack((X[in_inds].reshape(-1,1), Y[in_inds].reshape(-1,1), Z[in_inds].reshape(-1,1)))
# hull = ConvexHull(points)
# polygons = []
# for simplex in hull.simplices:
#     vertices = points[simplex]
#     polygons.append(vertices)
    
# ax.add_collection3d(Poly3DCollection(polygons, facecolors='tab:blue', alpha=0.5))

# # Add labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_xlim(-0.3, 1.3)
# ax.set_ylim(-0.3, 1.3)
# ax.set_zlim(-0.3, 1.3)

# # Set the view angle (elevation and azimuth)
# ax.view_init(elev=20, azim=30)  # Adjust these angles

# # Show the plot
# plt.gca().set_aspect('equal')
# plt.tight_layout()
# plt.savefig("tetrahedron_keq{}.png".format(kappa))