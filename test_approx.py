import numpy as np
import matplotlib.pyplot as plt

###########################
# lb = -10
# ub = 10
# N = 1000
# eps = 0.1
# x = np.linspace(lb,ub,N)

# abs_x = np.abs(x)
# approx1 = np.sqrt(x**2 + eps)

# # plt.plot(x, np.sqrt(x**2+eps)+x)

# plt.plot(x, x/np.sqrt(x**2+eps)+1)

# # plt.plot((2*x**2+eps)/(x**2+eps)**(3/2))
# plt.show()

###########################
# n = 100000
# xy_min = [-1, -1]
# xy_max = [6, 6]
# eps = 0.1
# data = np.random.uniform(low=xy_min, high=xy_max, size=(n,2))
# k = 3
# # print(data)

# aprox_abs = lambda x: np.sqrt(eps+x**2)

# triangle = lambda x,y : np.abs(-x)+(-x) + np.abs(-y)+(-y) + np.abs(x+y-5)+(x+y-5)
# aprox_triangle = lambda x,y : aprox_abs(-x)+(-x) + aprox_abs(-y)+(-y) + aprox_abs(x+y-5)+(x+y-5)

# exact = np.zeros(n, dtype=bool)
# aprox = np.zeros(n, dtype=bool)

# for i in range(n):
#     x, y = data[i,:]
#     if triangle(x,y) <= 0.0001:
#         exact[i] = True
#     if aprox_triangle(x,y) <= k*np.sqrt(eps):
#         aprox[i] = True

# print(np.sum(exact))
# print(np.sum(aprox))




# plt.scatter(data[aprox,0], data[aprox,1], label='aprox')

# # plt.scatter(data[exact,0], data[exact,1], label='exact')

# x = np.linspace(xy_min[0], xy_max[0],1000)
# y = -x+5

# plt.plot(x,y, color='r', linestyle='-')
# plt.axhline(y=0.0, color='r', linestyle='-')
# plt.axvline(x=0.0, color='r', linestyle='-')

# plt.legend()
# plt.xlim(xy_min[0], xy_max[0])
# plt.ylim(xy_min[1], xy_max[1])
# plt.show()

###########################

n = 100000
xy_min = [-1, -1]
xy_max = [6, 6]
eps = 1e-3
data = np.random.uniform(low=xy_min, high=xy_max, size=(n,2))
k = 4
# print(data)

aprox_abs = lambda x: np.sqrt(eps+x**2)

square = lambda x,y : np.abs(-x)+(-x) + np.abs(-y)+(-y) + np.abs(x-5)+(x-5) + np.abs(y-5)+(y-5)
aprox_square = lambda x,y : aprox_abs(-x)+(-x) + aprox_abs(-y)+(-y) + aprox_abs(x-5)+(x-5) + aprox_abs(y-5)+(y-5)

exact = np.zeros(n, dtype=bool)
aprox = np.zeros(n, dtype=bool)

for i in range(n):
    x, y = data[i,:]
    if square(x,y) <= 0.0001:
        exact[i] = True
    if aprox_square(x,y) <= k*np.sqrt(eps):
        aprox[i] = True

print(np.sum(exact))
print(np.sum(aprox))


plt.scatter(data[aprox,0], data[aprox,1], label='aprox')

# plt.scatter(data[exact,0], data[exact,1], label='exact')


plt.axhline(y=0.0, color='r', linestyle='-')
plt.axvline(x=0.0, color='r', linestyle='-')
plt.axhline(y=5.0, color='r', linestyle='-')
plt.axvline(x=5.0, color='r', linestyle='-')

plt.legend()
plt.xlim(xy_min[0], xy_max[0])
plt.ylim(xy_min[1], xy_max[1])
plt.show()