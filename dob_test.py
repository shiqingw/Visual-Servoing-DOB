import numpy as np
import matplotlib.pyplot as plt

a = -0.1
dt = 0.01
N = 10000
x = 1
t = 0
disturbance = lambda t: np.sin(t)
l = 10
eps = 0
d_hat = l*x - eps

times= [t]
x_list = [x]
d_list = [disturbance(t)]
d_hat_list = [d_hat]


for i in range(N):
    dx = a*x + disturbance(t)
    x = x + dx*dt
    t = t + dt
    deps = l*(a*x + d_hat)
    eps = eps+ deps*dt
    d_hat = l*x - eps
    times.append(t)
    x_list.append(x)
    d_list.append(disturbance(t))
    d_hat_list.append(d_hat)

plt.plot(times, d_list, label="true_d")
plt.plot(times, d_hat_list, label="d_hat")
plt.legend()
plt.show()