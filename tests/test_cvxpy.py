import cvxpy as cp
import numpy as np

m,n = 50,10
A = np.random.randn(m,n)
b = np.random.randn(m)
x = cp.Variable(n)
lam = 0.1

f = cp.sum_squares(A@x - b) + lam*cp.norm1(x)
cons = [x >= 0]
cp.Problem(cp.Minimize(f), cons).solve(verbose=False, eps_abs=1e-8, eps_rel=1e-8)
print(np.round(x.value,5))
print(np.round(cons[0].dual_value, 5))