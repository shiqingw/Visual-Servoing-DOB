from copy import deepcopy
import time
import numpy as np

a = np.random.rand(600, 800)
for i in range(10):
    start = time.time()
    b = deepcopy(a)
    print(time.time() - start)

