import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from sys import platform
if platform == "darwin":
    time1 = time.time()
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    time2 = time.time()
    print("==> Initializing Julia took {} seconds".format(time2-time1))

print("==> Importing differentiable collision")
time1 = time.time()
try:
    from differentiable_collision_utils.dc_cbf import DifferentiableCollisionCBF
except:
    from differentiable_collision_utils.dc_cbf import DifferentiableCollisionCBF
time2 = time.time()
print("==> Importing differentiable collision took {} seconds".format(time2-time1))