from scipy.spatial.transform import Rotation as R
import numpy as np

R1 = R.from_euler('z', 90, degrees=True)
print(R1.as_matrix())

R2 = R.from_euler('x', 2, degrees=True)
print(R2.as_matrix())

R_total = R1.as_matrix() @ R2.as_matrix()
r =  R.from_matrix(R_total)
angles = r.as_euler("xyz",degrees=False)
print(angles)