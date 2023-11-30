import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from all_utils.cbf_utils import Sphere_Projected_CBF

cbf = Sphere_Projected_CBF(f=2, r=1)
print(cbf.evaluate(x=2/np.sqrt(3), y=0, x0=0, y0=0, Zc=2))
print(cbf.evaluate_gradient(x=2/np.sqrt(3), y=0, x0=0, y0=0, Zc=2))


R = 2/np.sqrt(3)
thetas = np.linspace(0, 2*np.pi, 10)
x = R * np.cos(thetas)
y = R * np.sin(thetas)
print(cbf.evaluate(x=x, y=y, x0=0, y0=0, Zc=2))
print(cbf.evaluate_gradient(x=x, y=y, x0=0, y0=0, Zc=2))