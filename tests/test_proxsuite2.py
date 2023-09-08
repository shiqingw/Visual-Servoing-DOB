import proxsuite
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from all_utils.proxsuite_utils import init_prosuite_qp
 
qp = init_prosuite_qp(2, 0, 0)
qp.update(H=2*np.eye(2), g=np.array([1.0, 1.0]))
qp.settings.max_iter = 1
qp.solve()
print(qp.results.x)

qp.update(H=2*np.eye(2), g=np.array([1.0, 9.0]))
qp.settings.max_iter = 0
qp.solve()
print(qp.results.x)